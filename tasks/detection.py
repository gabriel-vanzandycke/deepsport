from dataclasses import dataclass
from functools import cached_property
import os
import pickle
import typing

import numpy as np
import pandas
import tensorflow as tf

from calib3d import Point2D, Point3D
from tf_layers import AvoidLocalEqualities, PeakLocalMax, ComputeElementaryMetrics

from deepsport_utilities.transforms import Transform
from deepsport_utilities.ds.instants_dataset import Ball, BallState, InstantKey
from experimentator import Callback, ChunkProcessor, ExperimentMode, build_experiment
from experimentator.tf2_experiment import TensorflowExperiment
from deepsport_utilities.utils import crop_padded

DEFAULT_THRESHOLDS = np.linspace(0,1,51) # Detection thresholds on a detection map between 0 and 1.

class HeatmapDetectionExperiment(TensorflowExperiment):
    batch_inputs_names = ["batch_target", "batch_input_image", "batch_input_image2"]
    @cached_property
    def metrics(self):
        metrics = ["TP", "FP", "TN", "FN", "topk_TP", "topk_FP", "P", "N"]
        return {**{
            name:self.chunk[name] for name in metrics if name in self.chunk
        }, **self.outputs}
    @cached_property
    def outputs(self):
        outputs = ["batch_heatmap", "topk_indices", "topk_outputs", "topk_targets"]
        return {
            name:self.chunk[name] for name in outputs if name in self.chunk
        }


def divide(num: np.ndarray, den: np.ndarray):
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den>0)

@dataclass
class ComputeMetrics(Callback):
    before = ["AuC", "GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    thresholds: typing.Tuple[int, np.ndarray, list, tuple] = DEFAULT_THRESHOLDS
    class_index: int = 0
    def on_cycle_begin(self, **_):
        self.acc = {}
    def on_batch_end(self, state, **_): # 'state' argument in R/W
        for name in ["TP", "FP", "TN", "FN"]:
            if name in state:
                value = np.sum(state[name], axis=0) # sums the batch dimension
                self.acc[name] = self.acc.setdefault(name, np.zeros_like(value)) + value
    def on_cycle_end(self, state, **_):
        TP = self.acc["TP"][self.class_index]
        FP = self.acc["FP"][self.class_index]
        TN = self.acc["TN"][self.class_index]
        FN = self.acc["FN"][self.class_index]

        data = {
            "thresholds": self.thresholds,
            "accuracy": (TP+TN)/(TP+TN+FP+FN),
            "precision": divide(TP, TP+FP),
            "recall": divide(TP, TP+FN),
            "TP rate": divide(TP, TP+FN),
            "FP rate": divide(FP, FP+TN),
        }
        state["metrics"] = pandas.DataFrame(np.vstack([data[name] for name in data]).T, columns=list(data.keys()))

@dataclass
class ComputeTopkMetrics(Callback):
    """
        Arguments:
            k  list of top-k of interest (e.g. [1,3,10] for top-1, top-3 and top-10)
    """
    before = ["AuC", "GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    k: typing.Tuple[tuple, list, np.ndarray]
    thresholds: typing.Tuple[int, np.ndarray, list, tuple] = DEFAULT_THRESHOLDS
    class_index: int = 0
    def on_cycle_begin(self, **_):
        self.acc = {}
    def on_batch_end(self, state, **_): # 'state' argument in R/W
        for name in ["topk_TP", "topk_FP", "P", "N"]:
            if name in state:
                value = np.sum(state[name], axis=0) # sums the batch dimension
                self.acc[name] = self.acc.setdefault(name, np.zeros_like(value)) + value
    def on_cycle_end(self, state, **_):
        if self.acc:
            for k in self.k:
                FP = np.sum(self.acc["topk_FP"][self.class_index, :, 0:k], axis=1)
                TP = np.sum(self.acc["topk_TP"][self.class_index, :, 0:k], axis=1)
                P = self.acc["P"][np.newaxis]
                N = self.acc["N"][np.newaxis]
                data = {
                    "thresholds": self.thresholds,
                    "FP rate": divide(FP, P + N),  # #-possible cases is the number of images
                    "TP rate": divide(TP, P),      # #-possible cases is the number of images on which there's a ball to detect
                    "precision": divide(TP, TP + FP),
                    "recall": divide(TP, P),
                }
                state[f"top{k}_metrics"] = pandas.DataFrame(np.vstack([data[name] for name in data]).T, columns=list(data.keys()))

        # TODO: handle multple classes cases (here only class index is picked and the rest is discarded)

@dataclass
class AuC(Callback):
    after = ["ComputeTopkMetrics", "ComputeMetrics", "HandleMetricsPerBallSize"]
    before = ["GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    name: str
    table_name: str
    x_label: str = "FP rate"
    y_label: str = "TP rate"
    x_lim: int = 1
    close_curve: bool = True

    def on_cycle_end(self, state, **_):
        if self.table_name not in state:
            return

        x = state[self.table_name][self.x_label][::-1]
        y = state[self.table_name][self.y_label][::-1]
        state[self.name] = self(x,y)

    def __call__(self, x, y):
        auc = 0
        for xi1, yi1, xi2, yi2 in zip(x, y, x[1:], y[1:]):
            if xi1 == xi2:
                continue
            if xi2 >= self.x_lim: # last trapezoid
                auc += (yi1+yi2)*(self.x_lim-xi1)/2
                break
            auc += (yi1+yi2)*(xi2-xi1)/2

        if self.close_curve:
            auc += (1-xi2)*yi2 # pylint: disable=undefined-loop-variable
        return auc


class ComputeKeypointsDetectionHitmap(ChunkProcessor):
    mode = ExperimentMode.EVAL | ExperimentMode.INFER
    def __init__(self, non_max_suppression_pool_size=50, threshold=DEFAULT_THRESHOLDS):
        if isinstance(threshold, np.ndarray):
            thresholds = threshold
        elif isinstance(threshold, list):
            thresholds = np.array(threshold)
        elif isinstance(threshold, float):
            thresholds = np.array([threshold])
        else:
            raise ValueError(f"Unsupported type for input argument 'threshold'. Recieved {threshold}")
        assert len(thresholds.shape) == 1, "'threshold' argument should be 1D-array (a scalar is also accepted)."

        # Saved here for 'config' property
        self.non_max_suppression_pool_size = non_max_suppression_pool_size
        self.threshold = threshold

        self.avoid_local_eq = AvoidLocalEqualities()
        self.peak_local_max = PeakLocalMax(min_distance=non_max_suppression_pool_size//2, thresholds=thresholds)

    def __call__(self, chunk):
        chunk["batch_hitmap"] = self.peak_local_max(self.avoid_local_eq(chunk["batch_heatmap"])) # B,H,W,C,T [bool]

class ComputeKeypointsDetectionMetrics(ChunkProcessor):
    mode = ExperimentMode.EVAL
    def __init__(self):
        self.compute_metric = ComputeElementaryMetrics()

    def __call__(self, chunk):
        batch_hitmap = tf.cast(chunk["batch_hitmap"], tf.int32) # B,H,W,C,T
        batch_target = tf.cast(chunk["batch_target"], tf.int32)[..., tf.newaxis] # B,H,W,C,T

        batch_metric = self.compute_metric(batch_hitmap=batch_hitmap, batch_target=batch_target)
        chunk["TP"] = batch_metric["batch_TP"] # B x K x C
        chunk["FP"] = batch_metric["batch_FP"]
        chunk["TN"] = batch_metric["batch_TN"]
        chunk["FN"] = batch_metric["batch_FN"]

class ConfidenceHitmap(ChunkProcessor):
    mode = ExperimentMode.EVAL | ExperimentMode.INFER
    def __call__(self, chunk):
        chunk["batch_confidence_hitmap"] = tf.cast(chunk["batch_hitmap"], tf.float32)*chunk["batch_heatmap"][..., tf.newaxis]

class ComputeTopK(ChunkProcessor):
    mode = ExperimentMode.EVAL | ExperimentMode.INFER
    def __init__(self, k):
        """ From a `confidence_hitmap` tensor where peaks are identified with non-zero pixels whose
            value correspnod to the peaks intensity, compute the `topk_indices` holding (x,y) positions
            and `topk_outputs` holding the intensity of the `k` highest peaks.
            Inputs:
                batch_confidence_hitmap - a (B,H,W,C,N) tensor where C is the number of keypoint types
                                          and N is the threshold dimension where only peaks above the
                                          corresponding threshold are reported.
            Outputs:
                topk_outputs - a (B,C,N,K) tensor where values along the K dimensions are sorted by
                               peak intensity.
                topk_indices - a (B,C,N,K,S) tensor where y coordinates are located in S=0 and x
                               coordinates are located in S=1.
        """
        self.k = np.max(k)
    def __call__(self, chunk):
        # Flatten hitmap to feed `top_k`
        _, H, W, C, N = [tf.shape(chunk["batch_confidence_hitmap"])[d] for d in range(5)]
        shape = [-1, C, N, H*W]
        flatten_hitmap = tf.reshape(tf.transpose(chunk["batch_confidence_hitmap"], perm=[0,3,4,1,2]), shape=shape)
        topk_values, topk_indices = tf.math.top_k(flatten_hitmap, k=self.k, sorted=True)

        chunk["topk_outputs"] = topk_values # B, C, K
        #                                          y         ,          x
        chunk["topk_indices"] = tf.stack(((topk_indices // W), (topk_indices % W)), -1) # B, C, K, D

class ComputeKeypointsTopKDetectionMetrics(ChunkProcessor):
    mode = ExperimentMode.EVAL
    def __call__(self, chunk):
        assert len(chunk["batch_target"].get_shape()) == 3 or chunk["batch_target"].get_shape()[3] == 1, \
            "Only one keypoint type is allowed. If 'batch_target' is one_hot encoded, it needs to be compressed before."
        batch_target = tf.cast(chunk["batch_target"], tf.int32)
        batch_target = batch_target[..., 0] if len(batch_target.shape) == 4 else batch_target
        chunk["topk_targets"] = tf.gather_nd(batch_target, chunk["topk_indices"], batch_dims=1)

        chunk["P"] = tf.cast(tf.reduce_any(batch_target!=0, axis=[1,2]), tf.int32)
        chunk["N"] = 1-chunk["P"]
        chunk["topk_TP"] = tf.cast(tf.cast(tf.math.cumsum(chunk["topk_targets"], axis=-1), tf.bool), tf.int32)
        chunk["topk_FP"] = tf.cast(tf.cast(chunk["topk_outputs"], tf.bool), tf.int32)-chunk["topk_targets"]

class EnlargeTarget(ChunkProcessor):
    mode = ExperimentMode.EVAL
    def __init__(self, pool_size):
        self.pool_size = pool_size
    def __call__(self, chunk):
        chunk["batch_target"] = tf.nn.max_pool2d(chunk["batch_target"][..., tf.newaxis], self.pool_size, strides=1, padding='SAME')


PIFBALL_THRESHOLD = 0.05
BALLSEG_THRESHOLD = 0.6

# class RepeatedInstantKey(InstantKey):
#     occurrence: int = 0

# @dataclass
# class BuildDetectionEvaluationSet(Callback):
#     filename: str
#     side_length: int = None
#     def on_cycle_begin(self, **_):
#         self.acc = defaultdict(defaultdict(list))
#     def on_batch_end(self, keys, batch_target, batch_input_image, batch_input_image2, batch_heatmap, topk_outputs, topk_targets, topk_indices, **_):
#         for view_key, target, image, image2, heatmap, outputs, targets, indices in zip(keys, batch_target, batch_input_image, batch_input_image2, batch_heatmap, topk_outputs, topk_targets, topk_indices):
#             detections = []
#             _, K = targets.shape
#             data[view_key.timestamp] = (heatmap, outputs, targets, indices))
#             for k in range(K):
#             self.acc[(view_key.arena_label, view_key.game_id)][view_key.timestamp].append(detections)

class DetectBalls():
    def __init__(self, dataset_folder, name, config, k, filename, detection_threshold=0, side_length=None):
        self.dataset_folder = dataset_folder
        self.detection_threshold = detection_threshold
        self.name = name
        self.filename = filename
        self.database = {}
        self.model = build_experiment(config, k=k)
        self.side_length = side_length

    def detect(self, instant):
        cameras = range(instant.num_cameras)
        offset = instant.offsets[1]
        data = {
            "batch_input_image": np.stack(instant.images),
            "batch_input_image2": np.stack([instant.all_images[(c, offset)] for c in cameras])
        }

        result = self.model.predict(data)
        B, _, _, K = result['topk_outputs'].shape
        for camera_idx in range(B):
            for i in range(K):
                y, x = np.array(result['topk_indices'][camera_idx, 0, 0, i])
                value = result['topk_outputs'][camera_idx, 0, 0, i].numpy()
                if value > self.detection_threshold:
                    calib = instant.calibs[camera_idx]
                    point = Point2D(x, y)
                    ball = Ball({
                        "origin": self.name,
                        "center": calib.project_2D_to_3D(point, Z=0),
                        "image": camera_idx,
                        "visible": True, # visible enough to have been detected by a detector
                        "value": value,
                        "state": getattr(instant, "ball_state", BallState.NONE),
                    })
                    if not calib.projects_in(ball.center):
                        continue # sanity check for detections that project behind the camera
                    ball.point = point # required to extract pseudo-annotations
                    if self.side_length is not None:
                        batch_heatmap = result['batch_heatmap'][camera_idx].numpy()
                        x_slice = slice(x-self.side_length//2, x+self.side_length//2, None)
                        y_slice = slice(y-self.side_length//2, y+self.side_length//2, None)
                        ball.heatmap = crop_padded(batch_heatmap, x_slice, y_slice, self.side_length//2+1)
                    yield ball

    def __call__(self, instant_key, instant):
        filename = os.path.join(self.dataset_folder, instant_key.arena_label, str(instant_key.game_id), self.filename)

        # Load existing database
        database = pickle.load(open(filename, 'rb')) if os.path.isfile(filename) else {}

        # Detect balls
        detections = database.get(instant_key.timestamp, [])
        detections = list(filter(lambda d: d.origin != self.name, detections)) # remove previous detections from given model
        detections.extend(self.detect(instant))
        database[instant_key.timestamp] = detections

        # Save updated database
        pickle.dump(database, open(filename, 'wb'))

        return instant


class ImportDetectionsTransform(Transform):
    def __init__(self, dataset_folder, filename, proximity_threshold=15,
                 estimate_pseudo_annotation=True, remove_true_positives=True,
                 remove_duplicates=False, transfer_true_position=False):
        self.dataset_folder = dataset_folder
        self.filename = filename
        self.proximity_threshold = proximity_threshold # pixels
        self.remove_true_positives = remove_true_positives
        self.remove_duplicates = remove_duplicates
        self.estimate_pseudo_annotation = estimate_pseudo_annotation
        self.transfer_true_position = transfer_true_position
        self.database = {}

    def extract_pseudo_annotation(self, detections: Ball, ball_state=BallState.NONE):
        camera = np.array([d.camera for d in detections])
        models = np.array([d.origin for d in detections])
        points = Point2D([d.point for d in detections]) # d.point is a shortcut saved into the detection object
        values = np.array([d.value for d in detections])

        camera_cond      = camera[np.newaxis, :] == camera[:, np.newaxis]
        corroborate_cond = models[np.newaxis, :] != models[:, np.newaxis]
        proximity_cond   = np.linalg.norm(points[:, np.newaxis, :] - points[:, :, np.newaxis], axis=0) < self.proximity_threshold

        values_matrix = values[np.newaxis, :] + values[:, np.newaxis]
        values_matrix_filtered = np.triu(camera_cond * corroborate_cond * proximity_cond * values_matrix)
        i1, i2 = np.unravel_index(values_matrix_filtered.argmax(), values_matrix_filtered.shape)
        if i1 != i2: # means two different candidate were found
            center = Point3D(np.mean([detections[i1].center, detections[i2].center], axis=0))
            return Ball({
                "origin": "pseudo-annotation",
                "center": center,
                "image": detections[i1].camera,
                "visible": True, # visible enough to have been detected by a detector
                "value": values_matrix[i1, i2],
                "state": ball_state,
            })
        return None

    def keep_unique_detections(self, calibs, detections):
        kept_detections = []
        for d in detections:
            projected = lambda detection: calibs[d.camera].project_3D_to_2D(detection.center)
            if not any(np.linalg.norm(projected(d) - projected(d2)) < self.proximity_threshold for d2 in kept_detections):
                kept_detections.append(d)
        return kept_detections

    def set_true_position(self, calibs, detections, ball):
        for d in detections:
            projected = lambda ball: calibs[d.camera].project_3D_to_2D(ball.center)
            if np.linalg.norm(projected(d) - projected(ball)) < self.proximity_threshold:
                d.center = ball.center

    def __call__(self, instant_key, instant):
        # Load database
        key = (instant_key.arena_label, instant_key.game_id)
        if key not in self.database:
            filename = self.database_path = os.path.join(self.dataset_folder, instant_key.arena_label, str(instant_key.game_id), self.filename)
            self.database[key] = pickle.load(open(filename, "rb"))
        if self.filename == "balls3d.pickle": # OLD VERSION
            detections = self.database[key].get(instant.frame_indices[0], [])
            def unpack(detection):
                point = Point2D(detection.point.y, detection.point.x) # y, x were inverted in the old version
                ball = Ball({
                    "origin": detection.model,
                    "center": instant.calibs[detection.camera_idx].project_2D_to_3D(point, Z=0),
                    "image": detection.camera_idx,
                    "visible": True, # visible enough to have been detected by a detector
                    "state": getattr(instant, "ball_state", BallState.NONE),
                    "value": detection.value
                })
                ball.point = point # required to extract pseudo-annotations
                return ball
            detections = list(map(unpack, detections))
        else:
            detections = self.database[key].get(instant_key.timestamp, [])

        # Compute pseudo-annotation from detections
        annotations = [a for a in instant.annotations if isinstance(a, Ball)]
        instant.ball = None
        if annotations:
            assert len(annotations) == 1
            instant.ball = annotations[0]
            if self.transfer_true_position:
                self.set_true_position(instant.calibs, detections, instant.ball)
        elif self.estimate_pseudo_annotation and len(detections) > 1:
            pseudo_annotation = self.extract_pseudo_annotation(detections, getattr(instant, "ball_state", BallState.NONE))
            if pseudo_annotation is not None:
                instant.annotations.extend([pseudo_annotation])
                annotations = [pseudo_annotation]
                instant.ball = pseudo_annotation

        # Remove true positives
        if self.remove_true_positives and instant.ball:
            cond = lambda detection: np.linalg.norm(detection.point - instant.calibs[detection.camera].project_3D_to_2D(instant.ball)) > self.proximity_threshold
            detections = filter(cond, detections)

        # Remove duplicates
        if self.remove_duplicates and detections:
            detections = self.keep_unique_detections(instant.calibs, detections)

        instant.detections = getattr(instant, "detections", [])
        instant.detections.extend(detections)
        return instant
