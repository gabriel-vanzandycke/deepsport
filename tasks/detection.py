from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
import os
import pickle
import typing

import numpy as np
import pandas
import tensorflow as tf

from calib3d import Point2D, Point3D
from tf_layers import AvoidLocalEqualities, PeakLocalMax, ComputeElementaryMetrics#, GaussianBlur

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
    def train(self, *args, **kwargs):
        self.cfg['testing_arena_labels'] = self.cfg['dataset_splitter'].testing_arena_labels
        return super().train(*args, **kwargs)


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
class ComputeDetectionMetrics(Callback):
    origin: str = 'ballseg'
    before = ["AuC", "GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    thresholds: typing.Tuple[int, np.ndarray, list, tuple] = np.linspace(0,1,51)
    key: typing.Callable = lambda view_key: (view_key.instant_key, view_key.camera)
    def on_cycle_begin(self, **_):
        self.d_acc = defaultdict(list)
        self.t_acc = defaultdict(bool) # defaults to False

    def on_batch_end(self, keys, batch_ball, batch_ball_presence, predicted_presence, **_):
        for view_key, ball, target_presence, predicted in zip(keys, batch_ball, batch_ball_presence, predicted_presence):
            if isinstance(view_key.instant_key, InstantKey): # Keep only views from deepsport dataset for evaluation
                key = self.key(view_key)#.instant_key, view_key.camera)
                if ball.origin == self.origin:
                    self.d_acc[key].append((ball, target_presence, predicted))
                    if np.any(target_presence):
                        self.t_acc[key] = True # balls might be visible on an image despite having been annotated on another.
                elif ball.origin == 'annotation':
                    self.t_acc[key] = True

    def on_cycle_end(self, state, **_):
        keys = set(list(self.d_acc.keys()) + list(self.t_acc.keys()))
        state['detection_data'] = {k: (self.d_acc.get(k, []), self.t_acc.get(k, False)) for k in keys}
        for k in [None, 1, 2, 4, 8]:
            TP = np.zeros((len(self.thresholds), ))
            FP = np.zeros((len(self.thresholds), ))
            P = N = 0
            P_upper_bound = 0
            for key in keys:
                if zipped := self.d_acc[key]:
                    balls, target_presence, predicted_presence = zip(*zipped)
                    values = [b.value for b in balls]
                    if k is None: # Detection rate of the initial detector
                        index = np.argmax(values)
                        P_upper_bound += np.any(target_presence)
                    else: # Detection rate of the top-k strategy
                        indices = np.argsort(values)[-k:]
                        index = indices[np.argmax(np.array(predicted_presence)[indices])]
                        values = predicted_presence
                        P_upper_bound += np.any(np.array(target_presence)[indices])

                    output = (values[index] >= self.thresholds).astype(np.uint8)
                    target = target_presence[index]
                    TP +=   target   *  output
                    FP += (1-target) *  output

                has_ball = self.t_acc[key]
                P  +=   has_ball
                N  += not has_ball

            name = 'initial_TP_rate_upper_bound' if k is None else f'top{k}_TP_rate_upper_bound'
            state[name] = P_upper_bound/P if P > 0 else 0

            P = np.array(P)[np.newaxis]
            N = np.array(N)[np.newaxis]
            data = {
                "thresholds": self.thresholds,
                "FP rate": divide(FP, P + N),  # #-possible cases is the number of images
                "TP rate": divide(TP, P),      # #-possible cases is the number of images on which there's a ball to detect
                "precision": divide(TP, TP + FP),
                "recall": divide(TP, P),
            }

            name = 'initial_top1_metrics' if k is None else f'top{k}_metrics'
            state[name] = pandas.DataFrame(np.vstack([data[name] for name in data]).T, columns=list(data.keys()))




def compute_auc(x, y, close_curve=True):
    if close_curve:
        x = np.concatenate([[0], x, [1]])
        y = np.concatenate([[0], y, [y[-1]]])
    return np.trapz(y, x)

@dataclass
class AuC(Callback):
    """ Compute the Area Under the Curve (AuC) of the ROC curve defined by the
        columns `x_label` and `y_label` from the table `table_name`.
        The result is stored in the state under the key `name`.

        Arguments:
            name: key under which the result is stored in the state
            table_name: name of the table from which the data is extracted
            x_label: name of the column containing the x values
            y_label: name of the column containing the y values
            x_lim: ROC curve is clipped at x=x_lim (required for top-k metric
                when the number of detections (k) is larger than the number of
                positives samples (1).
            close_curve: if True, curve is prolungated horizontally util x=1.
    """
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
    def __init__(self, non_max_suppression_pool_size=50, fast=True, threshold=DEFAULT_THRESHOLDS):
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
        self.fast = fast

        self.avoid_local_eq = AvoidLocalEqualities() if fast else GaussianBlur(30, 7)
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
# class SaveDetectionEvaluationSet(Callback):
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



class ExtractGlimpse(ChunkProcessor):
    def __init__(self, side_length, oracle=False):
        self.oracle = oracle
        self.side_length = side_length

    def __call__(self, chunk):
        ks = range(chunk['topk_indices'].get_shape()[-2])
        if self.oracle:
            offsets = lambda _: tf.cast(tf.stack([tf.reduce_mean(tf.where(t), axis=0) for t in chunk["batch_target"]]))
            assert len(ks) == 1, "Oracle detection can only work when k=1 (as placeholders are constructed using k)"
        else:
            offsets = lambda k: tf.cast(chunk['topk_indices'][:,0,0,k,:], tf.float32) - self.side_length//2

        chunk["batch_input_image"] = tf.stack([
            tf.image.extract_glimpse(
                tf.cast(chunk['batch_input_image'], tf.float32),
                size=(self.side_length, self.side_length), offsets=offsets(k), centered=False, normalized=False, noise='zero'
            ) for k in ks
        ], 1)
        print(chunk["batch_input_image"].shape)
        raise
        #chunk["batch_input_image"] = tf.reshape(chunk["batch_input_image"], [-1, self.side_length, self.side_length, 3])

        chunk["batch_heatmap"] = tf.stack([
            tf.image.extract_glimpse(
                tf.cast(chunk['batch_heatmap'], tf.float32)[..., tf.newaxis],
                size=(self.side_length, self.side_length), offsets=offsets(k), centered=False, normalized=False, noise='zero'
            ) for k in ks
        ], 1)
        #chunk["batch_heatmap"] = tf.reshape(chunk["batch_heatmap"], [-1, self.side_length, self.side_length])



class DetectBalls():
    def __init__(self, config, name, threshold=0, side_length=None, **kwargs):
        self.exp = build_experiment(config, **kwargs)

        for cp in self.exp.chunk_processors:
            if isinstance(cp, ComputeKeypointsDetectionHitmap):
                print("Found ComputeKeypointsDetectionHitmap")
                cp.avoid_local_eq = GaussianBlur(30, 7)
                break

        self.detection_threshold = threshold
        self.name = name
        self.side_length = side_length
    def __call__(self, keys, data):
        result = self.exp.predict(data)
        B = len(data['batch_calib'])
        W = data['batch_input_image'].shape[2]//B
        _, _, _, K = result['topk_outputs'].shape
        #for b, key in enumerate(keys):
        for b in [0]:
            for i in range(K):
                y, x = np.array(result['topk_indices'][b, 0, 0, i])
                camera_index = x // W
                x = x % W
                value = float(result['topk_outputs'][b, 0, 0, i].numpy())
                if value > self.detection_threshold:
                    calib = data["batch_calib"][camera_index]
                    point = Point2D(x, y)
                    ball = Ball({
                        "origin": self.name,
                        "center": calib.project_2D_to_3D(point, Z=0),
                        "image": camera_index,
                        "visible": True, # visible enough to have been detected by a detector
                        "value": value,
                        "state": data['batch_ball_state'][b] if 'batch_ball_state' in data else BallState.NONE,
                    })
                    if not calib.projects_in(ball.center):
                        continue # sanity check for detections that project behind the camera
                    ball.point = point # required to extract pseudo-annotations
                    if self.side_length is not None: # BallSeg detection heatmap added to ball object
                        raise NotImplementedError("stitched image not implemented here yet")
                        batch_heatmap = np.uint8(np.clip(result['batch_heatmap'][b].numpy()*255, 0, 255))
                        x_slice = slice(x-self.side_length//2, x+self.side_length//2, None)
                        y_slice = slice(y-self.side_length//2, y+self.side_length//2, None)
                        ball.heatmap = crop_padded(batch_heatmap, x_slice, y_slice, self.side_length//2+1)
                    yield (keys[camera_index], i), ball


class DetectBallsFromInstants(DetectBalls):
    def __call__(self, instant_key, instant, database):
        #detections = database.get(instant_key.timestamp, [])
        #detections = list(filter(lambda d: d.origin != self.name, detections)) # remove previous detections from given model

        cameras = range(instant.num_cameras)
        offset = instant.offsets[1]
        data = {
            "batch_input_image": np.hstack(instant.images)[None],
            "batch_input_image2": np.hstack([instant.all_images[(c, offset)] for c in cameras])[None],
            "batch_calib": np.array(instant.calibs),
        }
        keys = tuple((instant_key, c) for c in cameras)
        #detections.extend([kv[1] for kv in super().__call__(keys, data)])
        detections = [kv[1] for kv in super().__call__(keys, data)]
        database[instant_key.timestamp] = detections
        return instant


class ImportDetectionsTransform(Transform):
    def __init__(self, dataset_folder, filename, proximity_threshold=15,
                 estimate_pseudo_annotation=True, remove_true_positives=True,
                 remove_duplicates=False, transfer_true_position=False,
                 force_origin=False, exclusive=True):
        self.dataset_folder = dataset_folder
        self.filename = filename
        self.proximity_threshold = proximity_threshold # pixels
        self.remove_true_positives = remove_true_positives
        self.remove_duplicates = remove_duplicates
        self.estimate_pseudo_annotation = estimate_pseudo_annotation
        self.transfer_true_position = transfer_true_position
        self.exclusive = exclusive
        self.force_origin = force_origin
        self.database = {}
        self.distances = []

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
            if not any((np.linalg.norm(projected(d) - projected(d2)) < self.proximity_threshold and d.camera == d2.camera) for d2 in kept_detections):
                kept_detections.append(d)
        return kept_detections

    def set_true_position(self, calibs, detections, ball):
        for d in detections:
            projected = lambda ball: calibs[d.camera].project_3D_to_2D(ball.center)
            proximity = np.linalg.norm(projected(d) - projected(ball))
            if proximity < self.proximity_threshold:
                self.distances.append(proximity)
                d.center = ball.center

    def __call__(self, instant_key, instant):
        # Load database
        key = (instant_key.arena_label, instant_key.game_id)
        if key not in self.database:
            filename = os.path.join(self.dataset_folder, instant_key.arena_label, str(instant_key.game_id), self.filename)
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

        # Export annotated position to detections
        if instant.ball and self.transfer_true_position:
            self.set_true_position(instant.calibs, detections, instant.ball)
        # Compute pseudo-annotation from detections
        if not instant.ball and self.estimate_pseudo_annotation and len(detections) > 1:
            pseudo_annotation = self.extract_pseudo_annotation(detections, getattr(instant, "ball_state", BallState.NONE))
            if pseudo_annotation is not None:
                instant.annotations = [pseudo_annotation]

        # Remove true positives
        if self.remove_true_positives and instant.ball:
            cond = lambda detection: np.linalg.norm(detection.point - instant.calibs[detection.camera].project_3D_to_2D(instant.ball.center)) > self.proximity_threshold
            detections = list(filter(cond, detections))

        if self.force_origin:
            for d in detections:
                d.origin = self.force_origin

        if not self.exclusive:
            detections = getattr(instant, "detections", []) + detections # existing detections take precedence

        # Remove duplicates
        if self.remove_duplicates and detections:
            detections = self.keep_unique_detections(instant.calibs, detections)

        instant.detections = detections
        return instant
