from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
import typing

import numpy as np
import pandas
import tensorflow as tf

from calib3d import Calib, Point3D, Point2D
from experimentator.tf2_chunk_processors import ChunkProcessor
from experimentator.tf2_experiment import TensorflowExperiment
from experimentator import Callback, ExperimentMode
from deepsport_utilities.transforms import Transform
from deepsport_utilities.ds.instants_dataset import ViewKey, View

from .detection import divide

BALL_DIAMETER = 23


class BallSizeEstimation(TensorflowExperiment):
    batch_metrics_names = ["predicted_is_ball", "predicted_height", "predicted_diameter", "regression_loss", "classification_loss", "mask_loss"]
    batch_outputs_names = ["predicted_diameter", "predicted_is_ball", "predicted_height", "predicted_mask"]
    @cached_property
    def batch_inputs_names(self):
        batch_inputs_names = ["batch_is_ball", "batch_ball_size", "batch_input_image", "epoch", "batch_ball_position"]
        if self.cfg.get('with_diff', None):
            batch_inputs_names += ["batch_input_image2"]
        if self.cfg.get('estimate_height', False):
            batch_inputs_names += ["batch_ball_height"]
        if self.cfg.get('estimate_mask', False):
            batch_inputs_names += ["batch_target"]
        return batch_inputs_names

    # def batch_generator(self, subset: Subset, *args, batch_size=None, **kwargs):
    #     if subset.type == SubsetType.EVAL or self.balancer is None:
    #         yield from super().batch_generator(subset, *args, batch_size=batch_size, **kwargs)
    #     else:
    #         batch_size = batch_size or self.batch_size
    #         keys_gen = BallStateClassification.balanced_keys_generator(subset.shuffled_keys(), self.balancer, subset.query_item)
    #         # yields pairs of (keys, data)
    #         yield from subset.batches(keys=keys_gen, batch_size=batch_size, *args, **kwargs)


    def train(self, *args, **kwargs):
        self.cfg['testing_arena_labels'] = self.cfg['dataset_splitter'].testing_arena_labels
        return super().train(*args, **kwargs)

def compute_point3D_from_diameter(calib: Calib, point2D: Point2D, pixel_size: float, true_size: float):
    point_c = Point2D(calib.Kinv@calib.rectify(point2D).H)
    v = point2D - Point2D(calib.K[0:2,2])
    side2D = point2D + Point2D(v.y, -v.x)/np.linalg.norm(v)*pixel_size
    side_c = Point2D(calib.Kinv@calib.rectify(side2D).H)
    point3D_c = Point3D(point_c.H * true_size / np.linalg.norm(point_c - side_c)) # scaling in the camera coordinates system
    point3D = calib.R.T@(point3D_c-calib.T)                              # recover real world coordinates system
    #print("error:", calib.compute_length2D(point3D, true_size)[0] - pixel_size)
    return point3D

def compute_point3D_from_height(calib: Calib, point2D: Point2D, pixel_height: float):
    # compute angle in the image space of the line between the point projected on the ground and a point 1 meter below
    point3D = calib.project_2D_to_3D(point2D, Z=0)
    point2D100 = calib.project_3D_to_2D(Point3D(point3D.x, point3D.y, 100))
    alpha = np.arctan2(point2D100.x - point2D.x, point2D100.y - point2D.y)

    # use the angle to estimate the point's projection on the ground
    shift = Point2D(pixel_height*np.sin(alpha), pixel_height*np.cos(alpha))
    floor3D = calib.project_2D_to_3D(point2D+shift, Z=0)

    point3DX = calib.project_2D_to_3D(point2D, X=floor3D.x)
    point3DY = calib.project_2D_to_3D(point2D, Y=floor3D.y)
    return (point3DX+point3DY)/2

def compute_projection_error(true_center: Point3D, pred_center: Point3D):
    difference = true_center - pred_center
    return np.linalg.norm(difference[0:2], axis=0)*np.sign(difference.z)

def compute_relative_error(calib: Calib, point3D: Point3D, pixel_size: float):
    num = np.linalg.norm(point3D - compute_point3D_from_diameter(calib, calib.project_3D_to_2D(point3D), pixel_size, BALL_DIAMETER))
    den = np.linalg.norm(point3D - calib.C)
    return num/den


@dataclass
class PrintMetricsCallback(Callback):
    after = ["ComputeDiameterError"]
    when = ExperimentMode.EVAL
    metrics: list
    def on_epoch_end(self, **state):
        print(", ".join([f"{metric}={state[metric]}" for metric in self.metrics]))


@dataclass
class ComputeDiameterError(Callback):
    before = ["GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    def on_cycle_begin(self, **_):
        self.acc = defaultdict(lambda: [])
        self.evaluation_data = []
    def on_batch_end(self, predicted_diameter, batch_ball_size, batch_ball, batch_calib, **_):
        for true_diameter, diameter, ball, calib in zip(batch_ball_size, predicted_diameter, batch_ball, batch_calib):
            if np.isnan(true_diameter):
                continue

            center = ball.center

            predicted_position = compute_point3D_from_diameter(calib, calib.project_3D_to_2D(center), diameter, BALL_DIAMETER)
            projection_error = compute_projection_error(center, predicted_position)[0]
            relative_error = compute_relative_error(calib, center, diameter)

            self.acc["true_diameter"].append(true_diameter)
            self.acc["predicted_diameter"].append(diameter)
            self.acc["diameter_error"].append(diameter - true_diameter)
            self.acc["projection_error"].append(projection_error)
            self.acc["relative_error"].append(relative_error)
            self.acc["world_error"].append(np.linalg.norm(center - predicted_position))

            self.evaluation_data.append({
                "ball": center,
                "calib": calib,
                "predicted_diameter": diameter,
            })

    def on_cycle_end(self, state, **_): # state in R/W mode
        try:
            df = pandas.DataFrame(np.vstack(list(self.acc.values())).T, columns=self.acc.keys())
            state["ball_size_metrics"] = df
            state["MADE"] = np.mean(np.abs(df['diameter_error']))
            state["MAPE"] = np.mean(np.abs(df['projection_error']))
            state["MARE"] = np.mean(np.abs(df['relative_error']))
            state["MAWE"] = np.mean(np.abs(df['world_error']))
            state["mADE"] = np.median(np.abs(df['diameter_error']))
            state["mAPE"] = np.median(np.abs(df['projection_error']))
            state["mARE"] = np.median(np.abs(df['relative_error']))
            state["mAWE"] = np.median(np.abs(df['world_error']))
            state['evaluation_data'] = self.evaluation_data
        except ValueError:
            state["ball_size_metrics"] = None
            for name in ["MADE", "MAPE", "MARE"]:
                state[name] = np.nan

@dataclass
class ComputeDetectionMetrics(Callback):
    before = ["AuC", "GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    thresholds: typing.Tuple[int, np.ndarray, list, tuple] = np.linspace(0,1,51)
    def on_cycle_begin(self, **_):
        self.acc = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "P": 0, "N": 0}
    def on_batch_end(self, batch_is_ball, predicted_is_ball, batch_ball, batch_has_ball=None, **_):
        balls, inverse = np.unique(np.array(batch_ball), axis=0, return_inverse=True)
        for index, _ in enumerate(balls):
            indices = np.where(inverse==index)[0]

            # keep index with the largest confidence
            i = indices[np.argmax(predicted_is_ball[indices])]
            output = (predicted_is_ball[i] > self.thresholds).astype(np.uint8)
            target = batch_is_ball[i]
            self.acc['TP'] +=   target   *   output
            self.acc['FP'] += (1-target) *   output
            self.acc['FN'] +=   target   * (1-output)
            self.acc['TN'] += (1-target) * (1-output)
            has_ball = batch_has_ball[i] if batch_has_ball else target
            self.acc['P']  += has_ball
            self.acc['N']  += 1 - has_ball
    def on_cycle_end(self, state, **_):
        FP = self.acc["FP"]
        TP = self.acc["TP"]
        P = np.array(self.acc["P"])[np.newaxis]
        N = np.array(self.acc["N"])[np.newaxis]
        data = {
            "thresholds": self.thresholds,
            "FP rate": divide(FP, P + N),  # #-possible cases is the number of images
            "TP rate": divide(TP, P),      # #-possible cases is the number of images on which there's a ball to detect
            "precision": divide(TP, TP + FP),
            "recall": divide(TP, P),
            }
        state["top1_metrics"] = pandas.DataFrame(np.vstack([data[name] for name in data]).T, columns=list(data.keys()))

class OffsetSupervision(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __init__(self):
        self.loss = tf.keras.losses.MeanSquaredError()
    def __call__(self, chunk):
        _, H, W, _ = chunk['batch_input'].shape
        x_true, y_true = chunk['batch_ball_position'][:,0,0] - W//2, chunk['batch_ball_position'][:,1,0] - H//2
        x_pred, y_pred = chunk['predicted_xoffset'], chunk['predicted_yoffset']
        chunk["offset_loss"] = self.loss(x_true, x_pred) + self.loss(y_true, y_pred)

class MaskSupervision(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __call__(self, chunk):
        x, y, d = chunk['predicted_xoffset'], chunk['predicted_yoffset'], chunk['predicted_diameter']
        _, H, W = chunk['batch_target'].shape
        x_range = tf.range(-W//2, W//2, dtype=tf.float32)+.5
        y_range = tf.range(-H//2, H//2, dtype=tf.float32)+.5
        X, Y = tf.meshgrid(x_range, y_range)
        predicted_mask = tf.where((X[tf.newaxis]-x[:,tf.newaxis,tf.newaxis])**2+(Y[tf.newaxis]-y[:,tf.newaxis,tf.newaxis])**2 < (d[:,tf.newaxis,tf.newaxis]/2)**2, 1, 0)
        chunk['predicted_mask'] = tf.cast(predicted_mask, tf.float32)
        mask = tf.math.logical_not(tf.math.is_nan(chunk["batch_ball_size"]))
        loss_map = tf.keras.losses.binary_crossentropy(chunk["batch_target"][...,tf.newaxis], chunk['predicted_mask'][...,tf.newaxis], False)
        loss = tf.reduce_mean(loss_map, axis=[1,2])
        chunk['mask_loss'] = tf.reduce_mean(loss[mask])

class NamedOutputs(ChunkProcessor):
    def __init__(self, input_name='batch_logits',
                 estimate_height=False,
                 estimate_presence=False,
                 estimate_mask=False):
        self.input_name = input_name
        self.estimate_presence = estimate_presence
        self.estimate_height = estimate_height
        self.estimate_mask = estimate_mask

    def __call__(self, chunk):
        i = 0
        chunk["predicted_diameter"] = chunk[self.input_name][...,i]
        if self.estimate_presence:
            i += 1
            chunk["predicted_is_ball"] = chunk[self.input_name][...,i]
        if self.estimate_height:
            i += 1
            chunk["predicted_height"] = chunk[self.input_name][...,i]
        if self.estimate_mask:
            i += 1
            chunk["predicted_xoffset"] = chunk[self.input_name][...,i]
            i += 1
            chunk["predicted_yoffset"] = chunk[self.input_name][...,i]


class IsBallClassificationLoss(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __call__(self, chunk):
        # TODO: check if binary crossentropy fits the unconfident targets
        chunk["classification_loss"] = tf.keras.losses.binary_crossentropy(y_true=chunk["batch_is_ball"], y_pred=chunk["predicted_is_ball"], from_logits=True)

class ClassificationLoss(IsBallClassificationLoss):
    pass # retrocompatibility

