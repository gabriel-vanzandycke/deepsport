from collections import defaultdict
from dataclasses import dataclass
import typing

import numpy as np
import pandas
import tensorflow as tf

from calib3d import Calib, Point3D, Point2D
from deepsport_utilities.ds.instants_dataset.views_transforms import ViewRandomCropperTransform
from deepsport_utilities.transforms import Transform
from experimentator.tf2_chunk_processors import ChunkProcessor
from experimentator.tf2_experiment import TensorflowExperiment
from experimentator import Callback, ExperimentMode

from .detection import divide

BALL_DIAMETER = 23


class BallSizeEstimation(TensorflowExperiment):
    batch_inputs_names = ["batch_ball_size", "batch_input_image"]
    batch_metrics_names = ["target_is_ball", "predicted_is_ball", "predicted_diameter", "regression_loss", "classification_loss"]
    batch_outputs_names = ["predicted_diameter", "predicted_is_ball"]

class BallRandomCropperTransform(ViewRandomCropperTransform):
    def __init__(self, *args, size_min=None, size_max=None, def_min=None, def_max=None, **kwargs):
        msg = "Only one of ('size_min' and 'size_max') or ('def_min' and 'def_max') should be defined"
        if size_min is not None and size_max is not None:
            assert def_min is None and def_max is None, msg
            super().__init__(*args, size_min=size_min, size_max=size_max, **kwargs)
            self.true_size = BALL_DIAMETER
        elif def_min is not None and def_max is not None:
            assert size_min is None and size_max is None, msg
            super().__init__(*args, size_min=def_min, size_max=def_max, **kwargs)
            self.true_size = 100

    def _get_current_parameters(self, view_key, view):
        keypoints = view.calib.project_3D_to_2D(view.ball.center)
        size = float(view.calib.compute_length2D(self.true_size, view.ball.center))
        input_shape = view.calib.width, view.calib.height
        return keypoints, size, input_shape



def compute_point3D(calib: Calib, point2D: Point2D, pixel_size: float, true_size: float):
    center = Point2D(calib.Kinv@calib.rectify(point2D).H)             # center expressed in camera coordinates system
    corner = Point2D(calib.Kinv@calib.rectify(point2D+pixel_size).H)  # corner expressed in camera coordinates system
    point3D_c = Point3D(center.H * true_size / (corner.x - center.x)) # scaling in the camera coordinates system
    return calib.R.T@(point3D_c-calib.T)                              # recover real world coordinates system

def compute_projection_error(calib: Calib, point3D: Point3D, pixel_error: float):
    if calib is None:
        return np.nan
    difference = point3D - calib.C # ball to camera vector
    difference.z = 0 # set z coordinate to 0 to compute pixel_error on the projected plane
    distance = np.linalg.norm(difference, axis=0) # ball - camera distance projected on Z=0 plane
    diameter = calib.compute_length2D(BALL_DIAMETER, point3D) # pixel diameter
    return distance*pixel_error/diameter

def compute_relative_error(calib: Calib, point3D: Point3D, pixel_size: float):
    num = np.linalg.norm(point3D - compute_point3D(calib, calib.project_3D_to_2D(point3D), pixel_size, BALL_DIAMETER))
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
    def on_batch_end(self, predicted_diameter, batch_ball_size, batch_ball, batch_calib, **_):
        for target, output, ball, calib in zip(batch_ball_size, predicted_diameter, batch_ball, batch_calib):
            if np.isnan(target):
                continue

            ball = Point3D(ball)
            diameter_error = output - target
            projection_error = compute_projection_error(calib, ball, diameter_error)[0]
            relative_error = compute_relative_error(calib, ball, output)

            self.acc["true_diameter"].append(target)
            self.acc["predicted_diameter"].append(output)
            self.acc["diameter_error"].append(diameter_error)
            self.acc["projection_error"].append(projection_error)
            self.acc["relative_error"].append(relative_error)
    def on_cycle_end(self, state, **_): # state in R/W mode
        try:
            df = pandas.DataFrame(np.vstack(list(self.acc.values())).T, columns=self.acc.keys())
            state["ball_size_metrics"] = df
            state["MADE"] = np.mean(np.abs(df['diameter_error']))
            state["MAPE"] = np.mean(np.abs(df['projection_error']))
            state["MARE"] = np.mean(np.abs(df['relative_error']))
        except ValueError:
            state["ball_size_metrics"] = None
            for name in ["MADE", "MAPE", "MARE"]:
                state[name] = np.nan
@dataclass
class ComputeDetectionMetrics(Callback):
    before = ["AuC", "GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    thresholds: typing.Tuple[int, np.ndarray, list, tuple] = np.linspace(0,1,51)
    def init(self, exp):
        self.k = exp.cfg.get('k', 1)
    def on_cycle_begin(self, **_):
        self.acc = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "P": 0, "N": 0}
    def on_batch_end(self, target_is_ball, predicted_is_ball, batch_ball, batch_has_ball=None, **_):
        balls, inverse = np.unique(np.array(batch_ball), axis=0, return_inverse=True)
        for index, _ in enumerate(balls):
            indices = np.where(inverse==index)[0]

            # keep index with the largest confidence
            i = indices[np.argmax(predicted_is_ball[indices])]
            output = (predicted_is_ball[i] > self.thresholds).astype(np.uint8)
            target = target_is_ball[i]
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


class NamedOutputs(ChunkProcessor):
    def __call__(self, chunk):
        chunk["predicted_diameter"] = chunk["batch_logits"][...,0]
        chunk["predicted_is_ball"] = chunk["batch_logits"][...,1]

class ClassificationLoss(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __call__(self, chunk):
        chunk["target_is_ball"] = tf.where(tf.math.is_nan(chunk["batch_ball_size"]), 0, 1)
        chunk["classification_loss"] = tf.keras.losses.binary_crossentropy(chunk["target_is_ball"], chunk["predicted_is_ball"], from_logits=True)

class RegressionLoss(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __init__(self, delta=1.0):
        self.delta = delta # required to print config
        self.loss = tf.keras.losses.Huber(delta=delta, name='huber_loss')
    def __call__(self, chunk):
        mask = tf.math.logical_not(tf.math.is_nan(chunk["batch_ball_size"]))
        losses = self.loss(y_true=chunk["batch_ball_size"][mask], y_pred=chunk["predicted_diameter"][mask])
        chunk["regression_loss"] = tf.where(tf.math.is_nan(losses), tf.zeros_like(losses), losses)

