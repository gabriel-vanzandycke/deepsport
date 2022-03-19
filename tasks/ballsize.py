from collections import defaultdict
from dataclasses import dataclass
import pandas

import tensorflow as tf
import numpy as np

from calib3d import Calib, Point3D, Point2D

from experimentator.tf2_chunk_processors import ChunkProcessor
from experimentator.tf2_experiment import TensorflowExperiment
from experimentator import Callback, ExperimentMode

BALL_DIAMETER = 23


class BallSizeEstimation(TensorflowExperiment):
    batch_inputs_names = ["batch_ball_size", "batch_input_image"]
    batch_metrics_names = ["target_is_ball", "predicted_is_ball", "predicted_diameter", "regression_loss", "classification_loss"]
    batch_outputs_names = ["predicted_diameter", "predicted_is_ball"]

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
        df = pandas.DataFrame(np.vstack(list(self.acc.values())).T, columns=self.acc.keys())
        state["ball_size_metrics"] = df
        state["MADE"] = np.mean(np.abs(df['diameter_error']))
        state["MAPE"] = np.mean(np.abs(df['projection_error']))
        state["MARE"] = np.mean(np.abs(df['relative_error']))


class NamedOutputs(ChunkProcessor):
    def __call__(self, chunk):
        chunk["predicted_diameter"] = chunk["batch_logits"][...,0]
        chunk["predicted_is_ball"] = chunk["batch_logits"][...,1]

class ClassificationLoss(ChunkProcessor):
    def __call__(self, chunk):
        chunk["target_is_ball"] = tf.where(tf.math.is_nan(chunk["batch_ball_size"]), 0, 1)
        chunk["classification_loss"] = tf.keras.losses.binary_crossentropy(chunk["target_is_ball"], chunk["predicted_is_ball"], from_logits=True)

class RegressionLoss(ChunkProcessor):
    def __init__(self, delta=1.0):
        self.delta = delta # required to print config
        self.loss = tf.keras.losses.Huber(delta=delta, name='huber_loss')
    def __call__(self, chunk):
        mask = tf.math.logical_not(tf.math.is_nan(chunk["batch_ball_size"]))
        losses = self.loss(y_true=chunk["batch_ball_size"][mask], y_pred=chunk["predicted_diameter"][mask])
        chunk["regression_loss"] = tf.where(tf.math.is_nan(losses), tf.zeros_like(losses), losses)

