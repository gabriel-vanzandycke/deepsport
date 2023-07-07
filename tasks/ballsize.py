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
    batch_metrics_names = ["predicted_is_ball", "predicted_height", "predicted_diameter", "regression_loss", "classification_loss"]
    batch_outputs_names = ["predicted_diameter", "predicted_is_ball", "predicted_height"]
    @cached_property
    def batch_inputs_names(self):
        batch_inputs_names = ["batch_is_ball", "batch_ball_size", "batch_input_image", "epoch"]
        if self.cfg.get('with_diff', None):
            batch_inputs_names += ["batch_input_image2"]
        if self.cfg.get('predict_height', False):
            batch_inputs_names += ["batch_ball_height"]
        return batch_inputs_names

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
    def on_batch_end(self, predicted_diameter, batch_ball_size, batch_ball_position, batch_calib, **_):
        for true_diameter, diameter, ball_position, calib in zip(batch_ball_size, predicted_diameter, batch_ball_position, batch_calib):
            if np.isnan(true_diameter):
                continue

            ball = Point3D(ball_position)

            predicted_position = compute_point3D_from_diameter(calib, calib.project_3D_to_2D(ball), diameter, BALL_DIAMETER)
            projection_error = compute_projection_error(ball, predicted_position)[0]
            relative_error = compute_relative_error(calib, ball, diameter)

            self.acc["true_diameter"].append(true_diameter)
            self.acc["predicted_diameter"].append(diameter)
            self.acc["diameter_error"].append(diameter - true_diameter)
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

class ComputeJointError(Callback):
    before = ["GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    def on_cycle_begin(self, **_):
        self.acc = defaultdict(lambda: [])
        self.data = []
    def on_batch_end(self, predicted_height, predicted_diameter, batch_ball_height, batch_ball_size, batch_ball_position, batch_calib, **_):
        for true_height, true_diameter, diameter, height, ball_position, calib in zip(batch_ball_height, batch_ball_size, predicted_diameter, predicted_height, batch_ball_position, batch_calib):
            if np.isnan(true_diameter):
                continue

            ball = Point3D(ball_position)

            #predicted_position = compute_point3D_from_height(calib, calib.project_3D_to_2D(ball), height)
            #projection_error = compute_projection_error(ball, predicted_position)[0]

            predicted_position = compute_point3D_from_diameter(calib, calib.project_3D_to_2D(ball), diameter, BALL_DIAMETER)
            projection_error = compute_projection_error(ball, predicted_position)[0]

            self.acc["true_height"].append(true_height)
            self.acc["true_diameter"].append(true_diameter)
            self.acc['predicted_height'].append(height)
            self.acc["predicted_diameter"].append(diameter)
            self.acc["height_error"].append(height - true_height)
            self.acc["diameter_error"].append(diameter - true_diameter)
            self.acc["projection_error"].append(projection_error)

            self.data.append({
                "ball": ball,
                "calib": calib,
                "predicted_diameter": diameter,
                "predicted_height": height,
            })
    def on_cycle_end(self, state, **_): # state in R/W mode
        try:
            df = pandas.DataFrame(np.vstack(list(self.acc.values())).T, columns=self.acc.keys())
            state["ball_size_metrics"] = df
            state["MADE"] = np.mean(np.abs(df['predicted_diameter'] - df['true_diameter']))
            state["MAHE"] = np.mean(np.abs(df['predicted_height'] - df['true_height']))
            state["MAPE"] = np.mean(np.abs(df['projection_error']))
            state['tbd'] = self.data
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
    def on_batch_end(self, batch_is_ball, predicted_is_ball, batch_ball_position, batch_has_ball=None, **_):
        balls, inverse = np.unique(np.array(batch_ball_position), axis=0, return_inverse=True)
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


class NamedOutputs(ChunkProcessor):
    def __init__(self, input_name='batch_logits'):
        self.input_name = input_name
    def __call__(self, chunk):
        chunk["predicted_diameter"] = chunk[self.input_name][...,0]
        chunk["predicted_is_ball"] = chunk[self.input_name][...,1]
        if chunk[self.input_name].shape[-1] == 3:
            chunk["predicted_height"] = chunk[self.input_name][...,2]


class IsBallClassificationLoss(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __call__(self, chunk):
        # TODO: check if binary crossentropy fits the unconfident targets
        chunk["classification_loss"] = tf.keras.losses.binary_crossentropy(y_true=chunk["batch_is_ball"], y_pred=chunk["predicted_is_ball"], from_logits=True)

class ClassificationLoss(IsBallClassificationLoss):
    pass # retrocompatibility

class RegressionLoss(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __init__(self, delta=1.0):
        self.delta = delta # required to print config
        self.diameter_loss = tf.keras.losses.Huber(delta=delta, name='huber_loss')
        self.height_loss = tf.keras.losses.Huber(delta=delta, name='huber_loss')
    def __call__(self, chunk):
        mask = tf.math.logical_not(tf.math.is_nan(chunk["batch_ball_size"]))
        losses = self.diameter_loss(y_true=chunk["batch_ball_size"][mask], y_pred=chunk["predicted_diameter"][mask])
        chunk["regression_loss"] = tf.reduce_mean(losses)#tf.where(tf.math.is_nan(losses), tf.zeros_like(losses), losses)
        losses = self.height_loss(y_true=chunk["batch_ball_height"][mask], y_pred=chunk["predicted_height"][mask])
        chunk["height_regression_loss"] = tf.reduce_mean(losses)

class AddIsBallTargetFactory(Transform):
    def __call__(self, view_key: ViewKey, view: View):
        return {"is_ball": 1 if view.ball.origin == 'annotation' else 0}
