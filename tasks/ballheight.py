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


class BallHeightEstimation(TensorflowExperiment):
    batch_metrics_names = ["predicted_is_ball", "predicted_height", "height_regression_loss", "size_regression_loss", "regression_loss", "classification_loss"]
    batch_outputs_names = ["predicted_is_ball", "predicted_height"]
    @cached_property
    def batch_inputs_names(self):
        batch_inputs_names = ["batch_is_ball", "batch_ball_height", "batch_input_image", "epoch"]
        if self.cfg.get('with_diff', None):
            batch_inputs_names += ["batch_input_image2"]
        return batch_inputs_names

    def train(self, *args, **kwargs):
        self.cfg['testing_arena_labels'] = self.cfg['dataset_splitter'].testing_arena_labels
        return super().train(*args, **kwargs)

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


class ComputeHeightError(Callback):
    before = ["GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    def on_cycle_begin(self, **_):
        self.acc = defaultdict(lambda: [])
        self.evaluation_data = []
    def on_batch_end(self, predicted_height, batch_ball_height, batch_ball_position, batch_calib, **_):
        for true_height, height, ball_position, calib in zip(batch_ball_height, predicted_height, batch_ball_position, batch_calib):
            if np.isnan(true_height):
                continue

            ball = Point3D(ball_position)

            predicted_position = compute_point3D_from_height(calib, calib.project_3D_to_2D(ball), height)
            projection_error = compute_projection_error(ball, predicted_position)[0]

            self.acc["true_height"].append(true_height)
            self.acc['predicted_height'].append(height)
            self.acc["height_error"].append(height - true_height)
            self.acc["projection_error"].append(projection_error)

            self.evaluation_data.append({
                "ball": ball,
                "calib": calib,
                "predicted_height": height,
            })
    def on_cycle_end(self, state, **_): # state in R/W mode
        try:
            df = pandas.DataFrame(np.vstack(list(self.acc.values())).T, columns=self.acc.keys())
            state["evaluation_metrics"] = df
            state["MAHE"] = np.mean(np.abs(df['predicted_height'] - df['true_height']))
            state["MAPE"] = np.mean(np.abs(df['projection_error']))
            state['evaluation_data'] = self.evaluation_data
        except ValueError:
            state["evaluation_metrics"] = None
            for name in ["MAPE", "MAHE"]:
                state[name] = np.nan

class HeightEstimationNamedOutputs(ChunkProcessor):
    def __init__(self, input_name='batch_logits'):
        self.input_name = input_name
    def __call__(self, chunk):
        chunk["predicted_is_ball"] = chunk[self.input_name][...,0]
        chunk["predicted_diameter"] = chunk[self.input_name][...,1]
        chunk["predicted_height"] = chunk[self.input_name][...,2]
