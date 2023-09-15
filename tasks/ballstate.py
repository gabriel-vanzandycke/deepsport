from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
import os
from typing import NamedTuple
import typing

import numpy as np
from calib3d import Point2D, Point3D
import pandas
import sklearn.metrics
import tensorflow as tf

from experimentator import ExperimentMode, ChunkProcessor, Subset, Callback
from experimentator.tf2_experiment import TensorflowExperiment, AugmentedExperiment # pylint: ignore=unused-import
from dataset_utilities.ds.raw_sequences_dataset import BallState
from deepsport_utilities.ds.instants_dataset import BallState, BallViewRandomCropperTransform, Ball, ViewKey, View, InstantKey

from deepsport_utilities.dataset import Subset, SubsetType, find
from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.transforms import Transform
from dataset_utilities.ds.raw_sequences_dataset import SequenceInstantKey
from tasks.detection import divide, ComputeDetectionMetrics as _ComputeDetectionMetrics
from tasks.classification import ComputeClassifactionMetrics as _ComputeClassifactionMetrics, ComputeConfusionMatrix as _ComputeConfusionMatrix


class BallStateAndBallSizeExperiment(TensorflowExperiment):
    batch_inputs_names = ["batch_input_image", "batch_input_image2",
                          "batch_ball_presence", "batch_ball_size", "batch_ball_state", "batch_ball_position"]
    batch_metrics_names = ["predicted_presence", "predicted_diameter", "predicted_state",
                           "diameter_loss", "presence_loss", "state_loss", "losses"]
    batch_outputs_names = ["predicted_presence", "predicted_diameter", "predicted_state"]


@dataclass
class StateFLYINGMetrics(Callback):
    before = ["GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    class_index: int
    def on_cycle_begin(self, **_):
        self.acc = {'true': [], 'pred': []}
    def on_batch_end(self, **state):
        _, C = state['predicted_state'].shape
        for predicted_state, target_state in zip(state['predicted_state'], state['batch_ball_state']):
            if np.any(np.isnan(target_state)):
                continue
            assert not np.any(np.isnan(predicted_state)), predicted_state
            self.acc['pred'].append(float(predicted_state[self.class_index]))
            self.acc['true'].append(float(target_state[self.class_index]))
    def on_cycle_end(self, state, **_):
        if not self.acc['true']:
            return
        P, R, T = sklearn.metrics.precision_recall_curve(self.acc['true'], self.acc['pred'])
        state[f"{str(BallState.FLYING)}_prc"] = pandas.DataFrame(np.vstack([P[:-1], R[:-1], T]).T, columns=['precision', 'recall', 'thresholds'])
        state[f"{str(BallState.FLYING)}_auc"] = sklearn.metrics.auc(R, P)


class AddBallPresenceFactory(Transform):
    """ supports untrusted origins
    """
    def __init__(self, unconfident_margin=.1, proximity_threshold=10):
        self.unconfident_margin = unconfident_margin
        self.proximity_threshold = proximity_threshold
    def __call__(self, view_key: ViewKey, view: View):
        ball_origin = view.ball.origin
        trusted_origins = ['annotation', 'interpolation']
        if ball_origin in trusted_origins:
            return {'ball_presence': 1}
        if 'random' == ball_origin:
            return {'ball_presence': 0}

        annotated_balls = [a for a in view.annotations if isinstance(a, Ball) and a.origin in trusted_origins]
        annotated_ball = annotated_balls[0] if len(annotated_balls) == 1 else None
        if annotated_ball:
            projected = lambda ball: view.calib.project_3D_to_2D(ball.center)
            if np.linalg.norm(projected(view.ball) - projected(annotated_ball)) < self.proximity_threshold:
                return {'ball_presence': 1}
            else:
                return {'ball_presence': 0}

        elif 'pseudo-annotation' in ball_origin:
            return {'ball_presence': 1 - self.unconfident_margin}
        else:
            return {'ball_presence': 0 + self.unconfident_margin}



class BallViewRandomCropperTransformCompat():
    def __init__(self, *args, size_min=None, size_max=None, scale_min=None, scale_max=None, **kwargs):
        self.size_cropper_transform = BallViewRandomCropperTransform(
            *args, size_min=size_min, size_max=size_max, **kwargs)
        self.scale_cropper_transform = BallViewRandomCropperTransform(
            *args, scale_min=scale_min, scale_max=scale_max, **kwargs)
    def __call__(self, view_key, view):
        trusted_origins = ['annotation', 'interpolation']
        if view.ball.origin not in trusted_origins:
            return self.scale_cropper_transform(view_key, view)
        else:
            return self.size_cropper_transform(view_key, view)


class ComputeClassifactionMetrics(_ComputeClassifactionMetrics):
    def on_batch_end(self, predicted_state, batch_ball_state, **_):
        B, C = predicted_state.shape
        onehot_true_state = tf.one_hot(batch_ball_state, C)
        super().on_batch_end(predicted_state, onehot_true_state, **_)

class ComputeConfusionMatrix(_ComputeConfusionMatrix):
    def on_batch_end(self, predicted_state, batch_ball_state, **_):
        B, C = predicted_state.shape
        onehot_true_state = tf.one_hot(batch_ball_state, C)
        super().on_batch_end(predicted_state, onehot_true_state, **_)

class ComputeDetectionMetrics(_ComputeDetectionMetrics):
    pass

@dataclass
class TopkNormalizedGain(Callback):
    before = ["GatherCycleMetrics"]
    after = ['AuC', 'ComputeDetectionMetrics']
    when = ExperimentMode.EVAL
    k: list
    close_curve: bool = True
    def on_cycle_end(self, state, **_):
        for k in self.k:
            baseline = state['initial_top1-AuC']
            gain             = state[f'top{k}-AuC']                 - baseline
            gain_upper_bound = state[f'top{k}_TP_rate_upper_bound'] - baseline
            state[f'top{k}_normalized_gain'] = gain/gain_upper_bound


class StateClassificationLoss(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __init__(self, nstates):
        self.loss_function = {
            True:  tf.keras.losses.CategoricalCrossentropy,
            False: tf.keras.losses.BinaryCrossentropy,
        }[nstates > 1](from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.nstates = nstates
    def __call__(self, chunk):
        mask = tf.math.logical_not(tf.math.is_nan(chunk["batch_ball_state"]))
        losses = self.loss_function(
            y_true=tf.where(mask, chunk["batch_ball_state"], 0),
            y_pred=chunk["predicted_state"]
        )
        mask = tf.reduce_all(mask, axis=-1)
        chunk["state_loss"] = tf.reduce_mean(losses[mask])


class BallDetection(NamedTuple): # for retro-compatibility
    model: str
    camera_idx: int
    point: Point2D
    value: float
