from dataclasses import dataclass
import pickle
import os
from typing import NamedTuple

import numpy as np
import tensorflow as tf
from matplotlib import cm

from calib3d import Point2D
from experimentator import build_experiment, Callback, ExperimentMode, ChunkProcessor, Subset
from experimentator.tf2_experiment import TensorflowExperiment
from deepsport_utilities.ds.instants_dataset import Ball
from deepsport_utilities.ds.instants_dataset.views_transforms import NaiveViewRandomCropperTransform
from deepsport_utilities.transforms import Transform
from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.utils import DefaultDict
from dataset_utilities.ds.raw_sequences_dataset import BallState

from models.other import CropBlockDividable
from tasks.detection import EnlargeTarget


class BallStateClassification(TensorflowExperiment):
    batch_inputs_names = ["batch_ball_state", "batch_input_image", "batch_input_image2"]
    batch_metrics_names = ["batch_output", "batch_target"]
    batch_outputs_names = ["batch_output"]

    @staticmethod
    def balanced_keys_generator(keys, get_class, classes, cache, query_item):
        pending = {c: [] for c in classes}
        for key in keys:
            c = cache.get(key) or cache.setdefault(key, get_class(key, query_item(key)))
            pending[c].append(key)
            if all([len(l) > 0 for l in pending.values()]):
                for c in classes:
                    yield pending[c].pop(0)

    class_cache = {}
    def batch_generator(self, subset: Subset, *args, batch_size=None, **kwargs):
        if subset.name == "ballistic":
            yield from super().batch_generator(subset, *args, batch_size=batch_size, **kwargs)
        else:
            batch_size = batch_size or self.batch_size
            classes = [BallState.FLYING, BallState.CONSTRAINT, BallState.DRIBBLING]
            get_class = lambda k,v: v['ball_state']
            keys = self.balanced_keys_generator(subset.shuffled_keys(), get_class, classes, self.class_cache, subset.dataset.query_item)
            # yields pairs of (keys, data)
            yield from subset.batches(keys=keys, batch_size=batch_size, *args, **kwargs)

class AddBallSizeFactory(Transform):
    def __call__(self, view_key, view):
        ball = view.ball
        predicate = lambda ball: ball.origin in ['annotation', 'interpolation']
        return {"ball_size": view.calib.compute_length2D(ball.center, BALL_DIAMETER)[0] if predicate(ball) else np.nan}

class AddBallStateFactory(Transform):
    def __call__(self, view_key, view):
        return {"ball_state": view.ball.state or BallState.NONE}

class AddIsBallTargetFactory(Transform):
    def __init__(self, unconfident_margin=.2):
        self.unconfident_margin = unconfident_margin
    def __call__(self, view_key, view):
        ball = view.ball
        if ball.origin in ['annotation', 'interpolation']:
            return {"is_ball": 1}
        elif 'pseudo-annotation' in ball.origin:
            return {"is_ball": 1 - self.unconfident_margin}
        elif 'random' in ball.origin:
            return {"is_ball": 0}
        else:
            return {'is_ball': 0 + self.unconfident_margin}



@dataclass
class ExtractClassificationMetrics(Callback):
    before = ["GatherCycleMetrics"]
    after = ["ComputeClassifactionMetrics", "ComputeConfusionMatrix"]
    when = ExperimentMode.EVAL
    class_name: str
    class_index: int
    def on_cycle_end(self, state, **_):
        for metric in ['precision', 'recall']:
            state[f"{self.class_name}_{metric}"] = state['classification_metrics'][metric].iloc[self.class_index]


class ChannelsReductionLayer(ChunkProcessor):
    mode = ExperimentMode.ALL
    def __init__(self, kernel_size=3, maxpool=True, batchnorm=True, padding='SAME', strides=1):
        layers = [
            tf.keras.layers.Conv2D(filters=3, kernel_size=kernel_size, padding=padding)
        ]
        if maxpool:
            layers.append(
                tf.keras.layers.MaxPool2D(padding=padding, strides=strides)
            )
        if batchnorm:
            layers.append(
                tf.keras.layers.BatchNormalization()
            )
        self.layers = tf.keras.models.Sequential(layers)
        # required for printing chunk processors
        self.kernel_size = kernel_size
        self.maxpool = maxpool
        self.batchnorm = batchnorm
        self.padding = padding
        self.strides = strides
    def __call__(self, chunk):
        chunk['batch_input'] = self.layers(chunk['batch_input'])



class NamedOutputs(ChunkProcessor):
    def __call__(self, chunk):
        chunk["predicted_diameter"] = chunk["batch_logits"][...,0]
        chunk["predicted_is_ball"] = chunk["batch_logits"][...,1]
        chunk["predicted_state"] = chunk["batch_logits"][...,2:]

class StateClassificationLoss(ChunkProcessor):
    def __init__(self, classes):
        self.classes = classes
    def __call__(self, chunk):
        batch_target = tf.one_hot(chunk['batch_ball_state'], len(self.classes))
        loss = tf.keras.losses.binary_crossentropy(batch_target, chunk["predicted_state"], from_logits=True)
        chunk["state_loss"] = tf.reduce_mean(loss)

class CombineLosses(ChunkProcessor):
    def __init__(self, names, weights):
        self.weights = weights
        self.names = names
    def __call__(self, chunk):
        chunk["loss"] = tf.reduce_sum([chunk[name]*w for name, w in zip(self.names, self.weights)])
