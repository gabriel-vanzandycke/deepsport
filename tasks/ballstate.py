from collections import defaultdict
from typing import NamedTuple

import numpy as np
from calib3d import Point2D
import tensorflow as tf

from experimentator import ExperimentMode, ChunkProcessor, Subset
from experimentator.tf2_experiment import TensorflowExperiment
from dataset_utilities.ds.raw_sequences_dataset import BallState


class BallStateClassification(TensorflowExperiment):
    batch_inputs_names = ["batch_ball_state", "batch_input_image", "batch_input_image2"]
    batch_metrics_names = ["batch_output", "batch_target"]
    batch_outputs_names = ["batch_output"]

    @staticmethod
    def balanced_keys_generator(keys, get_class, classes, cache, query_item):
        pending = defaultdict(list)
        for key in keys:
            try:
                c = cache.get(key) or cache.setdefault(key, get_class(key, query_item(key)))
            except KeyError:
                continue
            pending[c].append(key)
            if all([len(pending[c]) > 0 for c in classes]):
                for c in classes:
                    yield pending[c].pop(0)

    class_cache = {}
    def batch_generator(self, subset: Subset, *args, batch_size=None, **kwargs):
        if subset.name == "ballistic":
            yield from super().batch_generator(subset, *args, batch_size=batch_size, **kwargs)
        else:
            batch_size = batch_size or self.batch_size
            classes = list(self.cfg['classes']) # makes a copy
            classes.remove(BallState.NONE)
            get_class = lambda k,v: v['ball_state']
            keys = self.balanced_keys_generator(subset.shuffled_keys(), get_class, classes, self.class_cache, subset.dataset.query_item)
            # yields pairs of (keys, data)
            yield from subset.batches(keys=keys, batch_size=batch_size, *args, **kwargs)


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


class BallDetection(NamedTuple): # for retro-compatibility
    model: str
    camera_idx: int
    point: Point2D
    value: float
