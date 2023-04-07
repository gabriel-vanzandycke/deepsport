from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple
import typing

import numpy as np
from calib3d import Point2D
import tensorflow as tf
import pandas

from experimentator import ExperimentMode, ChunkProcessor, Subset, Callback
from experimentator.tf2_experiment import TensorflowExperiment
from dataset_utilities.ds.raw_sequences_dataset import BallState
from deepsport_utilities.ds.instants_dataset import BallState, BallViewRandomCropperTransform, Ball, ViewKey, View, InstantKey

from deepsport_utilities.dataset import Subset, SubsetType
from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.transforms import Transform
from dataset_utilities.ds.raw_sequences_dataset import SequenceInstantKey
from tasks.detection import divide
from tasks.classification import ComputeClassifactionMetrics as _ComputeClassifactionMetrics, ComputeConfusionMatrix as _ComputeConfusionMatrix


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



class BallStateAndBallSizeExperiment(TensorflowExperiment):
    batch_inputs_names = ["batch_input_image", "batch_input_image2",
                          "batch_is_ball", "batch_ball_size", "batch_ball_state"]
    batch_metrics_names = ["predicted_is_ball", "predicted_diameter", "predicted_state",
                           "regression_loss", "classification_loss", "state_loss"]
    batch_outputs_names = ["predicted_is_ball", "predicted_diameter", "predicted_state"]

    class_cache = {}
    def batch_generator(self, subset: Subset, *args, batch_size=None, **kwargs):
        if subset.type == SubsetType.EVAL:
            yield from super().batch_generator(subset, *args, batch_size=batch_size, **kwargs)
        else:
            batch_size = batch_size or self.batch_size
            classes = [str(c) for c in self.cfg['classes'] if c != BallState.NONE]
            classes.extend(['ball_annotation', 'ball_interpolation', 'noball', 'other'])
            trusted_origins = ['annotation', 'interpolation']
            def get_class(k,v):
                if v['ball'].origin in trusted_origins:
                    return 'ball_' + v['ball'].origin
                if v['ball_state'] != BallState.NONE:
                    return str(v['ball_state'])
                if v['is_ball'] == 0:
                    return 'noball'
                return 'other'
            keys_gen = BallStateClassification.balanced_keys_generator(subset.shuffled_keys(), get_class, classes, self.class_cache, subset.dataset.query_item)
            # yields pairs of (keys, data)
            yield from subset.batches(keys=keys_gen, batch_size=batch_size, *args, **kwargs)



class AddBallSizeFactory(Transform):
    def __call__(self, view_key, view):
        ball = view.ball
        #                        (          either ball is an annotation           or             true ball annotation transferred to detection         )
        predicate = lambda ball: ( ball.origin in ['annotation', 'interpolation']  or  (ball.origin in ['pifball', 'ballseg'] and ball.center.z < -10)  ) \
            and ball.visible is not False and view.calib.projects_in(ball.center)
        return {"ball_size": view.calib.compute_length2D(ball.center, BALL_DIAMETER)[0] if predicate(ball) else np.nan}

class AddIsBallTargetFactory(Transform):
    def __init__(self, unconfident_margin=.1, proximity_threshold=10):
        self.unconfident_margin = unconfident_margin
        self.proximity_threshold = proximity_threshold
    def __call__(self, view_key: ViewKey, view: View):
        ball_origin = view.ball.origin
        trusted_origins = ['annotation', 'interpolation']
        if ball_origin in trusted_origins:
            return {"is_ball": 1}
        if 'random' == ball_origin:
            return {"is_ball": 0}

        annotated_balls = [a for a in view.annotations if isinstance(a, Ball) and a.origin in trusted_origins]
        annotated_ball = annotated_balls[0] if len(annotated_balls) == 1 else None
        if annotated_ball:
            projected = lambda ball: view.calib.project_3D_to_2D(ball.center)
            if np.linalg.norm(projected(view.ball) - projected(annotated_ball)) < self.proximity_threshold:
                return {"is_ball": 1}
            else:
                return {"is_ball": 0}

        elif 'pseudo-annotation' in ball_origin:
            return {"is_ball": 1 - self.unconfident_margin}
        else:
            return {'is_ball': 0 + self.unconfident_margin}



class BallViewRandomCropperTransformCompat():
    def __init__(self, *args, size_min=None, size_max=None, scale_min=None, scale_max=None, **kwargs):
        self.size_cropper_transform = BallViewRandomCropperTransform(
            *args, size_min=size_min, size_max=size_max, **kwargs)
        self.scale_cropper_transform = BallViewRandomCropperTransform(
            *args, scale_min=scale_min, scale_max=scale_max, **kwargs)
    def __call__(self, view_key, view):
        trusted_origins = ['annotation', 'interpolation']
        if isinstance(view_key[0], SequenceInstantKey) or view.ball.origin not in trusted_origins:
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



@dataclass
class ComputeDetectionMetrics(Callback):
    origin: str = 'ballseg'
    before = ["AuC", "GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    thresholds: typing.Tuple[int, np.ndarray, list, tuple] = np.linspace(0,1,51)
    def on_cycle_begin(self, **_):
        self.d_acc = defaultdict(list)
        self.t_acc = defaultdict(bool) # defaults to False

    def on_batch_end(self, keys, batch_ball, batch_is_ball, predicted_is_ball, **_):
        for view_key, ball, target_is_ball, predicted in zip(keys, batch_ball, batch_is_ball, predicted_is_ball):
            if isinstance(view_key.instant_key, InstantKey): # Keep only views from deepsport dataset for evaluation
                key = (view_key.instant_key, view_key.camera)
                if ball.origin == self.origin:
                    self.d_acc[key].append((ball, target_is_ball, predicted))
                    if np.any(target_is_ball):
                        self.t_acc[key] = True # balls might be visible on an image despite having been annotated on another.
                elif ball.origin == 'annotation':
                    self.t_acc[key] = True

    def on_cycle_end(self, state, **_):
        for k in [None, 1, 2, 4, 8]:
            TP = np.zeros((len(self.thresholds), ))
            FP = np.zeros((len(self.thresholds), ))
            P = N = 0
            P_upper_bound = 0
            for key, zipped in self.d_acc.items():
                balls, target_is_ball, predicted_is_ball = zip(*zipped)
                values = [b.value for b in balls]
                if k is None:
                    index = np.argmax(values)
                    P_upper_bound += np.any(target_is_ball)
                else:
                    indices = np.argsort(values)[-k:]
                    index = indices[np.argmax(np.array(predicted_is_ball)[indices])]
                    values = predicted_is_ball
                    P_upper_bound += np.any(np.array(target_is_ball)[indices])

                output = (values[index] >= self.thresholds).astype(np.uint8)
                target = target_is_ball[index]
                TP +=   target   *  output
                FP += (1-target) *  output

                has_ball = self.t_acc[key]
                P  +=   has_ball
                N  += not has_ball

            name = 'initial_TP_rate_upper_bound' if k is None else f'top{k}_TP_rate_upper_bound'
            state[name] = P_upper_bound/P

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
        self.layers = tf.keras.models.Sequential(layers, "1layer")
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


class BallDetection(NamedTuple): # for retro-compatibility
    model: str
    camera_idx: int
    point: Point2D
    value: float
