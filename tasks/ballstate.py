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

from experimentator import ExperimentMode, ChunkProcessor, Subset, Callback, build_experiment
from experimentator.tf2_experiment import TensorflowExperiment
from dataset_utilities.ds.raw_sequences_dataset import BallState
from deepsport_utilities.ds.instants_dataset import BallState, BallViewRandomCropperTransform, Ball, ViewKey, View, InstantKey

from deepsport_utilities.dataset import Subset, SubsetType, find
from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.transforms import Transform
from dataset_utilities.ds.raw_sequences_dataset import SequenceInstantKey
from tasks.detection import divide, ComputeDetectionMetrics as _ComputeDetectionMetrics
from tasks.classification import ComputeClassifactionMetrics as _ComputeClassifactionMetrics, ComputeConfusionMatrix as _ComputeConfusionMatrix

class StateOnlyBalancer():
    def __init__(self, cfg):
        #self.classes = list(cfg['state_mapping'].keys()) # makes a copy
        self.classes = list(range(len(cfg['state_mapping'][1]))) # Tout ceci est dégeuh. à cleaner dès que je sais quel nstate choisir
        try:
            self.classes.remove(BallState.NONE)
        except ValueError:
            pass
        #print(self.classes)
        #raise
        self.get_class = lambda k,v: np.where(v['ball_state'])[0][0] # returns the index in state_mapping that is 1
        self.dataset_name = cfg['dataset_name']

    @cached_property
    def cache(self):
        return {}

class BallStateClassification(TensorflowExperiment):
    batch_inputs_names = ["batch_ball_state", "batch_input_image", "batch_input_image2"]
    batch_metrics_names = ["batch_output", "batch_target"]
    batch_outputs_names = ["batch_output"]

    @staticmethod
    def balanced_keys_generator(keys, balancer, query_item):
        pending = defaultdict(list)
        for key in keys:
            try:
                c = balancer.cache.get(key) or balancer.cache.setdefault(key, balancer.get_class(key, query_item(key)))
            except KeyError:
                continue
            except TypeError: # if query_item(key) is None (i.e. impossible to satisfy crop) a TypeError will be raised.
                continue
            pending[c].append(key)
            if all([len(pending[c]) > 0 for c in balancer.classes]):
                for c in balancer.classes:
                    yield pending[c].pop(0)

    @cached_property
    def balancer(self):
        return self.cfg['balancer'](self.cfg)

    def batch_generator_bkp(self, subset: Subset, *args, batch_size=None, **kwargs):
        if subset.name == "ballistic":
            yield from super().batch_generator(subset, *args, batch_size=batch_size, **kwargs)
        else:
            batch_size = batch_size or self.batch_size
            keys = self.balanced_keys_generator(subset.shuffled_keys(), self.balancer, subset.dataset.query_item)
            # yields pairs of (keys, data)
            yield from subset.batches(keys=keys, batch_size=batch_size, *args, **kwargs)


class BallStateAndBallSizeExperiment(TensorflowExperiment):
    batch_inputs_names = ["batch_input_image", "batch_input_image2",
                          "batch_ball_presence", "batch_ball_size", "batch_ball_state", "batch_ball_position"]
    batch_metrics_names = ["predicted_presence", "predicted_diameter", "predicted_state",
                           "diameter_loss", "presence_loss", "state_loss"]
    batch_outputs_names = ["predicted_presence", "predicted_diameter", "predicted_state"]

    @cached_property
    def balancer(self):
        return self.cfg['balancer'](self.cfg) if self.cfg['balancer'] else None

    def batch_generator_bkp(self, subset: Subset, *args, batch_size=None, **kwargs):
        if subset.type == SubsetType.EVAL or self.balancer is None:
            yield from super().batch_generator(subset, *args, batch_size=batch_size, **kwargs)
        else:
            batch_size = batch_size or self.batch_size
            keys_gen = BallStateClassification.balanced_keys_generator(subset.shuffled_keys(), self.balancer, subset.query_item)
            # yields pairs of (keys, data)
            yield from subset.batches(keys=keys_gen, batch_size=batch_size, *args, **kwargs)

    def train(self, *args, **kwargs):
        self.cfg['testing_arena_labels'] = self.cfg['dataset_splitter'].testing_arena_labels
        return super().train(*args, **kwargs)


class MissingChunkProcessor(ValueError):
    pass

class AugmentedExperiment(TensorflowExperiment):
    @cached_property
    def chunk(self):
        # build model
        chunk = super().chunk

        def matching_chunk_processor(chunk_processor, chunk_processors):
            for cp in chunk_processors:
                if hasattr(cp, 'model') and cp.model.name == chunk_processor.model.name:
                    return cp
            raise MissingChunkProcessor

        # load weights
        if experiment_id := self.cfg.get("starting_weights"):
            trainable = self.cfg.get("starting_weights_trainable", {})
            folder = os.path.join(os.environ['RESULTS_FOLDER'], "ballstate", experiment_id)
            exp = build_experiment(os.path.join(folder, "config.py"))
            exp.load_weights(now=True)

            for cp in self.chunk_processors:
                if hasattr(cp, "model"):
                    filename = os.path.join(folder, cp.model.name)
                    try:
                        if not os.path.exists(f"{filename}.index"):
                            matching_chunk_processor(cp, exp.chunk_processors).model.save_weights(filename)
                        cp.model.load_weights(filename)
                        cp.model.trainable = trainable.get(cp.model.name, False)
                        print(f"Loading {cp.model.name} weights from {filename} (trainable={cp.model.trainable})")
                    except MissingChunkProcessor:
                        print(f"'{cp.model.name}' chunk processor couldn't be found in {experiment_id}. Weights not loaded.")
        return chunk

@dataclass
class StateFLYINGMetrics(Callback):
    before = ["GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    def on_cycle_begin(self, **_):
        self.acc = {'true': [], 'pred': []}
    def on_batch_end(self, **state):
        _, C = state['predicted_state'].shape
        for predicted_state, target_state in zip(state['predicted_state'], state['batch_ball_state']):
            self.acc['pred'].append(float(predicted_state[1] if C > 1 else predicted_state[0]))
            self.acc['true'].append(float(target_state[1] if C > 1 else target_state[0]))
    def on_cycle_end(self, state, **_):
        P, R, T = sklearn.metrics.precision_recall_curve(self.acc['true'], self.acc['pred'])
        state[f"{str(BallState.FLYING)}_prc"] = pandas.DataFrame(np.vstack([P[:-1], R[:-1], T]).T, columns=['precision', 'recall', 'thresholds'])
        state[f"{str(BallState.FLYING)}_auc"] = sklearn.metrics.auc(R, P)


class AddSingleBallStateFactory(Transform):
    def __call__(self, view_key, view):
        predicate = lambda ball: view.calib.projects_in(ball.center) and ball.visible is not False and ball.state == BallState.FLYING
        return {"ball_state": [1] if predicate(view.ball) else [0]}


class AddBallPresenceFactory(Transform):
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


class TwoTasksBalancer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("retro-compatibility")

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
    def __call__(self, chunk):
        loss = tf.keras.losses.binary_crossentropy(chunk["batch_ball_state"], chunk["predicted_state"], from_logits=True)
        if len(loss.shape) > 1: # if BallState.NONE class is used, avoid computing loss for it
            loss = loss[:,1:]
        chunk["state_loss"] = tf.reduce_mean(loss)


class BallDetection(NamedTuple): # for retro-compatibility
    model: str
    camera_idx: int
    point: Point2D
    value: float
