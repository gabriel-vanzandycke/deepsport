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
from tasks.detection import divide
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

    def batch_generator(self, subset: Subset, *args, batch_size=None, **kwargs):
        if subset.name == "ballistic":
            yield from super().batch_generator(subset, *args, batch_size=batch_size, **kwargs)
        else:
            batch_size = batch_size or self.batch_size
            keys = self.balanced_keys_generator(subset.shuffled_keys(), self.balancer, subset.dataset.query_item)
            # yields pairs of (keys, data)
            yield from subset.batches(keys=keys, batch_size=batch_size, *args, **kwargs)


class BallStateAndBallSizeExperiment(TensorflowExperiment):
    batch_inputs_names = ["batch_input_image", "batch_input_image2", "batch_ball_height",
                          "batch_is_ball", "batch_ball_size", "batch_ball_state"]
    batch_metrics_names = ["predicted_is_ball", "predicted_diameter", "predicted_state", "predicted_height"
                           "regression_loss", "classification_loss", "state_loss"]
    batch_outputs_names = ["predicted_is_ball", "predicted_diameter", "predicted_state", "predicted_height"]

    @cached_property
    def balancer(self):
        return self.cfg['balancer'](self.cfg) if self.cfg['balancer'] else None

    def batch_generator(self, subset: Subset, *args, batch_size=None, **kwargs):
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

    @cached_property
    def chunk(self):
        chunk = super().chunk
        if experiment_id := self.cfg.get("ballsize_weights"):
            folder = os.path.join(os.environ['RESULTS_FOLDER'], "ballstate", experiment_id)
            exp = None

            for name, condition in [
                ("backbone",        lambda cp: hasattr(cp, "model")),
                ("regression_head", lambda cp: hasattr(cp, "model") and cp.model.name == 'regression_head'),
            ]:
                filename = os.path.join(folder, name)
                if not os.path.exists(f"{filename}.index"):
                    exp = exp or build_experiment(os.path.join(folder, "config.py"))
                    exp.load_weights(now=True)
                    for cp in exp.chunk_processors:
                        if condition(cp):
                            cp.model.save_weights(filename)
                            break
                for cp in self.chunk_processors:
                    if condition(cp):
                        print(f"Loading {name} weights from {filename}")
                        cp.model.load_weights(filename)
                        if self.cfg.get('freeze_ballsize', False):
                            cp.model.trainable = False
                        break
        elif self.cfg.get('freeze_ballsize', False) is True:
            raise ValueError("Cannot freeze ballsize weights if no experiment_id is provided")

        return chunk

    def save_weights(self, *args, **kwargs):
        folder = os.path.join(os.environ['RESULTS_FOLDER'], "ballstate", self.cfg['experiment_id'])
        if self.cfg['nstates']:
            for cp in self.chunk_processors:
                if hasattr(cp, "model") and cp.model.name == 'classification_head':
                    filename = os.path.join(folder, "classification_head")
                    cp.model.save_weights(filename)
                    break
        else:
            super().save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        experiment_id = self.cfg.get('experiment_id', os.path.basename(os.path.dirname(self.cfg["filename"])))
        folder = os.path.join(os.environ['RESULTS_FOLDER'], "ballstate", experiment_id)
        print(self.cfg['nstates'])
        #super().load_weights(*args, **kwargs)
        if self.cfg['nstates']:
            for cp in self.chunk_processors:
                if hasattr(cp, "model") and cp.model.name == 'classification_head':
                    filename = os.path.join(folder, "classification_head")
                    cp.model.load_weights(filename)
                    break
        else:
            super().load_weights(*args, **kwargs)



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


class AddBallSizeFactory(Transform):
    def __init__(self, predict_height=False):
        self.predict_height = predict_height
    def __call__(self, view_key, view):
        ball = view.ball
        #                        (          either ball is an annotation           or             true ball annotation transferred to detection         )
        predicate = lambda ball: ( ball.origin in ['annotation', 'interpolation']  or  (ball.origin in ['pifball', 'ballseg'] and ball.center.z < -10)  ) \
            and ball.visible is not False and view.calib.projects_in(ball.center)
        data = {"ball_size": view.calib.compute_length2D(ball.center, BALL_DIAMETER)[0] if predicate(ball) else np.nan}
        if self.predict_height:
            center2D = view.calib.project_3D_to_2D(ball.center)
            ground2D = view.calib.project_3D_to_2D(Point3D(ball.center.x, ball.center.y, 0))
            data["ball_height"] = np.linalg.norm(center2D - ground2D) if predicate(ball) else np.nan
        return data

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

class ComputeDetectionMetrics(ComputeDetectionMetrics_Detection):
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
        chunk["loss"] = chunk["state_loss"] = tf.reduce_mean(loss)


class CombineLosses(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __init__(self, names, weights):
        self.weights = weights
        self.names = names
    def __call__(self, chunk):
        chunk["loss"] = tf.reduce_sum([chunk[name]*w for name, w in zip(self.names, self.weights) if name in chunk])


class BallDetection(NamedTuple): # for retro-compatibility
    model: str
    camera_idx: int
    point: Point2D
    value: float
