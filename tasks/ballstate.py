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

class BallDetection(NamedTuple):
    model: str
    camera_idx: int
    point: Point2D
    value: float

class Detector():
    def __init__(self, model, config, k=[1]):
        self.exp = build_experiment(config, k=k)
        self.k = np.max(k)
        self.model = model
        if model == 'ballseg':
            self.exp.chunk_processors.insert(0, CropBlockDividable(tensor_names=['batch_input_image', 'batch_input_image2']))
            self.exp.chunk_processors[-2] = EnlargeTarget

    def __call__(self, instant):
        offset = instant.offsets[1]
        data = {
            "batch_input_image": np.stack(instant.images),
            "batch_input_image2": np.stack([instant.all_images[(c, offset)] for c in range(instant.num_cameras)])
        }

        result = self.exp.predict(data)
        for b in range(len(result['topk_outputs'])):
            for i in range(self.k):
                y, x = np.array(result['topk_indices'][b,0,0,i]) # TODO: check if this passes
                value = result['topk_outputs'][b,0,0,i].numpy()
                yield BallDetection(self.model, b, Point2D(x, y), value)

PIFBALL_THRESHOLD = 0.05
BALLSEG_THRESHOLD = 0.6


class AddBallDetectionsTransform(Transform):
    def __init__(self, dataset_folder, xy_inverted=False):
        self.dataset_folder = dataset_folder
        self.database_path = os.path.join(dataset_folder, "{}/{}/balls3d_new.pickle") # file created by `process_raw_sequences.py` script
        def factory(args):
            arena_label, game_id = args
            filename = self.database_path.format(arena_label, game_id)
            try:
                return pickle.load(open(filename, "rb"))
            except FileNotFoundError:
                return {}
        self.database = DefaultDict(factory)
        self.detection_thresholds = {
            "pifball": 0.1,
            "ballseg": 0.8
        }
        self.max_distance = 28 # pixels
        self.xy_inverted = xy_inverted

    def extract_pseudo_annotation(self, detections):
        camera    = np.array([d.camera_idx for d in detections])
        models    = np.array([d.model for d in detections])
        points    = Point2D([d.point for d in detections])
        values    = np.array([d.value for d in detections])
        threshold = np.array([d.value > self.detection_thresholds[d.model] for d in detections])

        camera_cond      = camera[np.newaxis, :] == camera[:, np.newaxis]
        corroborate_cond = models[np.newaxis, :] != models[:, np.newaxis]
        proximity_cond   = np.linalg.norm(points[:, np.newaxis, :] - points[:, :, np.newaxis], axis=0) < self.max_distance
        threshold_cond   = threshold[:, np.newaxis] @ threshold[np.newaxis, :]

        values_matrix = values[np.newaxis, :] + values[:, np.newaxis]
        values_matrix_filtered = np.triu(camera_cond * corroborate_cond * proximity_cond * threshold_cond * values_matrix)
        i1, i2 = np.unravel_index(values_matrix_filtered.argmax(), values_matrix_filtered.shape)
        if i1 != i2: # means two different candidate were found
            point2D = Point2D(np.mean([detections[i1].point, detections[i2].point], axis=0))
            pseudo_annotation = BallDetection("pseudo-annotation", detections[i1].camera_idx, point2D, value=values_matrix[i1, i2])
            #for i in sorted([i1, i2], reverse=True): # safe delete using decreasing indices
            #    del detections[i]
            return pseudo_annotation
        return None

    def __call__(self, instant_key, instant):
        sequence_frame_index = instant.frame_indices[0] # use index from first camera by default
        detections = self.database[instant.arena_label, instant.game_id].get(sequence_frame_index, [])
        if detections:
            point = lambda point: Point2D(point.y, point.x) if self.xy_inverted else point
            unpack = lambda detection: Ball({
                "origin": detection.model,
                "center": instant.calibs[detection.camera_idx].project_2D_to_3D(point(detection.point), Z=0),
                "image": detection.camera_idx,
                "visible": True, # visible enough to have been detected by a detector
                "state": instant.ball_state,
                "value": detection.value
            })
            pseudo_annotation = self.extract_pseudo_annotation(detections)
            if pseudo_annotation is not None:
                instant.ball = unpack(pseudo_annotation)
                instant.annotations.extend([instant.ball])
            instant.detections.extend(map(unpack, detections))
        return instant


class AddBallSizeFactory(Transform):
    def __call__(self, view_key, view):
        ball = view.ball
        predicate = lambda ball: ball.origin in ['annotation', 'interpolation']
        return {"ball_size": view.calib.compute_length2D(ball, BALL_DIAMETER)[0] if predicate(ball) else np.nan}

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
