from dataclasses import dataclass
import pickle
import os
from typing import NamedTuple

import numpy as np
import tensorflow as tf
from matplotlib import cm

from calib3d import Point2D
from experimentator import build_experiment, Callback, ExperimentMode, ChunkProcessor
from experimentator.tf2_experiment import TensorflowExperiment
from experimentator.dataset import Subset, collate_fn
from deepsport_utilities.ds.instants_dataset import Ball
from deepsport_utilities.ds.instants_dataset.views_transforms import NaiveViewRandomCropperTransform
from deepsport_utilities.transforms import Transform
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
            yield from subset.dataset.batches(keys=keys, batch_size=batch_size, collate_fn=collate_fn, *args, **kwargs)

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
        self.database_path = os.path.join(dataset_folder, "{}/{}/balls3d_new.pickle")
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

    def extract_pseudo_annotation(self, data):
        camera    = np.array([c.camera_idx for c in data])
        models    = np.array([c.model for c in data])
        points    = Point2D([c.point for c in data])
        values    = np.array([c.value for c in data])
        threshold = np.array([c.value > self.detection_thresholds[c.model] for c in data])

        camera_cond      = camera[np.newaxis, :] == camera[:, np.newaxis]
        corroborate_cond = models[np.newaxis, :] != models[:, np.newaxis]
        proximity_cond   = np.linalg.norm(points[:, np.newaxis, :] - points[:, :, np.newaxis], axis=0) < self.max_distance
        threshold_cond   = threshold[:, np.newaxis] @ threshold[np.newaxis, :]

        values_matrix = values[np.newaxis, :] + values[:, np.newaxis]
        values_matrix_filtered = np.triu(camera_cond * corroborate_cond * proximity_cond * threshold_cond * values_matrix)
        i1, i2 = np.unravel_index(values_matrix_filtered.argmax(), values_matrix_filtered.shape)
        if i1 != i2: # means two different candidate were found
            point2D = Point2D(np.mean([data[i1].point, data[i2].point], axis=0))
            pseudo_annotation = BallDetection("pseudo-annotation", data[i1].camera_idx, point2D, value=values_matrix[i1, i2])
            for i in sorted([i1, i2], reverse=True): # safe delete using decreasing indices
                del data[i]
            return pseudo_annotation
        return None

    def __call__(self, instant_key, instant):
        sequence_frame_index = instant.frame_indices[0] # use index from first camera by default
        data = self.database[instant.arena_label, instant.game_id].get(sequence_frame_index, [])
        if data:
            point = lambda point: Point2D(point.y, point.x) if self.xy_inverted else point
            unpack = lambda detection: Ball({
                "origin": detection.model,
                "center": instant.calibs[detection.camera_idx].project_2D_to_3D(point(detection.point), Z=0),
                "image": detection.camera_idx,
                "visible": True, # visible enough to have been detected by a detector
                "state": instant.ball_state
            })
            pseudo_annotation = self.extract_pseudo_annotation(data)
            if pseudo_annotation is not None:
                instant.ball = unpack(pseudo_annotation)
                instant.annotations.extend([instant.ball])
            instant.detections.extend(map(unpack, data))
        return instant


class BallCropperTransform(NaiveViewRandomCropperTransform):
    def _get_current_parameters(self, view_key, view):
        input_shape = view.calib.width, view.calib.height
        keypoints = view.calib.project_3D_to_2D(view.ball.center)
        return keypoints, 1, input_shape

class AddBallStateFactory(Transform):
    def __call__(self, view_key, view):
        predicate = lambda a: a.camera == view_key.camera and a.type == "ball" and view.calib.projects_in(a.center) and a.visible is not False
        balls = [a for a in view.annotations if predicate(a)]
        return {"ball_state": balls[0].state if balls and balls[0].state is not None else BallState.NONE} # takes the first ball by convention

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

