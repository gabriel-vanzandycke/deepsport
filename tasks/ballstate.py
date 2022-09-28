from dataclasses import dataclass
import json
import os
from typing import NamedTuple

import numpy as np
import pandas
import tensorflow as tf

from calib3d import Point2D
from experimentator import build_experiment, ExperimentMode, Callback
from experimentator.tf2_experiment import TensorflowExperiment
from experimentator.dataset import Subset, collate_fn
from deepsport_utilities.ds.instants_dataset.views_transforms import NaiveViewRandomCropperTransform
from deepsport_utilities.transforms import Transform
from deepsport_utilities.utils import DefaultDict
from dataset_utilities.ds.raw_sequences_dataset import BallState, ball_states

from models.other import CropBlockDividable
from tasks.detection import EnlargeTarget, divide


class BallStateClassification(TensorflowExperiment):
    batch_inputs_names = ["batch_ball_state", "batch_input_image"]
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
        batch_size = batch_size or self.batch_size
        classes = [BallState.FLYING, BallState.CONSTRAINT, BallState.DRIBBLING]
        get_class = lambda k,v: v['ball_state']
        keys = self.balanced_keys_generator(subset.shuffled_keys(), get_class, classes, self.class_cache, subset.dataset.query_item)
        # yields pairs of (keys, data)
        yield from subset.dataset.batches(keys=keys, batch_size=batch_size, collate_fn=collate_fn, *args, **kwargs)

class BallDetection(NamedTuple):
    model: str
    image: int
    point: Point2D
    value: float

class Detector:
    def __init__(self, model, experiment_id):
        config = os.path.join(os.environ['RESULTS_FOLDER'], model, experiment_id, "config.py")
        self.exp = build_experiment(config)
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
        outputs = np.array(result['topk_outputs'][:,0,0,0])
        stream_idx = np.argmax(outputs)
        point2D = Point2D(np.array(result['topk_indices'][stream_idx,0,0,0]))

        return BallDetection(self.model, stream_idx, point2D, outputs[stream_idx])

PIFBALL_THRESHOLD = 0.1
BALLSEG_THRESHOLD = 0.8


class AddBallDetectionTransform(Transform):
    database_path = "/DATA/datasets/raw-games/{}/{}/balls.json"
    def __init__(self):
        def factory(args):
            arena_label, game_id = args
            filename = self.database_path.format(arena_label, game_id)
            try:
                return json.load(open(filename, "r"))
            except FileNotFoundError:
                return {}
        self.database = DefaultDict(factory)
    def __call__(self, instant_key, instant):
        instant.ball2D = self.database[instant.arena_label, instant.game_id].get(str(instant.sequence_frame_index), None)
        return instant


class BallCropperTransform(NaiveViewRandomCropperTransform):
    def _get_current_parameters(self, view_key, view):
        input_shape = view.calib.width, view.calib.height
        keypoints = view.calib.project_3D_to_2D(view.annotations[0].center)
        return keypoints, 1, input_shape

class AddBallStateFactory(Transform):
    def __call__(self, view_key, view):
        predicate = lambda a: a.camera == view_key.camera and a.type == "ball" and view.calib.projects_in(a.center) and a.visible is not False
        balls = [a for a in view.annotations if predicate(a)]
        state_fct = lambda state: state if isinstance(state, BallState) else ball_states[state]
        return {"ball_state": state_fct(balls[0].state) if balls else BallState.NONE} # takes the first ball by convention