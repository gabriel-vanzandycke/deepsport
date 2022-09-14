import json
import os
from typing import NamedTuple

import numpy as np

from calib3d import Point2D
from experimentator import build_experiment
from deepsport_utilities.transforms import Transform
from deepsport_utilities.utils import DefaultDict

from models.other import CropBlockDividable
from tasks.detection import EnlargeTarget


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
        point3D = instant.calibs[stream_idx].project_2D_to_3D(point2D, Z=0)

        def best_camera(point3D, calibs):
            projects = lambda p, calib: calib.projects_in(p)
            distance = lambda x, width: np.min([x, width-x])
            camera_idx = np.nanargmax([distance(calib.project_3D_to_2D(point3D).x, calib.width) if projects(point3D, calib) else np.nan for calib in calibs])
            return camera_idx, calibs[camera_idx].project_3D_to_2D(point3D)

        return BallDetection(self.model, *best_camera(point3D, instant.calibs), outputs[stream_idx])

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

