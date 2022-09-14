#!/usr/bin/env python
import argparse
from datetime import timedelta
import json
import os
print("Python Executable:", os.sys.executable)

import boto3
from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from calib3d import Point2D
from dataset_utilities.ds.raw_sequences_dataset import RawSequencesDataset, InstantsDataset
from deepsport_utilities.utils import DelayedCallback

from tasks.ballstate import Detector, PIFBALL_THRESHOLD, BALLSEG_THRESHOLD

load_dotenv("/home/gva/repositories/deepsport/.env")

parser = argparse.ArgumentParser(description="""Process Sequence to detect ball""")
parser.add_argument("arena_label")
parser.add_argument("game_id", type=int)
parser.add_argument('--break-frame', type=int)
args = parser.parse_args()


arena_label = args.arena_label
game_id = args.game_id
local_storage = "/DATA/datasets"
folder = os.path.join(local_storage, "raw-games", arena_label, str(game_id))


dummy = boto3.Session()
ds = RawSequencesDataset(local_storage=local_storage, progress_wrapper=tqdm, predicate=lambda k,v: k.arena_label == arena_label and k.game_id == game_id, session=dummy)
ids = InstantsDataset(ds)


DISTANCE_THRESHOLD = 28 # pixels
thresholds = {
    "pifball": PIFBALL_THRESHOLD,
    "ballseg": BALLSEG_THRESHOLD
}
models = {
    "pifball": "20220830_142415.448672",
    "ballseg": "20220829_144032.694734",
}


detectors = [Detector(model, experiment_id) for model, experiment_id in models.items()]
database = {}
callback = DelayedCallback(lambda : json.dump(database, open(os.path.join(folder, "balls.json"), "w")), timedelta=timedelta(seconds=10))

for instant_key in tqdm(ids.yield_keys()):
    instant = ids.query_item(instant_key)
    if args.break_frame and instant.sequence_frame_index > args.break_frame:
        break

    try:
        detections = [detector(instant) for detector in detectors]
    except:
        continue

    if any([detection.value < thresholds[detection.model] for detection in detections]):
        continue

    center = Point2D(np.mean([detection.point for detection in detections], axis=0))
    if any([np.linalg.norm(detection.point-center) > DISTANCE_THRESHOLD for detection in detections]):
        continue

    database[instant.sequence_frame_index] = (int(detections[0].image), int(center.x), int(center.y))
    callback()

del callback
