#!/usr/bin/env python
import argparse
import contextlib
from datetime import timedelta, datetime
import json
import os
print("Python Executable:", os.sys.executable)

import boto3
import cv2
from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from calib3d import Point2D
from dataset_utilities.ds.raw_sequences_dataset import RawSequencesDataset, InstantsDataset
from deepsport_utilities.utils import DelayedCallback, VideoMaker

from tasks.ballstate import Detector, PIFBALL_THRESHOLD, BALLSEG_THRESHOLD

load_dotenv("/home/gva/repositories/deepsport/.env")

parser = argparse.ArgumentParser(description="""Process Sequence to detect ball""")
parser.add_argument("arena_label")
parser.add_argument("game_id", type=int)
parser.add_argument('--break-frame', type=int)
parser.add_argument('--transcode', action='store_true')
args = parser.parse_args()


arena_label = args.arena_label
game_id = args.game_id
local_storage = "/DATA/datasets"
folder = os.path.join(local_storage, "raw-games", arena_label, str(game_id))

predicate = lambda k,v: k.arena_label == arena_label and k.game_id == game_id
dummy = boto3.Session()
ds = RawSequencesDataset(local_storage=local_storage, progress_wrapper=tqdm, predicate=predicate, session=dummy)
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
save_balls_callback = DelayedCallback(lambda : json.dump(database, open(os.path.join(folder, "balls.json"), "w")), timedelta=timedelta(seconds=10))

def detect_ball(instant):
    try:
        detections = [detector(instant) for detector in detectors]
    except:
        return None
    if any([detection.value < thresholds[detection.model] for detection in detections]):
        return None
    center = Point2D(np.mean([detection.point for detection in detections], axis=0))
    if any([np.linalg.norm(detection.point-center) > DISTANCE_THRESHOLD for detection in detections]):
        return None
    database[instant.sequence_frame_index] = (int(detections[0].image), int(center.x), int(center.y))
    return database[instant.sequence_frame_index]

def print_text(img, text, scale, thickness, color='white'):
    color = {
        "green": (0,255,0),
        "white": (255,255,255),
    }.get(color, color)
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, scale, thickness)[0]
    cv2.putText(img, text, ((img.shape[1]-textsize[0]+thickness)//2, int(img.shape[0]*0.1)), font, scale, color, thickness, bottomLeftOrigin=False)



concatenated_filename = os.path.join(folder, f"{arena_label}_{game_id}_concatenated.mp4")
if args.transcode:
    cm = VideoMaker(concatenated_filename)
else:
    cm = contextlib.nullcontext()
with cm as vm:
    for instant_key in tqdm(ids.yield_keys()):
        instant = ids.query_item(instant_key)
        if args.break_frame and instant.sequence_frame_index > args.break_frame:
            break

        ball = detect_ball(instant)

        if args.transcode:
            if ball is not None:
                camera_idx, x, y = ball
                instant.images[camera_idx][x-1:x+1,:] = np.array([0,255,0])
                instant.images[camera_idx][:,y-1:y+1] = np.array([0,255,0])

            img = np.hstack(instant.images)

            timestamp_str = str(datetime.fromtimestamp(instant.timestamp/1000.0))
            print_text(img, timestamp_str, 3, 20)
            print_text(img, timestamp_str, 3, 5, 'green')
            vm(img)

        save_balls_callback()

del save_balls_callback


