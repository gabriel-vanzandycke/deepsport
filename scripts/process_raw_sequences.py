#!/usr/bin/env python
import argparse
import contextlib
from datetime import timedelta, datetime
import pickle
import os
import sys
print("Python Executable:", sys.executable)

import boto3
import cv2
from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf

from dataset_utilities.ds.raw_sequences_dataset import RawSequencesDataset, SequenceInstantsDataset
from deepsport_utilities.utils import DelayedCallback, VideoMaker

from tasks.ballstate import Detector, PIFBALL_THRESHOLD, BALLSEG_THRESHOLD

load_dotenv()

parser = argparse.ArgumentParser(description="""Process Sequence to detect ball""")
parser.add_argument("arena_label")
parser.add_argument("game_id", type=int)
parser.add_argument('--break-frame', type=int)
parser.add_argument('--skip-video', action='store_true')
args = parser.parse_args()


predicate = lambda k,v: k.arena_label == args.arena_label and k.game_id == args.game_id
dummy = boto3.Session()
sds = RawSequencesDataset(progress_wrapper=tqdm, predicate=predicate, session=dummy)
ids = SequenceInstantsDataset(sds, tol=None) # tolerence set to None, ignoring delay between streams, as annotations were initially performed on such.


thresholds = {
    "pifball": PIFBALL_THRESHOLD,
    "ballseg": BALLSEG_THRESHOLD
}
models = {
    "pifball": "20220830_142415.448672",
    "ballseg": "20220829_144032.694734",
}

folder = os.path.join(sds.dataset_folder, args.arena_label, str(args.game_id))
ball_file = os.path.join(folder, "balls3d_new.pickle")

assert len(tf.config.list_physical_devices('GPU')) > 0, "A GPU is required to detect balls"

detectors = [Detector(model, os.path.join(os.environ['RESULTS_FOLDER'], model, experiment_id, "config_inference.py"), k=[4]) for model, experiment_id in models.items()]

database = {}
save_balls_callback = DelayedCallback(lambda : pickle.dump(database, open(ball_file, "wb")), timedelta=timedelta(seconds=10))

#import logging
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

concatenated_filename = os.path.join(folder, f"{args.arena_label}_{args.game_id}_concatenated.mp4")

if args.skip_video:
    cm = contextlib.nullcontext()
else:
    cm = VideoMaker(concatenated_filename)

from mlworkflow import SideRunner
sr = SideRunner()


def print_text(img, text, scale, thickness, color='white'):
    color = {
        "green": (0,255,0),
        "white": (255,255,255),
    }.get(color, color)
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, scale, thickness)[0]
    cv2.putText(img, text, ((img.shape[1]-textsize[0]+thickness)//2, int(img.shape[0]*0.1)), font, scale, color, thickness, bottomLeftOrigin=False)


with cm as vm:
    for instant_key in tqdm(sr.yield_async(ids.yield_keys())):
        instant = ids.query_item(instant_key)
        if args.break_frame and instant.frame_indices[0] > args.break_frame:
            break

        detections = [d for detector in detectors for d in detector(instant) if d.value > thresholds[d.model]]
        database[instant.frame_indices[0]] = detections

        for detection in detections:
            model, camera_idx, point2D, value = detection
            x, y = point2D.to_int_tuple()
            instant.images[camera_idx][y-1:y+1,:] = np.array([0,255,0])
            instant.images[camera_idx][:,x-1:x+1] = np.array([0,255,0])


        if not args.skip_video:
            img = np.hstack(instant.images)
            timestamp_str = str(datetime.fromtimestamp(instant.timestamp/1000.0))
            print_text(img, timestamp_str, 3, 20)
            print_text(img, timestamp_str, 3, 5, 'green')
            vm(img)

        save_balls_callback()

del save_balls_callback


