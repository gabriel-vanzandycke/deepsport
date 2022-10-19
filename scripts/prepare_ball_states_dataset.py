#!/usr/bin/env python
import argparse
import os
print("Python Executable:", os.sys.executable)

import boto3
from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from calib3d import Point2D
from dataset_utilities.ds.raw_sequences_dataset import RawSequencesDataset, InstantsDataset, AddBallStatesTransform, BallState
from deepsport_utilities.ds.instants_dataset import ViewsDataset, BuildBallViews, BallAnnotation, AddBallAnnotation
from mlworkflow import TransformedDataset, FilteredDataset, PickledDataset

from tasks.ballstate import AddBallDetectionTransform

load_dotenv("/home/gva/repositories/deepsport/.env")

parser = argparse.ArgumentParser(description="""From the (private) Keemotion raw-sequences dataset create a dataset of
ball crops.
    - ball positions are provided by `<arena_label>/<game_id>/balls.json` files (can be detections provided by
      `scripts/process_raw_sequences.py`)
    - ball states are provided by `<arena_label>/<game_id>/ball_states.csv` files (should be annotations provided by
      BORIS annotation tool)
Each dataset `View` item has a `ball` attribute with the following attributes:
    - state: a `BallState` enum
    - center: a `Point3D` (with Z=0 if ball position is given in the image space)
""")
parser.add_argument("output_folder")
args = parser.parse_args()

local_storage = "/DATA/datasets"
dummy = boto3.Session()
ds = RawSequencesDataset(local_storage=local_storage, progress_wrapper=tqdm, session=dummy)
ds = TransformedDataset(ds, [AddBallStatesTransform()])
ds = FilteredDataset(ds, lambda k,v: v.ball_states is not None)
ids = InstantsDataset(ds)

def convert_ball_format(_, instant):
    camera_idx = instant.ball2D[0]
    ball3D = instant.calibs[camera_idx].project_2D_to_3D(Point2D(instant.ball2D[2], instant.ball2D[1]), Z=0)
    instant.annotations = [BallAnnotation({'center': ball3D, 'visible': True, 'image': camera_idx, 'state': instant.ball_state})]
    return instant

ids = TransformedDataset(ids, [AddBallDetectionTransform()])
ids = FilteredDataset(ids, lambda k,v: v.ball2D is not None and v.ball_state is not BallState.NONE)
ids = TransformedDataset(ids, [convert_ball_format])
vds = ViewsDataset(ids, view_builder=BuildBallViews(margin=128, margin_in_pixels=True))
vds = TransformedDataset(vds, [AddBallAnnotation()])

PickledDataset.create(vds, os.path.join(args.output_folder, "ball_states_dataset.pickle"), yield_keys_wrapper=tqdm)
