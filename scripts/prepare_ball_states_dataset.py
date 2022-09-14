#!/usr/bin/env python
import argparse
import os
print("Python Executable:", os.sys.executable)

import boto3
from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from dataset_utilities.ds.raw_sequences_dataset import RawSequencesDataset, InstantsDataset, AddBallStatesTransform
from mlworkflow import TransformedDataset, FilteredDataset, PickledDataset

from tasks.ballstate import AddBallDetectionTransform

load_dotenv("/home/gva/repositories/deepsport/.env")

parser = argparse.ArgumentParser(description="""""")
parser.add_argument('--break-frame', type=int)
args = parser.parse_args()


local_storage = "/DATA/datasets"
dummy = boto3.Session()
ds = RawSequencesDataset(local_storage=local_storage, progress_wrapper=tqdm, session=dummy)
ds = TransformedDataset(ds, [AddBallStatesTransform()])
ds = FilteredDataset(ds, lambda k,v: v.ball_states is not None)
ids = InstantsDataset(ds)
ids = TransformedDataset(ids, [AddBallDetectionTransform()])
ids = FilteredDataset(ids, lambda k,v: v.ball2D is not None and v.ball_state is not None)
PickledDataset.create(ids, "ball_states_dataset.pickle", yield_keys_wrapper=tqdm)
