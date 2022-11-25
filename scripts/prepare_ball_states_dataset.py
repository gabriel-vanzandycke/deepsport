#!/usr/bin/env python
import argparse
import os
print("Python Executable:", os.sys.executable)

import boto3
from dotenv import load_dotenv
from tqdm.auto import tqdm

from dataset_utilities.ds.raw_sequences_dataset import RawSequencesDataset, SequenceInstantsDataset, BallState
from deepsport_utilities.ds.instants_dataset import ViewsDataset, BuildBallViews
from mlworkflow import TransformedDataset, FilteredDataset, PickledDataset

from experimentator import find

from tasks.ballstate import AddBallDetectionsTransform

load_dotenv()

parser = argparse.ArgumentParser(description="""From the (private) Keemotion raw-sequences dataset, creates a dataset of
ball crops.
    - ball positions are provided by `<arena_label>/<game_id>/balls.json` files (can be detections provided by
      `scripts/process_raw_sequences.py`)
    - ball states are provided by `<arena_label>/<game_id>/ball_states.csv` files (exported from BORIS annotation tool)
Each dataset `View` item has a `ball` attribute with the following attributes:
    - state: a `BallState` enum
    - center: a `Point3D` (with Z=0 if ball position is given in the image space)
""")
parser.add_argument("output_folder")
parser.add_argument("--local-storage", default=os.environ.get("LOCAL_STORAGE"))
args = parser.parse_args()

dummy = boto3.Session()
sds = RawSequencesDataset(local_storage=args.local_storage, session=dummy, progress_wrapper=tqdm)
ds = FilteredDataset(sds, lambda k,v: len(list(v.ball_states)) > 0) # only keep sequences on which ball state was annotated with BORIS
ids = SequenceInstantsDataset(ds)

class ReplaceBallAnnotationsTransform:
    def __init__(self, ballistic_trajectories_dataset_filename):
        ds = PickledDataset(find(ballistic_trajectories_dataset_filename))
        self.lookuptable = {}
        for k in tqdm(ds.keys, desc="building annotations lookuptable"):
            instants = ds.query_item(k)
            for instant in instants:
                annotation = instant in [instants[0], instants[-1]]
                instant.ball.origin = "annotation" if annotation else "interpolation"
                self.lookuptable[instant.arena_label, instant.game_id, instant.timestamp] = instant.ball
    def __call__(self, sequence_instant_key, sequence_instant):
        key = sequence_instant_key[0:3]
        ball = self.lookuptable.get(key, None)
        if ball is not None:
            sequence_instant.annotations = [ball]
        return sequence_instant

ids = TransformedDataset(ids, [
    AddBallDetectionsTransform(dataset_folder=sds.dataset_folder, xy_inverted=True),
    ReplaceBallAnnotationsTransform("instants_ballistic_trajectories_600ms_filtered.pickle"),
])

ids = FilteredDataset(ids, lambda k,v: (bool(v.annotations) or bool(v.detections)) and v.ball_state is not BallState.NONE)
origins = ['annotation', 'detection', 'pseudo-annotation', 'interpolation']
vds = ViewsDataset(ids, view_builder=BuildBallViews(origins=origins, margin=128, margin_in_pixels=True))

PickledDataset.create(vds, os.path.join(args.output_folder, "ball_states_dataset_with_annotations.pickle"), yield_keys_wrapper=tqdm)
