#!/usr/bin/env python
import argparse
import os
print("Python Executable:", os.sys.executable)

from dotenv import load_dotenv
from tqdm.auto import tqdm

from deepsport_utilities import import_dataset
from deepsport_utilities.ds.instants_dataset import InstantsDataset, ViewsDataset, BuildBallViews, AddBallAnnotation, DownloadFlags
from dataset_utilities.ds.raw_sequences_dataset import BallState
from mlworkflow import TransformedDataset, PickledDataset
from experimentator import find

load_dotenv()

parser = argparse.ArgumentParser(description="""From the public ballistic-raw-sequences dataset, creates a dataset of
ball crops. Balls state is set to `BallState.FLYING` as the public dataset only
contains flying balls.
""")
parser.add_argument("output_folder")
parser.add_argument("--margin", type=int, default=256)
args = parser.parse_args()

dataset_folder = os.path.dirname(find("ballistic-raw-sequences/raw-basketball-sequences-dataset.json"))

dataset_config = {
    "dataset_folder": dataset_folder,
    "download_flags": DownloadFlags.WITH_ALL_IMAGES | DownloadFlags.WITH_CALIB_FILE,
}
ids = import_dataset(InstantsDataset, find("ballistic-raw-sequences/raw-basketball-sequences-dataset.json"), **dataset_config)
vds = ViewsDataset(ids, view_builder=BuildBallViews(margin=args.margin, margin_in_pixels=True))

def set_flying_state(view_key, view):
    view.ball.state = BallState.FLYING
    return view

vds = TransformedDataset(vds, [AddBallAnnotation(), set_flying_state])
PickledDataset.create(vds, os.path.join(args.output_folder, f"ballistic_ball_views_{args.margin}.pickle"), yield_keys_wrapper=tqdm)
