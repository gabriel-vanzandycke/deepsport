import argparse
import os
from tqdm.auto import tqdm
from mlworkflow import PickledDataset, FilteredDataset, TransformedDataset
from deepsport_utilities import import_dataset
from deepsport_utilities.ds.instants_dataset import InstantsDataset, DownloadFlags, ViewsDataset, BuildBallViews, AddBallAnnotation

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-folder", help="Basketball Instants Dataset folder")
parser.add_argument("--output-folder", help="Folder in which specific dataset will be created")
args = parser.parse_args()


# The `dataset_config` is used to create each dataset item
dataset_config = {
    "download_flags": DownloadFlags.WITH_IMAGE | DownloadFlags.WITH_CALIB_FILE,
    "dataset_folder": args.dataset_folder  # informs dataset items of raw files location
}

# Import dataset
database_file = os.path.join(args.dataset_folder, "basketball-instants-dataset.json")
ds = import_dataset(InstantsDataset, database_file, **dataset_config)

# build a dataset of balls centered in the image with a margin of 100cm around the ball
ds = ViewsDataset(ds, BuildBallViews(margin=100))

# Add the 'ball' attribute to the views, a shortcut to the ball in the annotation list
ds = TransformedDataset(ds, [AddBallAnnotation()])

# Save the working dataset to disk with data contiguously stored for efficient reading during training
PickledDataset.create(ds, os.path.join(args.output_folder, "ball_views.pickle"), yield_keys_wrapper=tqdm)

