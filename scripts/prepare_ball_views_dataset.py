import argparse
from dotenv import load_dotenv
import os
from tqdm.auto import tqdm
from mlworkflow import PickledDataset, TransformedDataset
from deepsport_utilities import import_dataset
from deepsport_utilities.ds.instants_dataset import InstantsDataset, DownloadFlags, ViewsDataset, BuildBallViews, AddBallAnnotation

load_dotenv()

parser = argparse.ArgumentParser(description="""
Creates an mlworkflow.PickledDataset file named "ball_views.pickle" containing pairs of ViewKey, View objects of balls centered
on a 128x128 pixels image. The View objects contain image calibration data and ball annotation in 3D coordinates.
""")
parser.add_argument("--dataset-folder", default=os.path.join(os.environ['LOCAL_STORAGE'], 'basketball-instants-dataset'), help="Basketball Instants Dataset folder")
parser.add_argument("--output-folder", default=None, help="Folder in which specific dataset will be created. Defaults to `dataset_folder` given in arguments.")
args = parser.parse_args()


# The `dataset_config` is used to create each dataset item
dataset_config = {
    "download_flags": DownloadFlags.WITH_IMAGE | DownloadFlags.WITH_FOLLOWING_IMAGE | DownloadFlags.WITH_CALIB_FILE,
    "dataset_folder": args.dataset_folder  # informs dataset items of raw files location
}

# Import dataset
database_file = os.path.join(args.dataset_folder, "basketball-instants-dataset.json")
ds = import_dataset(InstantsDataset, database_file, **dataset_config)

# build a dataset of balls centered in the image with a margin of 100px around the ball, filling image with 0 if needed.
ds = ViewsDataset(ds, BuildBallViews(margin=128, margin_in_pixels=True))

# Add the 'ball' attribute to the views, a shortcut to the ball in the annotation list
ds = TransformedDataset(ds, [AddBallAnnotation()])

# Save the working dataset to disk with data contiguously stored for efficient reading during training
output_folder = args.output_folder or args.dataset_folder
path = os.path.join(output_folder, "ball_views.pickle")
PickledDataset.create(ds, path, yield_keys_wrapper=tqdm)
print(f"Successfully generated {path}")
