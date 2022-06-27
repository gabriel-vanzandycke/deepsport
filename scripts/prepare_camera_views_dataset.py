import argparse
import os
from tqdm.auto import tqdm
from mlworkflow import PickledDataset, FilteredDataset, TransformedDataset
from deepsport_utilities import import_dataset
from deepsport_utilities.ds.instants_dataset import InstantsDataset, DownloadFlags, ViewsDataset, BuildCameraViews, AddBallAnnotation

parser = argparse.ArgumentParser(description="""
Creates an mlworkflow.PickledDataset file named "camera_with_ball_visible_views.pickle" containing pairs of ViewKey, View objects.
The View objects contain full image (with calibration data) as captured by the Keemotion system.
""")
parser.add_argument("--dataset-folder", required=True, help="Basketball Instants Dataset folder")
parser.add_argument("--output-folder", default=None, help="Folder in which specific dataset will be created. Defaults to `dataset_folder` given in arguments.")
args = parser.parse_args()


# The `dataset_config` is used to create each dataset item
dataset_config = {
    "download_flags": DownloadFlags.WITH_IMAGE | DownloadFlags.WITH_CALIB_FILE | DownloadFlags.WITH_FOLLOWING_IMAGE,
    "dataset_folder": args.dataset_folder  # informs dataset items of raw files location
}

# Import dataset
database_file = os.path.join(args.dataset_folder, "basketball-instants-dataset.json")
ds = import_dataset(InstantsDataset, database_file, **dataset_config)

# Transform the dataset of instants into a dataset of views for each camera
ds = ViewsDataset(ds, view_builder=BuildCameraViews())

# Add the 'ball' attribute to the views, a shortcut to the ball in the annotation list
ds = TransformedDataset(ds, [AddBallAnnotation()])

# Filter only views for which camera index is the one in which the ball was annotated
ds = FilteredDataset(ds, predicate=lambda k,v: k.camera == v.ball.camera)

# Save the working dataset to disk with data contiguously stored for efficient reading during training
output_folder = args.output_folder or args.dataset_folder
path = os.path.join(output_folder, "camera_with_ball_visible_views.pickle")
PickledDataset.create(ds, path, yield_keys_wrapper=tqdm)
print(f"Successfully generated {path}")
