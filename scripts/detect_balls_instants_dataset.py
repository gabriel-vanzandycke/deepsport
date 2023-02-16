import argparse
import os

import dotenv
from tqdm.auto import tqdm

from mlworkflow import TransformedDataset, FilteredDataset
from deepsport_utilities import import_dataset
from deepsport_utilities.ds.instants_dataset import InstantsDataset, DownloadFlags, CropBlockDividable

from tasks.detection import DetectBalls, PIFBALL_THRESHOLD, BALLSEG_THRESHOLD

dotenv.load_dotenv()

default_shapes = ["2054,2456", "1458,1936", "2360,1992", "1234,1624", "2456,1520", "2456,1800", "1752,2336"]

parser = argparse.ArgumentParser(description="""
Detect ball in Basketball Instants Dataset.""")
parser.add_argument("name", help="Detector name (sets ball 'origin' field)")
parser.add_argument("shapes", nargs="*", default=default_shapes)
parser.add_argument("--dataset-folder", default=None, help="Basketball Instants Dataset folder")
parser.add_argument("--k", default=4, help="Maximum number of detections to consider")
args = parser.parse_args()

dataset_folder = args.dataset_folder or os.path.join(os.environ['LOCAL_STORAGE'], "basketball-instants-dataset")
dataset_config = {
    "download_flags": DownloadFlags.WITH_IMAGE | DownloadFlags.WITH_FOLLOWING_IMAGE | DownloadFlags.WITH_CALIB_FILE,
    "dataset_folder": dataset_folder  # informs dataset items of raw files location
}

database_file = os.path.join(dataset_config['dataset_folder'], "basketball-instants-dataset.json")
threshold, experiment_id = {
    "pifball": (PIFBALL_THRESHOLD, "20220830_142415.448672"),
    "ballseg": (BALLSEG_THRESHOLD, "20220829_144032.694734"),
}[args.name]
config = os.path.join(os.environ['RESULTS_FOLDER'], args.name, experiment_id, 'config_inference.py')

for shape in tqdm(args.shapes):
    ids = import_dataset(InstantsDataset, database_file, **dataset_config)
    ids = FilteredDataset(ids, lambda k,v: v.images[0].shape[0:2] == eval(shape))
    ids = TransformedDataset(ids, [CropBlockDividable()])

    detector = DetectBalls(dataset_folder, args.name, config, args.k, threshold)

    for instant_key in tqdm(ids.yield_keys()):
        instant = ids.query_item(instant_key)
        detector(instant_key, instant)
