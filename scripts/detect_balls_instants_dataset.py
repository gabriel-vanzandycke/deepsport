#!/usr/bin/env python
import argparse
import os
import multiprocessing

from collections import defaultdict

print("Python Executable:", os.sys.executable)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dotenv import load_dotenv, find_dotenv
from tqdm.auto import tqdm

from pyconfyg import Confyg
from mlworkflow import TransformedDataset, FilteredDataset
from deepsport_utilities import import_dataset
from deepsport_utilities.ds.instants_dataset import InstantsDataset, DownloadFlags, CropBlockDividable

from tasks.detection import DetectBalls, PIFBALL_THRESHOLD, BALLSEG_THRESHOLD

load_dotenv(find_dotenv(usecwd=True))

parser = argparse.ArgumentParser(description="""
Detect ball in Basketball Instants Dataset.""")
parser.add_argument("detector", help="Detector name (sets ball 'origin' field)")
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--k", default=4, help="Maximum number of detections to consider")
args = parser.parse_args()

dataset_config = {
    "download_flags": DownloadFlags.WITH_IMAGE | DownloadFlags.WITH_FOLLOWING_IMAGE | DownloadFlags.WITH_CALIB_FILE,
    "dataset_folder": os.path.join(os.environ['LOCAL_STORAGE'], "basketball-instants-dataset")
}
database_file = os.path.join(dataset_config['dataset_folder'], "basketball-instants-dataset.json")

experiment_ids = {
    "ballseg": ["20230221_210804.236154", "20230221_111901.374475", "20230217_184505.715658", "20230217_140646.069593", "20230221_165826.810944", "20230220_132352.389741", "20230217_184403.495582", "20230217_140644.619392"],
    "pifball": ["20230221_202338.343195", "20230220_202551.530667", "20230220_135417.554986", "20230217_150524.178161", "20230222_035443.776450", "20230221_210246.162130", "20230221_111435.426128", "20230217_150541.149283"],
}[args.detector]

threshold = {
    "pifball": PIFBALL_THRESHOLD/2,
    "ballseg": BALLSEG_THRESHOLD/2,
}[args.detector]

ids = import_dataset(InstantsDataset, database_file, **dataset_config)

def process(ds, config):
    ds = TransformedDataset(ds, [CropBlockDividable()])
    detector = DetectBalls(ids.dataset_folder, name=args.detector, config=config, k=args.k, detection_threshold=threshold)
    for instant_key in tqdm(ds.yield_keys(), leave=False):
        instant = ds.query_item(instant_key)
        detector(instant_key, instant)

def process_model(experiment_id):
    config = os.path.join(os.environ['RESULTS_FOLDER'], args.detector, experiment_id, 'config.py')
    arena_labels = Confyg(config).dict['testing_arena_labels']
    ds = FilteredDataset(ids, lambda k: k.arena_label in arena_labels)

    keys_per_shape = defaultdict(list)
    for key in tqdm(ds.keys, leave=False, desc=f"Collecting shapes for {experiment_id}"):
        keys_per_shape[ds.query_item(key).images[0].shape].append(key)

    for shape, keys in tqdm(keys_per_shape.items(), leave=False, desc=f"Processing shapes of {experiment_id}"):
        print(f"Processing {len(keys)} items with shape {shape} with {experiment_id}", flush=True)
        process(FilteredDataset(ds, lambda k: k in keys), config)
        print(f"Done processing {len(keys)} items with shape {shape} with {experiment_id}", flush=True)


with multiprocessing.Pool(args.workers) as pool:
    results = pool.map_async(process_model, experiment_ids,
        callback=lambda _: print("Done", flush=True),
        error_callback=lambda e: print(e, flush=True)
    )
    results.wait()

print("Done with all arenas and shapes")



