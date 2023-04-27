#!/usr/bin/env python
import argparse
import os
import multiprocessing
import pickle

from collections import defaultdict

print("Python Executable:", os.sys.executable)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dotenv import load_dotenv, find_dotenv
from tqdm.auto import tqdm

from pyconfyg import Confyg
from mlworkflow import TransformedDataset, FilteredDataset
from deepsport_utilities import import_dataset
from deepsport_utilities.ds.instants_dataset import InstantsDataset, DownloadFlags, CropBlockDividable, AddCalibFactory

from tasks.detection import DumpDetectedBallsFromInstants, DetectBalls

load_dotenv(find_dotenv(usecwd=True))

parser = argparse.ArgumentParser(description="""
Detect ball in Basketball Instants Dataset.""")
parser.add_argument("detector", help="Detector name (sets ball 'origin' field)")
parser.add_argument("version",  help="Detector version (sets file on which detection are saved")
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--data-augmentation", action="store_true", help="Use test-time data augmentation")
parser.add_argument("--loop", type=int, default=1, help="Repetitions of the items")
parser.add_argument("--side-length", type=int, default=256, help="Side length of heatmaps")
parser.add_argument("--k", type=int, default=4, help="Maximum number of detections to consider")
args = parser.parse_args()

dataset_folder = os.path.join(os.environ['LOCAL_STORAGE'], "basketball-instants-dataset")
dataset_config = {
    "download_flags": DownloadFlags.WITH_IMAGE | DownloadFlags.WITH_FOLLOWING_IMAGE | DownloadFlags.WITH_CALIB_FILE,
    "dataset_folder": dataset_folder
}
database_file = os.path.join(dataset_config['dataset_folder'], "basketball-instants-dataset.json")

experiment_ids = {
    "ballseg": {
        'full':      ["20230221_210804.236154", "20230221_111901.374475", "20230217_184505.715658", "20230217_140646.069593", "20230221_165826.810944", "20230220_132352.389741", "20230217_184403.495582", "20230217_140644.619392"],
        'deepsport': ["20230320_114400.066037" ,"20230320_114400.066085" ,"20230320_114036.563804" ,"20230320_114036.563858" ,"20230320_114003.086646" ,"20230320_113834.569108"],
    },
    "pifball": {
        'full': ["20230221_202338.343195", "20230220_202551.530667", "20230220_135417.554986", "20230217_150524.178161", "20230222_035443.776450", "20230221_210246.162130", "20230221_111435.426128", "20230217_150541.149283"],
    }
}[args.detector][args.version]

filename = {
    'full': "ball_detections.pickle",
    'deepsport': "deepsport_ball_detections.pickle",
}[args.version]



def process(ds, config):
    ds = TransformedDataset(ds, [CropBlockDividable()])
    detector = DumpDetectedBallsFromInstants(dataset_folder, name=args.detector, config=config, k=args.k, filename=filename, threshold=0, side_length=args.side_length)
    for instant_key in tqdm(ds.yield_keys(), leave=False):
        instant = ds.query_item(instant_key)
        detector(instant_key, instant)



def process_model(experiment_id):
    config = os.path.join(os.environ['RESULTS_FOLDER'], args.detector, experiment_id, 'config.py')

    if args.data_augmentation:
        detector = DetectBalls(name=args.detector, config=config, k=args.k, threshold=0, side_length=args.side_length)
        detector.exp.subsets[2].dataset.transforms[-1].factories.append(AddCalibFactory())
        detector.exp.subsets[2].query_item.loop = args.loop # Hacky hack to repeat the subset as if 'repetition' was set to args.loop.
        db = []
        for i in tqdm(range(args.loop)):
            for keys, data in tqdm(detector.exp.batch_generator(detector.exp.subsets[2])):
                db.append((i, list(detector(keys, data))))

        filename = os.path.join(os.environ['GLOBALSCRATCH'], f"{experiment_id}_{args.loop}_deepsport_detections_from_random_crops.pickle")
        with open(filename, 'wb') as f:
            pickle.dump(db, f)
    else:
        arena_labels = Confyg(config).dict['testing_arena_labels']
        ids = import_dataset(InstantsDataset, database_file, **dataset_config)
        ds = FilteredDataset(ids, lambda k: k.arena_label in arena_labels)

        keys_per_shape = defaultdict(list)
        for key in tqdm(ds.keys, leave=False, desc=f"Collecting shapes for {experiment_id}"):
            keys_per_shape[ds.query_item(key).images[0].shape].append(key)

        for shape, keys in tqdm(keys_per_shape.items(), leave=False, desc=f"Processing shapes of {experiment_id}"):
            print(f"Processing {len(keys)} items with shape {shape} with {experiment_id}", flush=True)
            process(FilteredDataset(ds, lambda k: k in keys), config)
            print(f"Done processing {len(keys)} items with shape {shape} with {experiment_id}", flush=True)

if args.workers > 1:
    with multiprocessing.Pool(args.workers) as pool:
        results = pool.map_async(process_model, experiment_ids,
            callback=lambda _: print("Done", flush=True),
            error_callback=lambda e: print(e, flush=True)
        )
        results.wait()
else:
    for experiment_id in tqdm(experiment_ids):
        process_model(experiment_id)
print("Done with all arenas and shapes")



