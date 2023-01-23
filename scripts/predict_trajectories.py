import argparse
import os

import imageio
import numpy as np
from tqdm.auto import tqdm
import dotenv

from mlworkflow import PickledDataset, TransformedDataset
from experimentator import find

from dataset_utilities.ds.raw_sequences_dataset import SequenceInstantsDataset
from deepsport_utilities.utils import VideoMaker

from tasks.ballistic import NaiveSlidingWindow, BallStateSlidingWindow, extract_predicted_trajectories, extract_annotated_trajectories, MatchTrajectories, TrajectoryRenderer

dotenv.load_dotenv()


parser = argparse.ArgumentParser(description="""""")
parser.add_argument("positions_dataset")
parser.add_argument("--method", default='baseline', choices=['baseline', 'usestate'])
args = parser.parse_args()


sds = PickledDataset(find("raw_sequences_dataset.pickle"))
ids = SequenceInstantsDataset(sds)

dds = PickledDataset(find(args.positions_dataset))
#dds = FilteredDataset(ds, lambda k,v: len([d for d in v.ball_detections if d.origin == 'ballseg']) > 0)
def set_ball(key, item):
    try:
        item.ball = max([d for d in item.ball_detections if d.origin == 'ballseg'], key=lambda d: d.value)
        item.ball.timestamp = item.timestamps[item.ball.camera]
        item.calib = item.calibs[item.ball.camera]
    except ValueError:
        pass
    return item
dds = TransformedDataset(dds, [set_ball])


if args.method == 'baseline':
    sw = NaiveSlidingWindow(min_distance=75, min_inliers=7, max_outliers_ratio=.25, display=False,
                    error_fct=lambda p_error, d_error: np.linalg.norm(p_error) + 10*np.linalg.norm(d_error),
                    inliers_condition=lambda p_error, d_error: p_error < np.hstack([[3]*3, [5]*(len(p_error)-6), [2]*3]), tol=.1)
elif args.method == 'usestate':
    sw = BallStateSlidingWindow(min_distance=75, min_inliers=7, max_outliers_ratio=.25, display=True,
                   error_fct=lambda p_error, d_error: np.linalg.norm(p_error) + 20*np.linalg.norm(d_error),
                   inliers_condition=lambda p_error, d_error: p_error < 6, tol=1)

predicted_trajectories = extract_predicted_trajectories(sw(filter(lambda s: hasattr(s, 'ball'), (dds.query_item(k) for k in tqdm(sorted(dds.keys, key=lambda k: k.timestamp))))))
annotated_trajectories = extract_annotated_trajectories((dds.query_item(k) for k in tqdm(sorted(dds.keys, key=lambda k: k.timestamp))))



#from multiprocessing.pool import Pool
#pool = Pool(10)
#self.pool.apply_async(functools.partial())
#self.pool.close()
#self.pool.join()

renderer = TrajectoryRenderer(ids, margin=2)

def create_video(trajectory, metric):
    arena_label = trajectory.samples[0].key.arena_label
    prefix = os.path.join(os.environ['HOME'], "globalscratch", args.method, f"{metric}_{arena_label}_{trajectory.T0}")
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    with VideoMaker(prefix+".mp4") as vm:
        for img in renderer(trajectory):
            vm(img)

def FP_callback(predicted_trajectory):
    create_video(predicted_trajectory, metric="FP")

def FN_callback(annotated_trajectory):
    create_video(annotated_trajectory, metric="FN")

def TP_callback(annotated_trajectory, predicted_trajectory):
    create_video(predicted_trajectory, metric="TP")


compare = MatchTrajectories(TP_cb=TP_callback, FP_cb=FP_callback, FN_cb=FN_callback)
compare(predicted_trajectories, annotated_trajectories)
