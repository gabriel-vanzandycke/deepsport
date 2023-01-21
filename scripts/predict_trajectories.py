import argparse
import os

import imageio
import numpy as np
from tqdm.auto import tqdm
import dotenv

from mlworkflow import PickledDataset, FilteredDataset, TransformedDataset
from experimentator import find

from dataset_utilities.ds.raw_sequences_dataset import SequenceInstantsDataset
from deepsport_utilities.utils import VideoMaker

from tasks.ballistic import NaiveSlidingWindow, BallStateSlidingWindow, extract_predicted_trajectories, extract_annotated_trajectories, MatchTrajectories, Renderer

dotenv.load_dotenv()


parser = argparse.ArgumentParser(description="""""")
parser.add_argument("positions_dataset")
parser.add_argument("--method", default='baseline', choices=['baseline', 'usestate'])
args = parser.parse_args()


sds = PickledDataset(find("raw_sequences_dataset.pickle"))
ids = SequenceInstantsDataset(sds)

ds = PickledDataset(find(args.positions_dataset))
dds = FilteredDataset(ds, lambda k,v: len([d for d in v.ball_detections if d.origin == 'ballseg']) > 0)
def set_ball(key, item):
    item.ball = max([d for d in item.ball_detections if d.origin == 'ballseg'], key=lambda d: d.value)
    item.timestamp = item.timestamps[item.ball.camera]
    item.calib = item.calibs[item.ball.camera]
    return item
dds = TransformedDataset(dds, [set_ball])


if args.method == 'baseline':
    sw = NaiveSlidingWindow(min_distance=75, min_inliers=7, max_outliers_ratio=.25, display=False,
                    error_fct=lambda p_error, d_error: np.linalg.norm(p_error) + 10*np.linalg.norm(d_error),
                    inliers_condition=lambda p_error, d_error: p_error < np.hstack([[3]*3, [5]*(len(p_error)-6), [2]*3]), tol=.1)
elif args.method == 'usestate':
    sw = BallStateSlidingWindow(min_distance=75, min_inliers=7, max_outliers_ratio=.25, display=False,
                   error_fct=lambda p_error, d_error: np.linalg.norm(p_error) + 20*np.linalg.norm(d_error),
                   inliers_condition=lambda p_error, d_error: p_error < 6, tol=1)

predicted_trajectories = extract_predicted_trajectories(sw((dds.query_item(k) for k in tqdm(sorted(dds.keys, key=lambda k: k.timestamp)))))
annotated_trajectories = extract_annotated_trajectories((ds.query_item(k) for k in tqdm(sorted(ds.keys, key=lambda k: k.timestamp))))



def FP_callback(pt):
    renderer = Renderer(display=False)
    i = pt[0].ball.camera
    for sample in pt:
        instant = ids.query_item(sample.key)
        renderer(sample.timestamp, instant.images[i], instant.calibs[i], sample)
    filename = os.environ['HOME']+f"/globalscratch/{args.method}/FP/{sample.key.arena_label}_{pt.T0}.png"
    imageio.imwrite(filename, renderer.canvas)
    print(filename, "written")

def FN_callback(at):
    filename = os.environ['HOME']+f"/globalscratch/{args.method}/FN/{at[0].key.arena_label}_{at.T0}.mp4"
    with VideoMaker(filename) as vm:
        for sample in at:
            instant = ids.query_item(sample.key)
            vm(np.hstack(instant.images))
    print(filename, "written")

def TP_callback(at, pt):
    renderer = Renderer(display=False)
    i = pt[0].ball.camera
    filename = os.environ['HOME']+f"/globalscratch/{args.method}/TP/{at[0].key.arena_label}_{at.T0}.mp4"
    a = {s.key: s for s in at}
    p = {s.key: s for s in pt}
    keys = sorted(set(a.keys()).union(set(p.keys())))
    with VideoMaker(filename) as vm:
        for key in keys:
            instant = ids.query_item(key)
            if key in p:
                renderer(p[key].timestamp, instant.images[i], instant.calibs[i], p[key])
            vm(instant.images[i])
    filename = os.environ['HOME']+f"/globalscratch/{args.method}/TP/{at[0].key.arena_label}_{at.T0}.png"
    imageio.imwrite(filename, renderer.canvas)
    print(filename, "written")


compare = MatchTrajectories(TP_cb=TP_callback, FP_cb=FP_callback, FN_cb=FN_callback)
compare(predicted_trajectories, annotated_trajectories)
