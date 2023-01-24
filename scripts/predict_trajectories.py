import argparse
import functools
from multiprocessing.pool import Pool
import os

import numpy as np
from tqdm.auto import tqdm
import dotenv

from mlworkflow import PickledDataset, TransformedDataset
from experimentator import find

from dataset_utilities.ds.raw_sequences_dataset import SequenceInstantsDataset
from deepsport_utilities.utils import VideoMaker

from tasks.ballistic import NaiveSlidingWindow, BallStateSlidingWindow, MatchTrajectories, TrajectoryRenderer, SelectBall

dotenv.load_dotenv()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument("positions_dataset")
    parser.add_argument("--method", default='baseline', choices=['baseline', 'usestate'])
    args = parser.parse_args()


    sds = PickledDataset(find("raw_sequences_dataset.pickle"))
    ids = SequenceInstantsDataset(sds)

    dds = PickledDataset(find(args.positions_dataset))
    dds = TransformedDataset(dds, [SelectBall('ballseg')])

    if args.method == 'baseline':
        sw = NaiveSlidingWindow(min_distance=75, min_inliers=7, max_inliers_decrease=.1, max_outliers_ratio=.25, display=False,
                        error_fct=lambda p_error, d_error: np.linalg.norm(p_error) + 10*np.linalg.norm(d_error),
                        inliers_condition=lambda p_error, d_error: p_error < np.hstack([[3]*3, [5]*(len(p_error)-6), [2]*3]), tol=.1)
    elif args.method == 'usestate':
        sw = BallStateSlidingWindow(min_distance=75, max_inliers_decrease=.1, window_size=8, min_flyings=5, min_inliers=7, max_outliers_ratio=.25, display=False,
                    error_fct=lambda p_error, d_error: np.linalg.norm(p_error) + 20*np.linalg.norm(d_error),
                    inliers_condition=lambda p_error, d_error: p_error < 6, tol=1)

    agen = (dds.query_item(k) for k in tqdm(sorted(dds.keys, key=lambda k: k.timestamp)))
    pgen = sw(dds.query_item(k) for k in tqdm(sorted(dds.keys, key=lambda k: k.timestamp)))

    pool = Pool(10)
    renderer = TrajectoryRenderer(ids, margin=2)

    def create_video(trajectory, metric):
        def f():
            key = trajectory.start_key
            prefix = os.path.join(os.environ['HOME'], "globalscratch", args.method, f"{metric}_{key.arena_label}_{key.timestamp}")
            os.makedirs(os.path.dirname(prefix), exist_ok=True)

            with VideoMaker(prefix+".mp4") as vm:
                for img in renderer(trajectory):
                    vm(img)
        #pool.apply_async(f)
        f()

    def FP_callback(annotated_trajectory, predicted_trajectory):
        create_video(predicted_trajectory, metric="FP")

    def FN_callback(annotated_trajectory, predicted_trajectory):
        create_video(annotated_trajectory, metric="FN")

    def TP_callback(annotated_trajectory, predicted_trajectory):
        create_video(predicted_trajectory, metric="TP")

    compare = MatchTrajectories(TP_cb=TP_callback, FP_cb=FP_callback, FN_cb=FN_callback)
    compare(agen, pgen)

    pool.close()
    pool.join()
