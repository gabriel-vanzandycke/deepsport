import argparse
import multiprocess as mp

import numpy as np
from tqdm.auto import tqdm
import dotenv

from experimentator import find
from mlworkflow import PickledDataset, TransformedDataset, SideRunner
import optuna

from dataset_utilities.ds.raw_sequences_dataset import SequenceInstantsDataset
from deepsport_utilities.utils import VideoMaker

from tasks.ballistic import NaiveSlidingWindow, BallStateSlidingWindow, MatchTrajectories, TrajectoryRenderer, SelectBall

dotenv.load_dotenv()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument("positions_dataset")
    parser.add_argument("--create-video", action='store_true', default=False)
    parser.add_argument("--method", default='baseline', choices=['baseline', 'usestate'])
    parser.add_argument("--min-size", type=int, default=7)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--show-progress", action='store_true', default=False)
    args = parser.parse_args()

    sds = PickledDataset(find("raw_sequences_dataset.pickle"))
    ids = SequenceInstantsDataset(sds)

    dds = PickledDataset(find(args.positions_dataset))
    dds = TransformedDataset(dds, [SelectBall('ballseg')])

    SlidingWindow = {
        'baseline': NaiveSlidingWindow,
        'usestate': BallStateSlidingWindow
    }[args.method]

    def objective(trial):
        window_size = 10#trial.suggest_int('window_size', 7, 10)
        tol = 1#trial.suggest_float('tol', .001, 10, log=True)
        min_inliers = None#trial.suggest_int('min_inliers', 5, 7)
        max_outliers_ratio = trial.suggest_float('max_outliers_ratio', .1, .5, step=.1)
        max_inliers_decrease = trial.suggest_float('max_inliers_decrease', 0, .2, step=.05)
        d_error_weight = 20#trial.suggest_int('d_error_weight', , 50, step=10)
        p_error_threshold = trial.suggest_int('p_error_threshold', 4, 10, step=1)

        error_fct = lambda p_error, d_error: np.linalg.norm(p_error) + d_error_weight*np.linalg.norm(d_error)
        inliers_condition = lambda p_error, d_error: p_error < p_error_threshold

        kwargs = dict(window_size=window_size, tol=tol, min_inliers=min_inliers,
            max_outliers_ratio=max_outliers_ratio, max_inliers_decrease=max_inliers_decrease,
            error_fct=error_fct, inliers_condition=inliers_condition)

        if args.method == 'usestate':
            min_flyings = None#trial.suggest_int('min_flyings', 3, window_size)
            kwargs.update(min_flyings=min_flyings)
        elif args.method == 'baseline':
            min_distance_cm = 100#trial.suggest_int('min_distance_cm', 50, 100, step=25)
            min_distance_px = 100#trial.suggest_int('min_distance_px', 25, 100, step=25)
            kwargs.update(min_distance_cm=min_distance_cm, min_distance_px=min_distance_px)

        sw = SlidingWindow(**kwargs)

        compare = MatchTrajectories(min_size=args.min_size)

        agen = (dds.query_item(k) for k in tqdm(sorted(dds.keys)))
        pgen = sw((dds.query_item(k) for k in tqdm(sorted(dds.keys))))
        compare(agen, pgen)

        return len(compare.TP), len(compare.FP), len(compare.FN)

    study = optuna.create_study(
        study_name=args.method,
        directions=['maximize', 'minimize', 'minimize'],
        storage="sqlite:///ballistic.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=args.show_progress)
