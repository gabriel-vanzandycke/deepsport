import argparse
import os
import time

import numpy as np
from tqdm.auto import tqdm
import dotenv

from experimentator import find
from mlworkflow import PickledDataset, TransformedDataset
import optuna

from dataset_utilities.ds.raw_sequences_dataset import SequenceInstantsDataset
from tasks.ballistic import NaiveSlidingWindow, BallStateSlidingWindow, MatchTrajectories, SelectBall

dotenv.load_dotenv()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument("positions_dataset")
    parser.add_argument("--method", default='baseline', choices=['baseline', 'usestate'])
    parser.add_argument("--min-duration", type=int, default=250)
    parser.add_argument("--split-penalty", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--show-progress", action='store_true', default=False)
    args = parser.parse_args()

    progress_wrapper = tqdm if args.show_progress else lambda x: x

    sds = PickledDataset(find("raw_sequences_dataset.pickle"))
    ids = SequenceInstantsDataset(sds)

    dds = PickledDataset(find(args.positions_dataset))
    dds = TransformedDataset(dds, [SelectBall('ballseg')])

    def objective(trial):
        trial.set_user_attr('method', args.method)
        trial.set_user_attr('job_id', os.environ['SLURM_JOB_ID'])

        d_error_weight    = trial.suggest_int('d_error_weight', 1, 100)
        p_error_threshold = trial.suggest_int('p_error_threshold', 2, 10)
        kwargs = dict(
            min_window_length    = 250,
            min_inliers          = trial.suggest_int('min_inliers', 2, 8),
            max_outliers_ratio   = trial.suggest_float('max_outliers_ratio', .1, .5),
            max_inliers_decrease = trial.suggest_float('max_inliers_decrease', 0, .2),
            error_fct            = lambda p_error, d_error: np.linalg.norm(p_error) + d_error_weight*np.linalg.norm(d_error),
            inliers_condition    = lambda p_error, d_error: p_error < p_error_threshold,
        )

        if args.method == 'baseline':
            min_distance_cm = 100#trial.suggest_int('min_distance_cm', 50, 100, step=25)
            min_distance_px = 100#trial.suggest_int('min_distance_px', 25, 100, step=25)
            kwargs.update(min_distance_cm=min_distance_cm, min_distance_px=min_distance_px)
            SlidingWindow = NaiveSlidingWindow
        elif args.method == 'usestate':
            min_flyings = None#trial.suggest_int('min_flyings', 3, 8)
            kwargs.update(min_flyings=min_flyings)
            SlidingWindow = BallStateSlidingWindow

        sw = SlidingWindow(**kwargs)

        agen = (dds.query_item(k) for k in progress_wrapper(sorted(dds.keys)))
        pgen = sw((dds.query_item(k) for k in progress_wrapper(sorted(dds.keys))))

        compare = MatchTrajectories(min_duration=args.min_duration, split_penalty=args.split_penalty)
        compare(agen, pgen)

        precision = len(compare.TP) / (len(compare.TP) + len(compare.FP))
        recall = len(compare.TP) / (len(compare.TP) + len(compare.FN))

        trial.set_user_attr('method', args.method)
        trial.set_user_attr('job_id', os.environ['SLURM_JOB_ID'])
        trial.set_user_attr('recovered', sum(compare.recovered))
        trial.set_user_attr('mean_dist_T0', np.mean(compare.dist_T0))
        trial.set_user_attr('dist_T0', np.mean(np.abs(compare.dist_T0)))
        trial.set_user_attr('mean_dist_TN', np.mean(compare.dist_TN))
        trial.set_user_attr('dist_TN', np.mean(np.abs(compare.dist_TN)))
        trial.set_user_attr('TP', len(compare.TP))
        trial.set_user_attr('FP', len(compare.FP))
        trial.set_user_attr('FN', len(compare.FN))

        return precision, recall

    # RDBStorage with sqlite doesn't handle concurrency on NFS storage
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(
            f"ballistic_{args.method}.db",
            optuna.storages.JournalFileOpenLock(f"ballistic_{args.method}.db.lock",)
        )
    )
    study = optuna.create_study(
        study_name=args.method,
        directions=['maximize', 'maximize'],
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)
