import argparse
import os

import dotenv
import numpy as np
import optuna
from tqdm.auto import tqdm

from experimentator import find
from mlworkflow import PickledDataset, TransformedDataset

from tasks.ballistic import NaiveSlidingWindow, BallStateSlidingWindow, MatchTrajectories, SelectBall

dotenv.load_dotenv()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument("positions_dataset")
    parser.add_argument("--method", default='baseline', choices=['baseline', 'usestate'])
    parser.add_argument("--min-duration", type=int, default=250)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--show-progress", action='store_true', default=False)
    parser.add_argument("--kwargs", nargs='*', default=[])
    args = parser.parse_args()

    # RDBStorage with sqlite doesn't handle concurrency on NFS storage
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(
            f"ballistic_{args.method}.db",
            optuna.storages.JournalFileOpenLock(f"ballistic_{args.method}.db.lock",)
        )
    )
    study = optuna.create_study(
        study_name=args.method,
        directions=['maximize', 'maximize', 'minimize'],
        storage=storage,
        load_if_exists=True,
    )

    cast = lambda k, v: (k, eval(v))
    fixed_kwargs = dict([cast(*kwarg.split('=')) for kwarg in args.kwargs])
    fixed_kwargs.update(min_window_length=args.min_duration)

    def objective(trial):
        trial.set_user_attr('method', args.method)
        trial.set_user_attr('job_id', os.environ['SLURM_JOB_ID'])

        kwargs = fixed_kwargs.copy()
        if "d_error_weight" not in kwargs:
            kwargs.update(d_error_weight=trial.suggest_int('d_error_weight', 0, 100, step=5))
        if "inliers_condition" not in kwargs:
            kwargs.update(inliers_condition='scalesin')#trial.suggest_categorical('inliers_condition', ['constant', 'scalesin']))
        if "max_outliers_ratio" not in kwargs:
            kwargs.update(max_outliers_ratio=trial.suggest_float('max_outliers_ratio', .1, .5, step=.01))
        if "max_inliers_decrease" not in kwargs:
            kwargs.update(max_inliers_decrease=trial.suggest_float('max_inliers_decrease', 0, .2, step=.01))
        if "scale" not in kwargs:
            kwargs.update(scale=trial.suggest_float('scale', 1, 2, step=.5))

        if args.method == 'baseline':
            if "min_distance_cm" not in kwargs:
                kwargs.update(min_distance_cm=100)#trial.suggest_int('min_distance_cm', 50, 100, step=25))
            if "min_distance_px" not in kwargs:
                kwargs.update(min_distance_px=100)#trial.suggest_int('min_distance_px', 25, 100, step=25))
            if "min_inliers" not in kwargs:
                kwargs.update(min_inliers=trial.suggest_int('min_inliers', 2, 8))
            if "p_error_threshold" not in kwargs:
                kwargs.update(p_error_threshold=trial.suggest_int('p_error_threshold', 2, 10, step=1))
            SlidingWindow = NaiveSlidingWindow
        elif args.method == 'usestate':
            if "min_flyings" not in kwargs:
                kwargs.update(min_flyings=trial.suggest_int('min_flyings', 1, 2))
            if "min_inliers" not in kwargs:
                kwargs.update(min_inliers=trial.suggest_int('min_inliers', 2, 3))
            if "p_error_threshold" not in kwargs:
                kwargs.update(p_error_threshold=trial.suggest_float('p_error_threshold', 1, 3, step=.1))
            SlidingWindow = BallStateSlidingWindow

        p_error_threshold = kwargs.pop('p_error_threshold')
        d_error_weight = kwargs.pop('d_error_weight')
        scale = kwargs.pop('scale')
        kwargs['inliers_condition'] = {
            "constant": lambda p_error, d_error: p_error < p_error_threshold,
            "scalesin": lambda p_error, d_error: p_error < p_error_threshold + scale*np.sin(np.linspace(0, np.pi, len(p_error))),
        }[kwargs.pop('inliers_condition')]

        kwargs['error_fct'] = lambda p_error, d_error: np.linalg.norm(p_error) + d_error_weight*np.linalg.norm(d_error)

        progress_wrapper = tqdm if args.show_progress else lambda x: x

        dds = PickledDataset(find(args.positions_dataset))
        dds = TransformedDataset(dds, [SelectBall('ballseg')])

        sw = SlidingWindow(**kwargs)

        agen = (dds.query_item(k) for k in progress_wrapper(sorted(dds.keys)))
        pgen = sw((dds.query_item(k) for k in progress_wrapper(sorted(dds.keys))))

        compare = MatchTrajectories(min_duration=args.min_duration)
        compare(agen, pgen)

        precision = len(compare.TP) / (len(compare.TP) + len(compare.FP))
        recall = len(compare.TP) / (len(compare.TP) + len(compare.FN))

        trial.set_user_attr('recovered', sum(compare.recovered))
        trial.set_user_attr('mean_dist_T0', np.mean(compare.dist_T0))
        trial.set_user_attr('dist_T0', np.mean(np.abs(compare.dist_T0)))
        trial.set_user_attr('mean_dist_TN', np.mean(compare.dist_TN))
        trial.set_user_attr('dist_TN', np.mean(np.abs(compare.dist_TN)))
        trial.set_user_attr('TP', len(compare.TP))
        trial.set_user_attr('FP', len(compare.FP))
        trial.set_user_attr('FN', len(compare.FN))
        trial.set_user_attr('precision', precision)
        trial.set_user_attr('recall', recall)
        trial.set_user_attr('splitted_predicted_trajectories', compare.splitted_predicted_trajectories)
        trial.set_user_attr('splitted_annotated_trajectories', compare.splitted_annotated_trajectories)
        trial.set_user_attr('ballistic_MAPE', compare.ballistic_MAPE)

        return precision, recall, compare.splitted_predicted_trajectories


    study.optimize(objective, n_trials=args.n_trials)
