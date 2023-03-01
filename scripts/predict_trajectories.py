import argparse
import os

import dotenv
import numpy as np
import optuna
from optuna.integration import WeightsAndBiasesCallback
from tqdm.auto import tqdm
import wandb

from experimentator import find
from mlworkflow import PickledDataset, TransformedDataset

from tasks.ballistic import MatchTrajectories, SelectBall
from models.ballistic import TrajectoryDetector, FilteredFitter2D, Fitter2D

dotenv.load_dotenv()

parameters = {
    "min_inliers":             ("suggest_int",   {'low':  2, 'high':   8, 'step':  1}),
    "max_outliers_ratio":      ("suggest_float", {'low': .1, 'high':  .9, 'step': .1}),
    #"min_flyings":             ("suggest_int",   {'low':  0, 'high':   5, 'step':  1}),
    #"max_nonflyings_ratio":    ("suggest_float", {'low':  0, 'high':   1, 'step': .1}),
    #"max_inliers_decrease": ('fixed', .1),#("suggest_float", {'low':  0, 'high':  .2, 'step':.05}),
    "scale":                   ("suggest_float", {'low': -2, 'high':   2, 'step': .5}),
    "position_error_threshold":("suggest_int",   {'low':  1, 'high':  10, 'step':  1}),
    "d_error_weight":          ("suggest_int",   {'low':  0, 'high': 100, 'step': 10}),
    "min_distance_cm":         ("suggest_int",   {'low': 50, 'high': 200, 'step': 50}),
    "min_distance_px":         ("suggest_int",   {'low': 50, 'high': 200, 'step': 50}),
    "min_window_length":       ("suggest_int",   {'low':160, 'high': 350, 'step': 10}),
    "first_inlier":            ("suggest_int",   {'low':  1, 'high':   4, 'step':  1}),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument("positions_dataset")
    parser.add_argument("--name", default='')
    parser.add_argument("--method", default='baseline', choices=['baseline', 'usestate'])
    parser.add_argument("--min-duration", type=int, default=250)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--show-progress", action='store_true', default=False)
    parser.add_argument("--kwargs", nargs='*', default=[])
    parser.add_argument("--skip-wandb", action='store_true', default=False)
    args = parser.parse_args()

    cast = lambda k, v: (k, eval(v))
    fixed_kwargs = dict([cast(*kwarg.split('=')) for kwarg in args.kwargs])

    # if args.method == 'baseline':
    #     fixed_kwargs.update(min_flyings=0)
    #     fixed_kwargs.update(max_nonflyings_ratio=0)

    objectives = {
        's_precision': 'maximize',
        's_recall': 'maximize',
    }

    wandb_kwargs = dict(project="ballistic")
    wandb_cb = WeightsAndBiasesCallback(list(objectives.keys()), wandb_kwargs=wandb_kwargs, as_multirun=True)

    @wandb_cb.track_in_wandb()
    def objective(trial):
        trial.set_user_attr('method', args.method)
        trial.set_user_attr('job_id', os.environ.get('SLURM_JOB_ID', None))

        kwargs = fixed_kwargs.copy()
        for name, (suggest_fct, params) in parameters.items():
            if name not in kwargs:
                if suggest_fct == 'fixed':
                    kwargs.update({name: params})
                else:
                    kwargs.update({name: getattr(trial, suggest_fct)(name, **params)})

        progress_wrapper = tqdm if args.show_progress else lambda x: x

        dds = PickledDataset(find(args.positions_dataset))
        dds = TransformedDataset(dds, [SelectBall('ballseg')])

        fitter_types = (FilteredFitter2D, Fitter2D)
        sw = TrajectoryDetector(fitter_types, **kwargs)

        agen = (dds.query_item(k) for k in progress_wrapper(sorted(dds.keys)))
        pgen = sw((dds.query_item(k) for k in progress_wrapper(sorted(dds.keys))))

        compare = MatchTrajectories(min_duration=args.min_duration)
        compare(agen, pgen)

        for key, value in compare.metrics.items():
            trial.set_user_attr(key, value)
        for key, value in kwargs.items():
            trial.set_user_attr(key, value)
        wandb.log({**compare.metrics, **kwargs})

        values = tuple(compare.metrics[name] for name in objectives)
        if np.any(np.isnan(values)):
            raise optuna.TrialPruned()
        return values


    # RDBStorage with sqlite doesn't handle concurrency on NFS storage
    filename = f"ballistic_{args.name}_{args.method}.db"
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(filename,
            optuna.storages.JournalFileOpenLock(filename)
        )
    )
    study = optuna.create_study(
        study_name=args.method,
        directions=objectives.values(),
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.n_trials, callbacks=[wandb_cb])
