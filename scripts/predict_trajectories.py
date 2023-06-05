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

from tasks.ballistic import TrajectoryBasedEvaluation, SampleBasedEvaluation#, PickDetection#SelectBall,
from models.ballistic import TrajectoryDetector, FilterBallStateSolution, InliersFiltersSolution, PositionAndDiameterFitter2D, FilterInliers, PositionFitter2D, FirstPositionNullSpeedSolution

dotenv.load_dotenv()

parameters = {
    "min_inliers":             ("suggest_int",   {'low':  2, 'high':   8, 'step':  1}),
    "max_outliers_ratio":      ("suggest_float", {'low': .0, 'high':  .9, 'step': .1}),
    "min_flyings":             ('fixed', 1),#("suggest_int",   {'low':  0, 'high':   5, 'step':  1}),
    "max_nonflyings_ratio":    ("suggest_float", {'low':  0, 'high':   1, 'step': .1}),
    #"max_inliers_decrease": ('fixed', .1),#("suggest_float", {'low':  0, 'high':  .2, 'step':.05}),
    "position_error_threshold":("suggest_int",   {'low':  1, 'high':  10, 'step':  1}),
    "d_error_weight":          ("suggest_int",   {'low':  0, 'high': 100, 'step': 10}),
    "min_distance_cm":         ("fixed", 100),#  {'low': 50, 'high': 200, 'step': 50}),
    "min_distance_px":         ("fixed", 100),#  {'low': 50, 'high': 200, 'step': 50}),
    "min_window_length":       ("suggest_int",   {'low':160, 'high': 450, 'step': 10}),
    "first_inlier":            ('fixed', 1),#("suggest_int",   {'low':  1, 'high':   4, 'step':  1}),
    "retries":                 ("suggest_int",   {'low':  0, 'high':   4, 'step':  1}),
    "ftol":                    ("suggest_float", {'low':  1e-6, 'high':  100, 'log': True}),
    "xtol":                    ("suggest_float", {'low':  1e-6, 'high':  100, 'log': True}),
    "gtol":                    ("suggest_float", {'low':  1e-6, 'high':  100, 'log': True}),
    "tol":                     ("suggest_float", {'low':  1e-6, 'high':  100, 'log': True}),
    "distance_threshold":      ("suggest_float", {'low':  1, 'high':  20, 'step':  1}),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument("positions_dataset")
    parser.add_argument("--name", default='')
    parser.add_argument("--method", default='baseline', choices=['baseline', 'usestate', 'twostages'])
    parser.add_argument("--min-duration", type=int, default=250)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--show-progress", action='store_true', default=False)
    parser.add_argument("--kwargs", nargs='*', default=[])
    parser.add_argument("--skip-wandb", action='store_true', default=False)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    cast = lambda k, v: (k, eval(v))
    fixed_kwargs = dict([cast(*kwarg.split('=')) for kwarg in args.kwargs])

    objectives = {
        'trajectory_sample_precision': 'maximize',
        'trajectory_sample_recall': 'maximize',
        'sample_ballistic_restricted_MAPE': 'minimize',
    }

    if args.method == 'baseline':
        for name in ['min_flyings', 'max_nonflyings_ratio']:
            del parameters[name]

    fitter_types = {
        #'baseline': (FilteredFitter2D, Fitter2D),
        #'usestate': (UseStateFilteredFitter2D, FilteredFitter2D, Fitter2D)
        'twostages': (FilterBallStateSolution, InliersFiltersSolution, PositionAndDiameterFitter2D, FilterInliers, PositionFitter2D, FirstPositionNullSpeedSolution)
    }[args.method]

    progress_wrapper = tqdm if args.show_progress else lambda x: x

    wandb_kwargs = dict(project="ballistic", config={
        'method': args.method,
        'job_id': os.environ.get('SLURM_JOB_ID', None),
        'group_name': args.name,
        'dataset': args.positions_dataset,
    })
    wandb_cb = WeightsAndBiasesCallback(list(objectives.keys()), wandb_kwargs=wandb_kwargs, as_multirun=True)

    @wandb_cb.track_in_wandb()
    def objective(trial):
        # Sample parameters
        kwargs = fixed_kwargs.copy()
        for name, (suggest_fct, params) in parameters.items():
            if name not in kwargs:
                if suggest_fct == 'fixed':
                    value = params if isinstance(params, (int, float)) else params['low']
                    kwargs.update({name: value})
                else:
                    kwargs.update({name: getattr(trial, suggest_fct)(name, **params)})

        # Record parameters
        trial.set_user_attr('method', args.method)
        trial.set_user_attr('job_id', os.environ.get('SLURM_JOB_ID', None))
        trial.set_user_attr('dataset', args.positions_dataset)
        for key, value in kwargs.items():
            trial.set_user_attr(key, value)
            wandb.config[key] = value

        # Process sequence
        dds = PickledDataset(find(args.positions_dataset))
        agen = (dds.query_item(k) for i, k in enumerate(sorted(dds.keys)) if i >= args.start)
        pgen = (dds.query_item(k) for i, k in enumerate(sorted(dds.keys)) if i >= args.start)
        sw = TrajectoryDetector(fitter_types, **kwargs)
        compare = SampleBasedEvaluation(min_duration=args.min_duration)
        mt = TrajectoryBasedEvaluation(args.min_duration, None)
        mt(agen, compare(sw(pgen)))

        metrics = {prefix+key: value for prefix, data in [('sample_', compare.metrics), ('trajectory_', mt.metrics)] for key, value in data.items()}
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        wandb.log(metrics)

        # Return objectives
        values = tuple(metrics[name] for name in objectives)
        if np.any(np.isnan(values)):
            raise optuna.TrialPruned()
        return values


    # Using 'JournalStorage' because 'RDBStorage' with sqlite doesn't handle concurrency on NFS storage
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
