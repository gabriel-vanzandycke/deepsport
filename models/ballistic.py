from dataclasses import dataclass, field
from functools import cached_property
import random

from calib3d import Point3D, Point2D
import numpy as np
import scipy.optimize

from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.utils import setdefaultattr
from deepsport_utilities.ds.instants_dataset import InstantsDataset, BallState, Ball
from deepsport_utilities.dataset import Subset


g = 9.81 * 100 /(1000*1000) # m/s² => cm/ms²

def project_3D_to_2D(P, points3D):
    points2D_H = np.einsum('bij,jb->ib', P, points3D.H)
    return Point2D(points2D_H) * np.sign(points2D_H[2])

def compute_length2D(K, RT, points3D, length, points2D=None):
    if points2D is None:
        points2D_H = np.einsum('bij,bjk,kb->ib', K, RT, points3D.H)
        points2D = Point2D(points2D_H) * np.sign(points2D_H[2])
    points3D_c = Point3D(np.einsum('bij,jb->ib', RT, points3D.H)) # Point3D expressed in camera coordinates system
    points3D_c.x += length # add the 3D length to one of the componant
    points2D_H = np.einsum('bij,jb->ib', K, points3D_c) # go in the 2D world
    answer = np.linalg.norm(points2D - Point2D(points2D_H) * np.sign(points2D_H[2]), axis=0)
    return answer


class BallisticModel():
    def __init__(self, initial_condition, T0):
        self.initial_condition = x0, y0, z0, vx0, vy0, vz0 = initial_condition
        self.p0 = Point3D(x0, y0, z0)
        self.v0 = Point3D(vx0, vy0, vz0)
        self.a0 = Point3D(0, 0, g)
        self.T0 = T0

    def __call__(self, t):
        t = t - self.T0
        return self.p0 + self.v0*t + self.a0*t**2/2


class Window(tuple):
    def __new__(cls, args, popped):
        self = super().__new__(cls, tuple(args))
        self.popped = popped
        return self
    @cached_property
    def indices(self):
        return [i for i, s in enumerate(self) if hasattr(s, 'ball') and hasattr(s, 'calib')]
    @cached_property
    def timestamps(self):
        return np.array([self[i].timestamp for i in self.indices])
    @cached_property
    def points3D(self):
        return Point3D([self[i].ball.center for i in self.indices])
    @cached_property
    def P(self):
        return np.stack([self[i].calib.P for i in self.indices])
    @cached_property
    def RT(self):
        return np.stack([np.hstack([self[i].calib.R, self[i].calib.T]) for i in self.indices])
    @cached_property
    def K(self):
        return np.stack([self[i].calib.K for i in self.indices])
    #def __str__(self):
        #return " "*self.popped + "|" + " ".join([repr_dict[s.ball_state == BallState.FLYING, inliers_mask[i]] for i, s in enumerate(window)]), end="|\t")


# MSE solution of the 3D problem
class Fitter3D:
    def __call__(self, window):
        timestamps = window.timestamps
        if not timestamps.size:
            return None
        T0 = window.timestamps[0]
        timestamps = window.timestamps - T0

        A = np.hstack([np.tile(np.eye(3), (len(timestamps), 1)), np.vstack(np.eye(3)[np.newaxis]*timestamps[...,np.newaxis, np.newaxis])])
        b = np.vstack(window.points3D) - np.vstack(np.eye(3, 1, k=-2)[np.newaxis]*g*timestamps[...,np.newaxis, np.newaxis]**2/2)
        try:
            initial_guess = (np.linalg.inv(A.T@A)@A.T@b).flatten()
        except np.linalg.LinAlgError:
            return None

        model = BallisticModel(initial_guess, T0)
        model.window = window
        return model

@dataclass
class RanSaC(Fitter3D):
    n_ini: int = 3
    n_models: int = 200
    inlier_threshold: float = 31 # cm between model and detection to be considered inlier
    min_inliers: int = 4
    distance_threshold: float = 100 # minimum ms betwen two initial random samples
    alpha: int = 2 # denominator exponent in mean square error computation
    #tau_cost_ransac: float = 0.5 # TODO
    display: bool = False
    def __call__(self, window):
        best_model_error = np.inf
        model = None

        indices = np.array(window.indices)
        timestamps = window.timestamps
        eye = np.eye(len(timestamps), dtype=np.bool)
        for _ in range(self.n_models):
            # initial samples: 3 detections separated by at least 100ms
            samples_mask = np.array(eye[np.random.randint(len(timestamps))])
            while samples_mask.sum() < self.n_ini:
                mask = np.all(np.abs(timestamps - timestamps[samples_mask, None]) >= self.distance_threshold, axis=0)
                samples_mask += random.choice(eye[mask])

            # fit a model to the sample
            sample_model = super().__call__(Window([window[i] for i in indices[samples_mask]], window.popped + np.where(samples_mask)[0][0]))
            inliers_mask = np.linalg.norm(sample_model(timestamps) - window.points3D, axis=0) < self.inlier_threshold
            if inliers_mask.sum() < self.min_inliers or inliers_mask.sum() < (model_length := np.where(inliers_mask)[0].ptp()) /2:
                continue

            # refine the model on initial inliers
            sample_model = super().__call__(Window([window[i] for i in indices[inliers_mask]], window.popped + np.where(inliers_mask)[0][0]))
            model_error = np.linalg.norm(sample_model(timestamps[inliers_mask]) - window.points3D[:, inliers_mask], axis=0)
            model_error = np.sum(model_error)/len(model_error)**self.alpha

            # update best model
            if model_error < best_model_error:
                model = sample_model
                model_indices = np.arange(np.min(np.where(inliers_mask)), np.max(np.where(inliers_mask))+1)
                model.window = Window([window[i] for i in model_indices], window.popped + model_indices[0])
                best_model_error = model_error

        # TODO:
        # - remove models whose vertical amplitude is flat (assessed based on appropriate threshold)
        # or because their ocupancy rate is lower than 50% (defined by the ratio between the number of 3D candidates
        #                                   close to the detected ballistic trajectory and its duration in timestamps.)
        # zhigh = 350;
        # zlow = 120;
        # thres_deltazhigh = 40;      % minimum difference between the highest
        #                             % and lowest ball candidates on a
        #                             % ballistic trajectory if all the candidates
        #                             % are above zhigh
        # thres_deltazlow = 30;       % minimum difference between the highest
        #                             % and lowest ball candidates on a
        #                             % ballistic trajectory if all the candidates
        #                             % are bellow zlow
        # thres_deltaz = 10;          % minimum difference between the highest
        #                             % and lowest ball candidates on a
        #                             % ballistic trajectory
        return model

@dataclass
class Fitter2D(Fitter3D):
    d_error_weight: float = 1
    underground_penalty_factor: float = 100
    optimizer_kwargs: dict = field(default_factory=dict)

    @cached_property
    def error_fct(self):
        return lambda p_error, d_error: np.linalg.norm(p_error) + self.d_error_weight*np.linalg.norm(d_error)

    def __call__(self, window):
        # initial guess computed from MSE solution of the 3D problem
        if (initial_guess := super().__call__(window)) is None:
            return None

        timestamps = window.timestamps
        T0 = timestamps[0]
        position_data = project_3D_to_2D(window.P, window.points3D)
        diameter_data = compute_length2D(window.K, window.RT, window.points3D, BALL_DIAMETER, points2D=position_data)

        def error(initial_condition):
            model = BallisticModel(initial_condition, T0)
            points3D = model(timestamps)

            position_pred = project_3D_to_2D(window.P, points3D)
            position_error = np.linalg.norm(position_data - position_pred, axis=0)

            diameter_pred = compute_length2D(window.K, window.RT, points3D, BALL_DIAMETER, points2D=position_pred)
            diameter_error = np.abs(diameter_data - diameter_pred)

            underground_balls = points3D.z > BALL_DIAMETER/2  # z points downward
            underground_penalty = self.underground_penalty_factor * np.sum(points3D.z[underground_balls])
            return self.error_fct(position_error, diameter_error) + underground_penalty

        result = scipy.optimize.minimize(error, initial_guess.initial_condition, **{**{'tol': 1}, **self.optimizer_kwargs})
        if not result.success:
            return None

        model = BallisticModel(result['x'], T0)
        model.window = window
        return model



repr_dict = {
    (True, True): 'ʘ',
    (True, False): '·',
    (False, True): 'O',
    (False, False): ' ',
}


@dataclass
class FilteredFitter2D(Fitter2D):
    max_outliers_ratio: float = 0.4
    min_inliers: int = 2
    first_inlier: int = 2
    position_error_threshold: float = 2
    scale: float = 0.1
    display: bool = False

    def compute_inliers(self, window, model):
        inliers_condition = lambda position_error: position_error < self.position_error_threshold * (1 + self.scale*np.sin(np.linspace(0, np.pi, len(position_error))))
        position_pred = project_3D_to_2D(window.P, model(window.timestamps))
        position_data = project_3D_to_2D(window.P, window.points3D)
        position_error = np.linalg.norm(position_data - position_pred, axis=0)

        inliers = np.array([False]*len(window))
        inliers[window.indices] = inliers_condition(position_error)
        return inliers

    def __call__(self, window):
        if (model := super().__call__(window)) is None:
            return None

        inliers_mask = self.compute_inliers(window, model)
        self.display and print("  "*window.popped + "|" + " ".join([repr_dict[s.ball_state == BallState.FLYING, inliers_mask[i]] for i, s in enumerate(window)]), end="|\t")

        if sum(inliers_mask) < self.min_inliers:
            model.message = "too few inliers"
            self.display and print(model.message, flush=True)
            return None

        if sum(inliers_mask[0:self.first_inlier]) == 0:
            model.message = f"{self.first_inlier} first samples are not inliers"
            self.display and print(model.message, flush=True)
            return None

        model_indices = np.arange(np.min(np.where(inliers_mask)), np.max(np.where(inliers_mask))+1)
        model.window = Window([window[i] for i in model_indices], popped=window.popped + model_indices[0])

        outliers_ratio = 1 - sum(inliers_mask)/len(model_indices)
        if outliers_ratio > self.max_outliers_ratio:
            model.message = "too many outliers"
            self.display and print(model.message, flush=True)
            return None

        model.message = "proposed"
        self.display and print(model.message, flush=True)
        return model

@dataclass
class UseStateFilteredFitter2D(FilteredFitter2D):
    min_flyings: int = 1
    max_nonflyings_ratio: float = 0.8
    display: bool = False
    def __call__(self, window):
        flyings_mask = np.array([hasattr(s, 'ball') and s.ball.state == BallState.FLYING for s in window])
        self.display and print("  "*window.popped + "|" + " ".join([repr_dict[s.ball_state == BallState.FLYING, flyings_mask[i]] for i, s in enumerate(window)]), end="|\t")

        if sum(flyings_mask) < self.min_flyings:
            message = "too few flying samples"
            self.display and print(message, flush=True)
            return None

        nonflyings_ratio = 1 - sum(flyings_mask)/(np.ptp(np.where(flyings_mask)) + 1) if np.any(flyings_mask) else 0
        if nonflyings_ratio > self.max_nonflyings_ratio:
            message = "too many non-flying samples"
            self.display and print(message, flush=True)
            return None

        message = "accepted by flying standards"
        self.display and print(message, flush=True)

        return super().__call__(window)


class SlidingWindow(list):
    def __init__(self, gen, min_duration):
        self.gen = gen
        self.min_duration = min_duration
        self.popped = 0
    def pop(self):
        self.popped += 1
        return super().pop(0)
    def enqueue(self, n=1):
        for _ in range(n):
            self.append(next(self.gen))
        while self.duration < self.min_duration:
            self.append(next(self.gen))
    def dequeue(self, n=1):
        for _ in range(n):
            yield self.pop()
        while self.duration > self.min_duration:
            yield self.pop()
    @property
    def duration(self):
        detection_timestamps = [item.timestamp for item in self if hasattr(item, "ball")]
        if len(detection_timestamps) < 2:
            return 0
        return detection_timestamps[-1] - detection_timestamps[0]



RECOVERED_BALL_ORIGIN = 'fitting'


class TrajectoryDetector:
    """ Detect ballistic motion by sliding a window over successive frames and
        fitting a ballistic model to the detections.
        Arguments:
            fitter (Fitter): Fits a model to a window of detections.
            min_window_length (int): minimum window length (in miliseconds).
    """
    def __init__(self, fitter_types, min_window_length, min_distance_px, min_distance_cm,
                 **fitter_kwargs):
        self.min_window_length = min_window_length
        self.min_distance_px = min_distance_px
        self.min_distance_cm = min_distance_cm
        self.fitter = type("Fitter", fitter_types, {})(**fitter_kwargs)
        self.display = True
    """
        Inputs: generator of `Sample`s (containing ball detections or not)
        Outputs: generator of `Sample`s (with added balls where a model was found)
    """
    def __call__(self, gen):
        sliding_window = SlidingWindow(gen, self.min_window_length)
        while True:
            try:
                # Initialize sliding_window
                sliding_window.enqueue()

                # Move sliding_window forward until a model is found
                while not (model := self.fitter(Window(sliding_window, popped=sliding_window.popped))):
                    yield from sliding_window.dequeue()
                    sliding_window.enqueue()

                # Grow sliding_window while model fits data
                while True:
                    sliding_window.enqueue()
                    if not (new_model := self.fitter(Window(sliding_window, popped=sliding_window.popped))):
                        break
                    if len(new_model.window) >= len(model.window):
                        model = new_model

                self.display and print("  "*model.window.popped + "|" + " ".join([repr_dict[s.ball_state == BallState.FLYING, True] for i, s in enumerate(model.window)]), end="|\t")

                # discard model if it is too short
                curve = model(model.window.timestamps)
                distances_cm = np.linalg.norm(curve[:, 1:] - curve[:, :-1], axis=0)

                dist = lambda calib, i: np.linalg.norm(calib.project_3D_to_2D(curve[:, i:i+1]) - calib.project_3D_to_2D(curve[:, i-1:i]))
                distances_px = [dist(model.window[i].calib, i) for i in model.window.indices]

                if sum(distances_cm) >= self.min_distance_cm and sum(distances_px) >= self.min_distance_px:
                    self.display and print("validated")
                    # Set model
                    for sample in model.window:
                        if hasattr(sample, 'ball'):
                            camera_idx = [s.ball.camera for s in model.window if hasattr(s, 'ball')][0]
                        else:
                            sample.timestamp = sample.timestamps[camera_idx]
                            sample.ball = Ball({
                                'origin': RECOVERED_BALL_ORIGIN,
                                'center': model(sample.timestamp).tolist(),
                                'image': camera_idx,
                                'state': BallState.FLYING,
                            })
                        sample.ball.model = model
                else:
                    self.display and print("skipped because of distance:", sum(distances_cm), sum(distances_px))

                # dump samples
                yield from sliding_window.dequeue(len(model.window))

            except StopIteration:
                break

        # empty window once generator is fully consumed
        yield from sliding_window


