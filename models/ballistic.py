import copy
from dataclasses import dataclass, field, make_dataclass
from functools import cached_property
import random
import warnings

from calib3d import Point3D, Point2D
import numpy as np
import scipy.optimize

from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.ds.instants_dataset import BallState, Ball


g = 9.81 * 100 /(1000*1000) # m/s² => cm/ms²

def compute_length2D(K, RT, points3D, length, points2D=None):
    if points2D is None:
        points2D_H = np.einsum('bij,bjk,kb->ib', K, RT, points3D.H)
        points2D = Point2D(points2D_H) * np.sign(points2D_H[2])
    points3D_c = Point3D(np.einsum('bij,jb->ib', RT, points3D.H)) # Point3D expressed in camera coordinates system
    points3D_c.x += length # add the 3D length to one of the componant
    points2D_H = np.einsum('bij,jb->ib', K, points3D_c) # go in the 2D world
    answer = np.linalg.norm(points2D - Point2D(points2D_H) * np.sign(points2D_H[2]), axis=0)
    return answer

def compute_position_error(window, model):
    position_data = window.calib.project_3D_to_2D(window.points3D)
    position_pred = window.calib.project_3D_to_2D(model(window.timestamps))
    return np.linalg.norm(position_data - position_pred, axis=0)

def fill(array):
    """ Fill inplace boolean array with True between first and last True
    """
    where = np.where(array)[0]
    if where.size:
        array[where[0]:where[-1]+1]= True
    return array

class BallisticModel():
    def __init__(self, initial_condition, T0, window=None):
        self.initial_condition = x0, y0, z0, vx0, vy0, vz0 = initial_condition
        self.p0 = Point3D(x0, y0, z0)
        self.v0 = Point3D(vx0, vy0, vz0)
        self.a0 = Point3D(0, 0, g)
        self.T0 = T0
        self.window = window

    def __call__(self, t):
        t = t - self.T0
        return self.p0 + self.v0*t + self.a0*t**2/2


class Window(tuple):
    def __new__(cls, args, popped):
        self = super().__new__(cls, tuple(args))
        self.popped = popped
        self.mask = np.array([s.ball is not None for s in self])
        return self
    @property
    def timestamps(self):
        return np.array([s.timestamp for s in self], dtype=np.float64)
    @property
    def points3D(self):
        return Point3D([s.ball.center if s.ball else Point3D(np.nan, np.nan, np.nan) for s in self])
    @property
    def calib(self):
        _, indices, counts = np.unique([getattr(s.ball, 'camera', np.nan) for s in self], return_index=True, return_counts=True)
        return self[indices[np.argmax(counts)]].calib
    @property
    def RT(self):
        return np.stack([np.hstack([s.calib.R, s.calib.T]) if s.calib else np.ones((3,4))*np.nan for s in self])
    @property
    def K(self):
        return np.stack([s.calib.K if s.calib else np.ones((3,3))*np.nan for s in self])
    #def __str__(self):
        #return " "*self.popped + "|" + " ".join([repr_dict[s.ball_state == BallState.FLYING, inliers_mask[i]] for i, s in enumerate(window)]), end="|\t")
    @property
    def duration(self):
        timestamps = self.timestamps[self.mask]
        return timestamps[-1] - timestamps[0]
    @property
    def T0(self):
        return self.timestamps[self.mask][0]


class Linear3DSolution:
    def __call__(self, window):
        warnings.warn("Linear3DSolution is not able to produce a good initial condition with strong outliers")
        timestamps = window.timestamps
        if not timestamps.size:
            return None
        T0 = window.T0
        timestamps = window.timestamps - T0

        A = np.hstack([np.tile(np.eye(3), (len(timestamps), 1)), np.vstack(np.eye(3)[np.newaxis]*timestamps[...,np.newaxis, np.newaxis])])
        b = np.vstack(window.points3D) - np.vstack(np.eye(3, 1, k=-2)[np.newaxis]*g*timestamps[...,np.newaxis, np.newaxis]**2/2)
        try:
            initial_guess = (np.linalg.inv(A.T@A)@A.T@b).flatten()
        except np.linalg.LinAlgError:
            return None
        return BallisticModel(initial_guess, T0, window)

@dataclass
class FirstPositionNullSpeedSolution:
    def __call__(self, window):
        T0 = window.T0
        initial_condition = *Point3D(window.points3D[:,0]).to_int_tuple(), 0, 0, 0
        return BallisticModel(initial_condition, T0)

class BoundedFitter:
    max_speed: int = 13 # m/s
    max_margin: int = 200 # cm
    max_height: int = 600 # cm
    @cached_property
    def bounds(self):
        max_speed = self.max_speed*100/1000 # cm/ms
        return (
            (   0-self.max_margin,    0-self.max_margin, -self.max_height, -max_speed, -max_speed, -max_speed),
            (2800+self.max_margin, 1500+self.max_margin, -BALL_DIAMETER/2,  max_speed,  max_speed,  max_speed)
        )

@dataclass
class PositionFitter2D(BoundedFitter):
    display: bool = False
    gtol: float = 10
    ftol: float = 10
    xtol: float = 10
    def __call__(self, window):
        # initial guess
        if (model := super().__call__(window)) is None:
            return None

        T0 = window.T0
        timestamps = window.timestamps
        position_data = window.calib.project_3D_to_2D(window.points3D)

        def error(initial_condition):
            model = BallisticModel(initial_condition, T0)
            position_pred = window.calib.project_3D_to_2D(model(timestamps))
            return np.linalg.norm(position_data - position_pred, axis=0)

        try:
            result = scipy.optimize.least_squares(error, model.initial_condition, loss='cauchy',
                bounds=self.bounds, method='dogbox', gtol=self.gtol, xtol=self.xtol, ftol=self.ftol)
            if not result.success:
                raise ValueError(result.message)
        except ValueError:
            self.display and print("  "*window.popped, self.__class__.__name__, "found no model")
            return None

        return BallisticModel(result.x, T0, window=window)

@dataclass
class FilterInliers:
    distance_threshold: float = 20
    display: bool = False

    def __call__(self, window):
        if (model := super().__call__(window)) is None:
            return None

        position_error = compute_position_error(window, model)
        inliers_mask = position_error < self.distance_threshold
        self.display and print("  "*window.popped + "|" + " ".join([repr_dict[s.ball_state == BallState.FLYING, inliers_mask[i]] for i, s in enumerate(window)]), end="|\t")
        with np.printoptions(precision=0, suppress=True, nanstr='-'):
            message = "Too many, returning No model." if np.sum(inliers_mask) < 3 else ""
            self.display and print(f"initial 2D inliers (POSITION-ONLY model fitting error: {position_error}). {message}", flush=True)

        if np.sum(inliers_mask) < 3:
            return None

        window.mask = inliers_mask
        return model

@dataclass
class PositionAndDiameterFitter2D(BoundedFitter):
    d_error_weight: float = 1
    underground_penalty_factor: float = 100
    optimizer_kwargs: dict = field(default_factory=dict)
    tol: float = 1

    @cached_property
    def error_fct(self):
        return lambda p_error, d_error: np.linalg.norm(p_error) + self.d_error_weight*np.linalg.norm(d_error)

    def __call__(self, window):
        if (initial_guess := super().__call__(window)) is None:
            return None

        timestamps = window.timestamps
        T0 = window.T0
        initial_guess = BallisticModel(np.array([*Point3D(window.points3D[:,0]).to_int_tuple(), 0, 0, 0]), T0)
        position_data = window.calib.project_3D_to_2D(window.points3D)
        diameter_data = compute_length2D(window.K, window.RT, window.points3D, BALL_DIAMETER, points2D=position_data)

        def error(initial_condition):
            model = BallisticModel(initial_condition, T0)
            points3D = model(timestamps)

            position_pred = window.calib.project_3D_to_2D(points3D)
            position_error = np.linalg.norm(position_data - position_pred, axis=0)

            diameter_pred = compute_length2D(window.K, window.RT, points3D, BALL_DIAMETER, points2D=position_pred)
            diameter_error = np.abs(diameter_data - diameter_pred)

            underground_balls = points3D.z > BALL_DIAMETER/2  # z points downward
            underground_penalty = self.underground_penalty_factor * np.sum(points3D.z[underground_balls])

            position_error = position_error[~np.isnan(position_error)]
            diameter_error = diameter_error[~np.isnan(diameter_error)]

            return self.error_fct(position_error, diameter_error) + underground_penalty

        bounds = list(zip(*self.bounds))
        result = scipy.optimize.minimize(error, initial_guess.initial_condition, bounds=bounds, **{**{'tol': self.tol}, **self.optimizer_kwargs})
        if not result.success:
            self.display and print("  "*window.popped, self.__class__.__name__, "found no model", result.message, result.x)
            return None

        return BallisticModel(result.x, T0, window=window)



repr_dict = {
    (True, True): 'ʘ',
    (True, False): '·',
    (False, True): 'O',
    (False, False): ' ',
}



@dataclass
class InliersFiltersSolution:
    max_outliers_ratio: float = 0.4
    min_inliers: int = 2
    first_inlier: int = 1
    position_error_threshold: float = 2
    display: bool = False

    def __call__(self, window):
        if (model := super().__call__(window)) is None:
            return None

        position_error = compute_position_error(window, model)
        inliers_mask = position_error < self.position_error_threshold
        model.window.mask = fill(np.array(inliers_mask))
        self.display and print("  "*window.popped + "|" + " ".join([repr_dict[s.ball_state == BallState.FLYING, inliers_mask[i]] for i, s in enumerate(window)]), end="|\t")

        if sum(inliers_mask) < self.min_inliers:
            with np.printoptions(precision=0, suppress=True, nanstr='-'):
                model.message = f"too few inliers (POSITION+DIAMETER model fitting error: {position_error})"
            self.display and print(model.message, flush=True)
            model.window[0].models.append(model)
            return None

        if sum(inliers_mask[0:self.first_inlier]) != self.first_inlier:
            with np.printoptions(precision=0, suppress=True, nanstr='-'):
                model.message = f"first {self.first_inlier} samples are not inliers (POSITION+DIAMETER model fitting error: {position_error})"
            self.display and print(model.message, flush=True)
            model.window[0].models.append(model)
            return None

        outliers_ratio = 1 - sum(inliers_mask)/sum(model.window.mask)
        if outliers_ratio > self.max_outliers_ratio:
            model.message = "too many outliers"
            self.display and print(model.message, flush=True)
            model.window[0].models.append(model)
            return None

        with np.printoptions(precision=0, suppress=True, nanstr='-'):
            model.message = f"proposed (POSITION+DIAMETER model fitting error: {position_error})"

        self.display and print(model.message, flush=True)
        return model

@dataclass
class FilterBallStateSolution:
    min_flyings: int = 1
    max_nonflyings_ratio: float = 0.8
    display: bool = False
    def __call__(self, window):
        flyings_mask = np.array([s.ball is not None and s.ball.state == BallState.FLYING for s in window])
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
        while len(self) and self[0].ball is None:
            yield self.pop()
            if len(self) == 0:
                self.enqueue()
    @property
    def duration(self):
        detection_timestamps = [item.timestamp for item in self if item.ball is not None]
        if len(detection_timestamps) < 2:
            return 0
        return detection_timestamps[-1] - detection_timestamps[0]


class TrajectoryDetector:
    """ Detect ballistic motion by sliding a window over successive frames and
        fitting a ballistic model to the detections.
        Arguments:
            fitter (Fitter): Fits a model to a window of detections.
            min_window_length (int): minimum window length (in miliseconds).
    """
    def __init__(self, fitter_types, min_window_length, min_distance_px, min_distance_cm,
                 retries, **fitter_kwargs):
        self.min_window_length = min_window_length
        self.min_distance_px = min_distance_px
        self.min_distance_cm = min_distance_cm
        self.retries = retries
        Fitter = make_dataclass("Fitter", [], bases=fitter_types)
        self.fitter = Fitter(**fitter_kwargs)
        self.display = fitter_kwargs.get('display', False)
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
                retries = self.retries
                while True:
                    sliding_window.enqueue()
                    if not (new_model := self.fitter(Window(sliding_window, popped=sliding_window.popped))):
                        if retries == 0:
                            break
                        retries = retries - 1
                        self.display and print("  "*sliding_window.popped, "retries:", retries)
                        continue
                    if len(new_model.window) > len(model.window):
                        model = new_model
                        retries = self.retries

                self.display and print("  "*model.window.popped + "|" + " ".join([repr_dict[s.ball_state == BallState.FLYING, True] for i, s in enumerate(model.window)]), end="|\t")

                model.window = Window([sample for sample, mask in zip(model.window, model.window.mask) if mask], popped=model.window.popped + np.where(model.window.mask)[0][0])

                # discard model if it is too short
                curve = model(model.window.timestamps)
                distances_cm = np.linalg.norm(curve[:, 1:] - curve[:, :-1], axis=0)

                curve = model.window.calib.project_3D_to_2D(curve)
                distances_px = np.linalg.norm(curve[:, 1:] - curve[:, :-1], axis=0)

                if sum(distances_cm) >= self.min_distance_cm and sum(distances_px) >= self.min_distance_px:
                    self.display and print("validated")
                    for sample in model.window:
                        sample.model = model
                else:
                    self.display and print("skipped because of distance:", sum(distances_cm), sum(distances_px))

                # dump samples
                yield from sliding_window.dequeue(len(model.window))

            except StopIteration:
                break

        # empty window once generator is fully consumed
        yield from sliding_window


