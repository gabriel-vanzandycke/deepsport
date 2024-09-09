from dataset_utilities.ds.raw_sequences_dataset.ballistic_trajectories import ParabolaModel, \
    Parabola2DLinearFitter, Parabola3DLinearFitter
# kept for retro-compatibility

from dataclasses import dataclass, field, make_dataclass
from functools import cached_property

from calib3d import Point3D, Point2D
import numpy as np
import scipy.optimize
from tasks.ballsize import compute_projection_error

from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.ds.instants_dataset import BallState


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
    """ Fill boolean array with True between first and last True
    """
    array = np.array(array)
    where = np.where(array)[0]
    if where.size:
        array[where[0]:where[-1]+1]= True
    return array

class MotionModel: # retro-compatibility class
    def __init__(self, initial_condition, T0, window=None, ModelType=ParabolaModel):
        x0, y0, z0, vx0, vy0, vz0 = initial_condition
        self.model = ModelType(Point3D(x0, y0, z0), Point3D(vx0, vy0, vz0), T0)
        self.window = window
    def __call__(self, t):
        return self.model(t)

class Window(tuple):
    def __new__(cls, args, popped=0):
        self = super().__new__(cls, tuple(args))
        self.popped = popped
        return self
    @property
    def mask(self):
        #return np.array([s.ball is not None for s in self])
        project = lambda b: self.calib.project_3D_to_2D(b.center)
        return np.array([s.ball is not None and np.linalg.norm(project(s.ball) - project(s.ball_annotations[0])) < 15 for s in self])
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
        #return " "*self.popped + "|" + " ".join([repr_dict[s.true_state == BallState.FLYING, inliers_mask[i]] for i, s in enumerate(window)]), end="|\t")
    @property
    def duration(self):
        timestamps = self.timestamps[self.mask]
        return timestamps[-1] - timestamps[0]
    @property
    def T0(self):
        return self.timestamps[self.mask][0]

class Linear3DSolution(Parabola3DLinearFitter): # retro-compatibility class
    def __call__(self, window):
        model = super().__call__(None, window.points3D, window.timestamps)
        initial_guess = model.p0.x, model.p0.y, model.p0.z, model.v0.x, model.v0.y, model.v0.z
        return MotionModel(initial_guess, model.t0, window, ModelType=ParabolaModel)

class Linear2DSolution(Parabola2DLinearFitter): # retro-compatibility class
    def __call__(slef, window):
        model = super().__call__(None, window.points3D, window.timestamps)
        initial_guess = model.p0.x, model.p0.y, model.p0.z, model.v0.x, model.v0.y, model.v0.z
        return MotionModel(initial_guess, model.t0, window, ModelType=ParabolaModel)

@dataclass
class NaiveSolution:
    def __call__(self, window):
        mask = ~np.any(np.isnan(window.points3D), axis=0)
        p0 = Point3D(window.points3D[:,mask][:,0])
        p1 = Point3D(window.points3D[:,mask][:,-1])
        T0 = window.timestamps[mask][0]
        T1 = window.timestamps[mask][-1]
        v0 = (p1 - p0)/(T1 - T0) - Point3D(0,0,g*(T1 - T0)/2)
        initial_condition = p0.x, p0.y, p0.z, v0.x, v0.y, v0.z
        return MotionModel(initial_condition, T0)

@dataclass
class BoundedFitter(NaiveSolution):
    max_speed: int = 13 # m/s (later converted to cm/ms: units used in the solver)
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
    ftol: float = .001
    xtol: float = 10
    def __call__(self, window):
        # initial guess
        if (initial_guess := super().__call__(window)) is None:
            return None

        T0 = window.T0
        timestamps = window.timestamps
        position_data = window.calib.project_3D_to_2D(window.points3D)

        def error(initial_condition):
            model = MotionModel(initial_condition, T0)
            position_pred = window.calib.project_3D_to_2D(model(timestamps))
            return np.linalg.norm(position_data - position_pred, axis=0)

        try:
            result = scipy.optimize.least_squares(error, initial_guess.initial_condition, loss='cauchy',
                bounds=self.bounds, method='dogbox', gtol=self.gtol, xtol=self.xtol, ftol=self.ftol)
            if not result.success:
                raise ValueError(result.message)
        except ValueError as e:
            self.display and print("  "*window.popped, self.__class__.__name__, "found no model", e)
            return None
        self.display and print("  "*window.popped, self.__class__.__name__, "found model", result.message)
        return MotionModel(result.x, T0, window=window)

@dataclass
class FilterInliers:
    distance_threshold: float = 20
    display: bool = False
    min_first_model_inliers = 3
    def __call__(self, window):
        if (model := super().__call__(window)) is None:
            return None

        position_error = compute_position_error(window, model)
        inliers_mask = position_error < self.distance_threshold
        self.display and print("  "*window.popped + "|" + " ".join([repr_dict[s.true_state == BallState.FLYING, inliers_mask[i]] for i, s in enumerate(window)]), end="|\t")
        with np.printoptions(precision=0, suppress=True, nanstr='-'):
            message = "Too many, returning No model." if np.sum(inliers_mask) < self.min_first_model_inliers else ""
            message = f"initial 2D inliers (POSITION-ONLY model fitting error: {position_error}). {message}"
            self.display and print(message, flush=True)
            model.message = message
            model.window[0].models.append(model)

        if np.sum(inliers_mask) < self.min_first_model_inliers:
            return None

        window.mask = inliers_mask
        return model

@dataclass
class PositionAndDiameterFitter2D(BoundedFitter):
    d_error_weight: float = 1
    optimizer_kwargs: dict = field(default_factory=dict)
    e_tol: int = -5
    mask: bool = True
    display: bool = False
    @cached_property
    def error_fct(self):
        return lambda p_error, d_error: np.linalg.norm(p_error) + self.d_error_weight*np.linalg.norm(d_error)
        #return lambda p_error, d_error: p_error + self.d_error_weight*d_error

    def __call__(self, window):
        if (initial_guess := super().__call__(window)) is None:
            return None

        T0 = window.T0
        calib = window.calib
        mask = window.mask if self.mask else np.ones((len(window),), dtype=bool)
        timestamps = window.timestamps
        position_data = calib.project_3D_to_2D(window.points3D)
        diameter_data = compute_length2D(window.K, window.RT, window.points3D, BALL_DIAMETER, points2D=position_data)

        position_error = 0
        diameter_error = 0
        di = 0
        #from matplotlib import pyplot as plt

        def error(initial_condition):
            nonlocal position_error
            nonlocal diameter_error
            nonlocal di

            model = MotionModel(initial_condition, T0)
            p = lambda t: model.a0*t**2/2 + model.v0*t + model.p0
            v = lambda t: model.a0*t + model.v0
            a = lambda t: model.a0
            pi3D = p(timestamps-T0)
            vi3D = v(timestamps-T0)
            ai3D = a(timestamps-T0)

            pi2D = calib.project_3D_to_2D(pi3D)
            vi2D = calib.project_3D_to_2D(pi3D + vi3D) - pi2D
            ai2D = calib.project_3D_to_2D(pi3D + ai3D) - pi2D

            oi = position_data
            #plt.axes().set_aspect('equal')

            #plt.plot(pi2D.x, pi2D.y)
            #plt.plot(oi.x, oi.y, marker='.', markersize=10, linestyle='None')

            curve = lambda t: ai2D*t**2/2 + vi2D*t + pi2D   # image space approximations of `model` for all datapoints
            ti = (vi2D.x*(oi.x - pi2D.x) + vi2D.y*(oi.y - pi2D.y))/(vi2D.x**2 + vi2D.y**2) # minimizes the distance between observations and a linear approximation of the image space approximations of `model`
            #plt.plot(curve(ti).x, curve(ti).y, marker='.', markersize=5, linestyle='None')

            #plt.show()
            di = np.linalg.norm(curve(ti) - oi, axis=0)    # distances

            position_error = di
            m = ~np.isnan(position_error) & mask
            position_error = np.sum(position_error[m])

            points3D = model(timestamps)
            position_pred = calib.project_3D_to_2D(points3D)
            #position_error = position_data - position_pred
            #m = ~np.any(np.isnan(position_error), axis=0) & mask
            #position_error = np.linalg.norm(np.sum(position_error[:, m], axis=1)) # norm of mean error vector
            #position_error = np.linalg.norm(position_data - position_pred, axis=0)
            #position_error = position_error[~np.isnan(position_error) & mask]

            diameter_pred = compute_length2D(window.K, window.RT, points3D, BALL_DIAMETER, points2D=position_pred)
            diameter_error = diameter_data - diameter_pred
            m = ~np.isnan(diameter_error) & mask
            diameter_error = np.abs(np.sum(diameter_error[m])) # abs of mean diameter error
            #diameter_error = np.abs(diameter_data - diameter_pred)
            #diameter_error = diameter_error[~np.isnan(diameter_error) & mask]

            return self.error_fct(position_error, diameter_error)

        bounds = list(zip(*self.bounds))

        result = scipy.optimize.minimize(error, initial_guess.initial_condition, bounds=bounds, **{**{'tol': 10**self.e_tol}, **self.optimizer_kwargs})
        if not result.success:
            self.display and print("  "*window.popped, self.__class__.__name__, "found no model", result.message, result.x)
            return None
        self.display and print("di", di)
        self.display and print("p_error:", position_error)
        self.display and print("d_error:", diameter_error)
        return MotionModel(result.x, T0, window=window)

@dataclass
class FitterFromGroundTruth(PositionAndDiameterFitter2D):
    def __call__(self, window):

        ball0 = window[0].ball_annotations[0]
        ball1 = window[-1].ball_annotations[0]
        p0 = ball0.center
        p1 = ball1.center
        T0 = window[0].timestamps[ball0.camera]
        T1 = window[-1].timestamps[ball1.camera]
        g = 9.81 * 100/(1000*1000) # m/s² => cm/ms²
        v0 = (p1 - p0)/(T1 - T0) - Point3D(0,0,g*(T1 - T0)/2)
        model = MotionModel(np.vstack((p0, v0)).flatten(), T0, window=window)

        annotated_points3D = model(window.timestamps)
        predicted_points3D = window.points3D

        p_error = np.linalg.norm(window.calib.project_3D_to_2D(annotated_points3D) - window.calib.project_3D_to_2D(predicted_points3D), axis=0)
        d_error = compute_length2D(window.K, window.RT, predicted_points3D, BALL_DIAMETER) - compute_length2D(window.K, window.RT, annotated_points3D, BALL_DIAMETER)


        model = super().__call__(window)

        with np.printoptions(precision=0, suppress=True, nanstr='-'):
            self.display and print("GT p_error", p_error)
            self.display and print("GT d_error", d_error)
            mask = window.mask if self.mask else np.ones((len(window),), dtype=bool)
            p_error = p_error[~np.isnan(p_error) & mask]
            d_error = d_error[~np.isnan(d_error) & mask]
            self.display and print("mask", mask)
            self.display and print("GT error", self.error_fct(p_error, d_error))

        return model

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
    first_inlier: int = 0
    position_error_threshold: float = 2
    display: bool = False

    def __call__(self, window):
        if (model := super().__call__(window)) is None:
            return None

        position_error = compute_position_error(window, model)
        inliers_mask = position_error < self.position_error_threshold
        self.display and print("  "*window.popped + "|" + " ".join([repr_dict[s.true_state == BallState.FLYING, inliers_mask[i]] for i, s in enumerate(window)]), end="|\t")

        if sum(inliers_mask) < self.min_inliers:
            with np.printoptions(precision=0, suppress=True, nanstr='-'):
                model.message = f"too few inliers (POSITION+DIAMETER model fitting error: {position_error})"
            self.display and print(model.message, flush=True)
            model.window[0].models.append(model)
            return None

        #model.window.mask = fill(inliers_mask)
        # New
        start = np.where(inliers_mask)[0][0]
        stop = np.where(inliers_mask)[0][-1]
        model.window = Window([s for s, m in zip(window, fill(inliers_mask)) if m], popped=window.popped+start)
        model.window.mask = inliers_mask[start:stop+1]
        # weN

        if self.first_inlier and sum(inliers_mask[0:self.first_inlier]) != self.first_inlier:
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
    max_nonflyings_ratio: float = 1.0
    display: bool = False
    def __call__(self, window):
        flyings_mask = np.array([s.ball is not None and s.ball.state == BallState.FLYING for s in window])
        self.display and print("  "*window.popped + "|" + " ".join([repr_dict[s.true_state == BallState.FLYING, flyings_mask[i]] for i, s in enumerate(window)]), end="|\t")

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


class TrajectoryDetectorAndFitter:
    """ Detect ballistic motion by sliding a window over successive frames and
        fitting a ballistic model to the detections.
        Arguments:
            fitter (Fitter): Fits a model to a window of detections.
            min_window_length (int): minimum window length (in miliseconds).
    """
    def __init__(self, fitter_types, min_window_length, min_distance_px, min_distance_cm,
                 retries=0, **fitter_kwargs):
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
                    if not (new_model := self.fitter(Window(sliding_window, popped=sliding_window.popped))) \
                     or sum(new_model.window.mask) <= sum(model.window.mask):
                        if retries == 0:
                            break
                        retries = retries - 1
                        self.display and print("  "*sliding_window.popped, "retries:", retries)
                        continue
                    #if len(new_model.window) > len(model.window):
                    # New
                    if sum(new_model.window.mask) > sum(model.window.mask):
                    # weN
                        model = new_model
                        retries = self.retries

                self.display and print("  "*model.window.popped + "|" + " ".join([repr_dict[s.true_state == BallState.FLYING, True] for i, s in enumerate(model.window)]), end="|\t")

                # removed New
                #model.window = Window([sample for sample, mask in zip(model.window, model.window.mask) if mask], popped=model.window.popped + np.where(model.window.mask)[0][0])

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


class TrajectoryFitter:
    def __init__(self, fitter_types, min_window_length=None,
                 min_distance_px=None, min_distance_cm=None,
                 retries=0, **fitter_kwargs):
        Fitter = make_dataclass("Fitter", [], bases=fitter_types)
        self.fitter = Fitter(**fitter_kwargs)
        self.display = fitter_kwargs.get('display', False)
    def fit_and_yield(self, trajectory):
        if model := self.fitter(Window(trajectory, popped=0)):
            errors = []
            for sample in model.window:
                sample.model = model
                try:
                    error = compute_projection_error(sample.ball_annotations[0].center, sample.model(sample.timestamp))[0]
                except TypeError:
                    error = np.nan
                errors.append(error)
            self.display and print("samples projection error:", f"\n{[f'{e:.0f}' for e in errors]}\n")
        else:
            self.display and print("no model")
        yield from trajectory
    def __call__(self, gen):
        trajectory = []
        ballistic_trajectory_index = None
        for sample in gen:
            if sample.ballistic_trajectory_index != ballistic_trajectory_index:
                if trajectory:
                    yield from self.fit_and_yield(trajectory)
                    trajectory = []
                ballistic_trajectory_index = sample.ballistic_trajectory_index
            if sample.ballistic_trajectory_index is not None:
                trajectory.append(sample)
        yield from self.fit_and_yield(trajectory)


