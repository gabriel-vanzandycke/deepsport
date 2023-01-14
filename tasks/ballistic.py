import numpy as np
import scipy.optimize

import cv2
from calib3d import Point3D, Point2D, ProjectiveDrawer

from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.ds.instants_dataset.instants_dataset import BallState

np.set_printoptions(precision=3, linewidth=110)#, suppress=True)

g = 9.81 * 100 /(1000*1000) # m/s² => cm/ms²


repr_map = { # true, perd
    (True, True): 'ʘ',
    (True, False): '·',
    (False, True): 'O',
    (False, False): ' ',
}

class SlidingWindow:
    """ Detect ballistic motion by sliding a window over successive ball
        detections using a two terms error: a position error in the image space
        (in pixels), and a diameter error in the image space (in pixels).
        Arguments:
            min_distance (int): trajectories shorter than `min_distance` (in cm)
                are discarded.
            max_outliers_ratio (float): trajectories with more than
                `max_outliers_ratio` outliers (counted between first and last
                inliers) are discarded.
            min_inliers (int): trajectories with less than `min_inliers` inliers
                are discarded.
            display (bool): if `True`, display the trajectory in the terminal.
            fitter_kwargs (dict): keyword arguments passed to model `Fitter`.
    """
    def __init__(self, min_distance=50, max_outliers_ratio=.8, min_inliers=5, display=False, **fitter_kwargs):
        self.window = []
        self.fitter = Fitter(**fitter_kwargs)
        self.max_outliers_ratio = max_outliers_ratio
        self.min_inliers = min_inliers
        self.min_distance = min_distance
        self.popped = 0
        self.display = display
        if self.display:
            for i, label in enumerate([
                "normal",
                "no model",
                "not enough inliers",
                "too many outliers",
                "proposed model",
                "fewer inliers than previous model",
                "first sample is not an inlier",
                "curve length is too short",
                "ball below the ground"
            ]):
                print(f"\x1b[3{i}m{i} - {label}\x1b[0m")

    @staticmethod
    def outliers_ratio(inliers):
        return 1 - sum(inliers)/(np.ptp(np.where(inliers)) + 1)

    def pop(self, count=1, callback=None):
        for i in range(count):
            sample = self.window.pop(0)
            if callback is not None:
                callback(i, sample)
            yield sample
            self.popped += 1

    def print(self, inliers=[], color=0):
        if self.display:
            print(
                "  "*self.popped + f"|\x1b[3{color}m" + \
                " ".join([repr_map[s.ball.state == BallState.FLYING, inliers[i] if i < len(inliers) else False] for i, s in enumerate(self.window)]) + \
                "\x1b[0m|"
            )

    def fit(self):
        raise NotImplementedError

    def __call__(self, gen):
        while True:
            try:
                model = None # required if `next(gen)` raises `StopIteration`
                while len(self.window) < self.min_inliers:
                    self.window.append(next(gen))

                # move window until a model is found
                while (model := self.fit()) is None:
                    yield from self.pop(1)
                    self.window.append(next(gen))

                # grow window while model fits data
                while True:
                    self.window.append(next(gen))
                    new_model = self.fit()
                    if new_model is None:
                        self.print([], color=1)
                        break
                    if sum(new_model.inliers) < sum(model.inliers):
                        self.print(new_model.inliers, color=5)
                        break
                    self.print(new_model.inliers, color=4)
                    model = new_model

                self.print(model.inliers)

            except StopIteration:
                if model:
                    cb = lambda i, s: setattr(s.ball, 'model', model if i in model.indices else None)
                    yield from self.pop(np.max(np.where(model.inliers))+1, callback=cb)
                yield from self.pop(len(self.window))
                return

            # pop model data
            cb = lambda i, s: setattr(s.ball, 'model', model if i in model.indices else None)
            yield from self.pop(np.max(np.where(model.inliers))+1, callback=cb)


class NaiveSlidingWindow(SlidingWindow):
    def fit(self):
        model = self.fitter(self.window)
        if model is None:
            self.print(color=1)
            return None

        inliers = model.inliers
        if sum(inliers) < self.min_inliers:
            self.print(inliers, color=2)
            return None

        if self.outliers_ratio(inliers) > self.max_outliers_ratio:
            self.print(inliers, color=3)
            return None

        if not inliers[0]:
            self.print(inliers, color=6)
            return None

        timestamps = np.array([s.timestamp for s in self.window])[model.indices]
        points3D = model(timestamps)
        if points3D.z.max() > 0: # ball is under the ground (z points down)
            self.print(model.inliers, color=8)
            return None

        distances = np.linalg.norm(points3D[:, 1:] - points3D[:, :-1], axis=0)
        if distances.sum() < self.min_distance:
            self.print(model.inliers, color=7)
            return None

        self.print(inliers, color=4)
        model.TN = self.window[max(np.where(inliers)[0])].timestamp
        return model


class BallStateSlidingWindow(SlidingWindow):
    def fit(self):
        flyings = [s.ball.pred_state == BallState.FLYING for s in self.window]
        if sum(flyings) < self.min_inliers:
            self.print(flyings, color=2)
            return None

        if self.outliers_ratio(flyings) > self.max_outliers_ratio:
            self.print(flyings, color=3)
            return None

        model = self.fitter(self.window)
        if model is None:
            self.print(flyings, color=1)
            return None

        inliers = model.inliers
        # if not inliers[0]:
        #     self.print(flyings, color=6)
        #     return None

        self.print(flyings, color=4)
        model.TN = self.window[max(np.where(inliers)[0])].timestamp
        return model


class BallisticModel():
    def __init__(self, initial_condition, T0, TN=None):
        x0, y0, z0, vx0, vy0, vz0 = initial_condition
        self.p0 = Point3D(x0, y0, z0)
        self.v0 = Point3D(vx0, vy0, vz0)
        self.a0 = Point3D(0, 0, g)
        self.T0 = T0
        self.TN = TN

    def __call__(self, t):
        t = t - self.T0
        return self.p0 + self.v0*t + self.a0*t**2/2


class Fitter:
    """
        Arguments:
            inliers_condition (Callable): given a vector of position errors and
                a vector of diameter errors, returns `True` for indices that
                should be considered inliers.
            error_fct (Callable): given a vector of position errors and a vector
                of diameter errors, returns the scalar error to minimize.
            optimizer (str): optimizer to use, member of `scipy.optimize`.
            optimizer_kwargs (dict): optimizer keyword arguments.
    """
    def __init__(self, inliers_condition, error_fct, optimizer='minimize', **optimizer_kwargs):
        self.optimizer = optimizer
        self.error_fct = error_fct
        self.inliers_condition = inliers_condition
        self.optimizer_kwargs = optimizer_kwargs

    def __call__(self, samples):
        T0 = samples[0].timestamp
        TN = samples[-1].timestamp
        P  = np.stack([s.calib.P for s in samples])
        RT = np.stack([np.hstack([s.calib.R, s.calib.T]) for s in samples])
        K  = np.stack([s.calib.K for s in samples])
        timestamps = np.array([s.timestamp for s in samples])
        points3D = Point3D([s.ball.center for s in samples])
        p_data = project_3D_to_2D(P, points3D)
        d_data = compute_length2D(K, RT, points3D, BALL_DIAMETER, points2D=p_data)

        p_error, d_error = None, None
        def error(initial_condition):
            nonlocal p_error, d_error
            model = BallisticModel(initial_condition, T0)
            points3D = model(timestamps)

            p_pred = project_3D_to_2D(P, points3D)
            p_error = np.linalg.norm(p_data - p_pred, axis=0)

            d_pred = compute_length2D(K, RT, points3D, BALL_DIAMETER, points2D=p_pred)
            d_error = np.abs(d_data - d_pred)

            return self.error_fct(p_error, d_error)

        # initial guess computed from MSE solution of the 3D problem
        A = np.hstack([np.tile(np.eye(3), (len(timestamps), 1)), np.vstack(np.eye(3)[np.newaxis]*(timestamps-T0)[...,np.newaxis, np.newaxis])])
        b = np.vstack(points3D) - np.vstack(np.eye(3, 1, k=-2)[np.newaxis]*g*(timestamps-T0)[...,np.newaxis, np.newaxis]**2/2)
        initial_guess = (np.linalg.inv(A.T@A)@A.T@b).flatten()

        result = getattr(scipy.optimize, self.optimizer)(error, initial_guess, **self.optimizer_kwargs)
        if not result.success:
            return None
        model = BallisticModel(result['x'], T0, TN)
        model.inliers = self.inliers_condition(p_error, d_error)
        model.indices = np.arange(np.min(np.where(model.inliers)), np.max(np.where(model.inliers))+1) if np.any(model.inliers) else np.array([])

        return model


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


class Renderer():
    model = None
    def __init__(self, f=25, thickness=2):
        self.thickness = thickness
        self.f = f
    def __call__(self, timestamp, image, calib, sample=None):
        if sample is not None and (model := getattr(sample.ball, 'model', None)) != self.model:
            self.model = model

        # Draw model
        if self.model and self.model.T0 <= timestamp <= self.model.TN:
            num  = int((self.model.TN - self.model.T0)*self.f/1000)
            points3D = self.model(np.linspace(self.model.T0, self.model.TN, num))
            ground3D = Point3D(points3D)
            ground3D.z = 0
            pd = ProjectiveDrawer(calib, (255,255,0), thickness=self.thickness, segments=1)
            pd.polylines(image, points3D, lineType=cv2.LINE_AA)
            pd.polylines(image, ground3D, lineType=cv2.LINE_AA)
            start = Point3D(np.vstack([points3D[:, 0], ground3D[:, 0]]).T)
            stop  = Point3D(np.vstack([points3D[:, -1], ground3D[:, -1]]).T)
            pd.polylines(image, start, lineType=cv2.LINE_AA)
            pd.polylines(image, stop, lineType=cv2.LINE_AA)

        # Draw detected position
        if sample is not None:
            pd = ProjectiveDrawer(calib, (0,120,255), thickness=self.thickness, segments=1)
            ground3D = Point3D(sample.ball.center.x, sample.ball.center.y, 0)
            pd.polylines(image, Point3D([sample.ball.center, ground3D]), markersize=10, lineType=cv2.LINE_AA)

        return image
