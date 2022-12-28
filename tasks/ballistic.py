from abc import ABC, abstractmethod
from dataclasses import dataclass

from calib3d import Point3D, Point2D
import numpy as np
import scipy.optimize

from deepsport_utilities.court import BALL_DIAMETER
from dataset_utilities.ds.raw_sequences_dataset import BallState


g = 9.81 * 100 /(1000*1000) # m/s² => cm/ms²
np.set_printoptions(precision=3, linewidth=110)#, suppress=True)

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
        x0, y0, z0, vx0, vy0, vz0 = initial_condition
        self.p0 = Point3D(x0, y0, z0)
        self.v0 = Point3D(vx0, vy0, vz0)
        self.a0 = Point3D(0, 0, g)
        self.T0 = T0

    def __call__(self, t):
        t = t - self.T0
        return self.p0 + self.v0*t + self.a0*t**2/2

    def error(self, samples, p):
        P  = np.stack([s.calib.P for s in samples])
        RT = np.stack([np.hstack([s.calib.R, s.calib.T]) for s in samples])
        K  = np.stack([s.calib.K for s in samples])
        timestamps = np.array([s.timestamp for s in samples])

        points3D = Point3D([s.ball.center for s in samples])
        p_data = project_3D_to_2D(P, points3D)
        d_data = compute_length2D(K, RT, points3D, BALL_DIAMETER, points2D=p_data)

        points3D = self(timestamps)
        p_pred = project_3D_to_2D(P, points3D)
        d_pred = compute_length2D(K, RT, points3D, BALL_DIAMETER, points2D=p_pred)

        p_error = np.linalg.norm(p_data - p_pred, axis=0)
        d_error = d_data - d_pred
        error = p(p_error, d_error)
        assert error.shape == (len(samples),), f"error shape ({error.shape}) is not the same as samples ({len(samples)})"
        return error

class Fitter:
    def __init__(self, p, v_scale=1, p_scale=1, **kwargs):
        self.p = p
        self.v_scale = v_scale
        self.p_scale = p_scale
        self.kwargs = kwargs

    def __call__(self, samples):
        T0 = samples[0].timestamp
        P  = np.stack([s.calib.P for s in samples])
        RT = np.stack([np.hstack([s.calib.R, s.calib.T]) for s in samples])
        K  = np.stack([s.calib.K for s in samples])
        timestamps = np.array([s.timestamp for s in samples])
        points3D = Point3D([s.ball.center for s in samples])
        p_data = project_3D_to_2D(P, points3D)
        d_data = compute_length2D(K, RT, points3D, BALL_DIAMETER, points2D=p_data)
        p_error = None
        d_error = None
        p0 = samples[0].ball.center

        def error(initial_condition):
            p0 = Point3D(initial_condition[0:3])/self.p_scale
            v0 = Point3D(initial_condition[3:6])/self.v_scale
            initial_condition = p0.x, p0.y, p0.z, v0.x, v0.y, v0.z
            nonlocal p_error, d_error
            model = BallisticModel(initial_condition, T0)
            points3D = model(timestamps)

            p_pred = project_3D_to_2D(P, points3D)
            p_error = np.linalg.norm(p_data - p_pred, axis=0)

            d_pred = compute_length2D(K, RT, points3D, BALL_DIAMETER, points2D=p_pred)
            d_error = d_data - d_pred

            return self.p(p_error, d_error)
            return p_error + np.linalg.norm(d_error, axis=0) # to delete
            return (np.stack([p_error, d_error])*weights).flatten()

        p0 = samples[0].ball.center
        v0 = (samples[-1].ball.center - samples[0].ball.center)/(samples[-1].timestamp - samples[0].timestamp)
        p0 = p0*self.p_scale
        v0 = v0*self.v_scale
        v0.z = (samples[1].ball.center.z - samples[0].ball.center.z)/(samples[1].timestamp - samples[0].timestamp)
        initial_guess = p0.x, p0.y, p0.z, v0.x, v0.y, v0.z
        result = scipy.optimize.least_squares(error, initial_guess, **self.kwargs)
        if not result.success:
            return None
        return BallisticModel(result['x'], T0)


class ModelSetter:
    def __init__(self, model, inliers):
        self.model = model
        self.inliers = inliers
    def __call__(self, index, sample):
        if self.model and self.inliers[0] <= index <= self.inliers[-1]:
            sample.ball.model = self.model


repr_map = { # true, perd
    (True, True): 'ʘ',
    (True, False): '·',
    (False, True): 'O',
    (False, False): ' ',
}

class SlidingWindow:
    def __init__(self, length, threshold, fitting_error_fct, acceptance_error_fct, min_inliers_ratio=.8, min_inliers=5, display=False, **kwargs):
        self.length = length
        self.window = []
        self.acceptance_error_fct = acceptance_error_fct
        self.fitter = Fitter(p=fitting_error_fct, **kwargs)
        self.threshold = threshold
        self.min_inliers_ratio = min_inliers_ratio    # min inliers ratio between first and last inliers
        self.min_inliers = min_inliers                # min inliers to consider the model valid
        self.popped = []
        self.stop = False
        self.display = display
        if self.display:
            for i, label in enumerate(["normal", "no model", "not enough inliers", "too many outliers", "proposed model", "fewer inliers than previous model"]):
                print(f"\x1b[3{i}m{label}\x1b[0m")

    def add(self, item):
        self.window.append(item)

    def pop(self, count=1, callback=None):
        for i in range(count):
            sample = self.window.pop(0)
            if callback is not None:
                callback(i, sample)
            yield sample
            self.popped.append((sample.ball.state is BallState.FLYING, hasattr(sample.ball, 'model')))

    def print(self, inliers=[], color=0):
        if self.display:
            print(
                f"\x1b[3{color}m" + \
                "".join([" " + repr_map[k[0], False] for k in self.popped]) + \
                "|" + \
                " ".join([repr_map[s.ball.state == BallState.FLYING, i in inliers] for i, s in enumerate(self.window)]) + \
                "|" + \
                "\x1b[0m"
            )

    def fit(self):
        model = self.fitter(self.window)
        if model is None:
            self.print(color=1)
            return None

        error = model.error(self.window, self.acceptance_error_fct)
        inliers = np.where(error < self.threshold)[0]
        if len(inliers) < self.min_inliers:
            self.print(inliers, color=2)
            return None

        inliers_ratio = len(inliers)/(inliers[-1] - inliers[0] + 1)
        if inliers_ratio < self.min_inliers_ratio:
            self.print(inliers, color=3)
            return None

        model.inliers = inliers
        return model


    def __call__(self, gen):
        while True:
            while len(self.window) < self.length:
                self.window.append(next(gen))

            # move window until model is found
            while (model := self.fit()) is None:
                yield from self.pop()
                self.add(next(gen))

            self.print(model.inliers, color=4)

            # grow window while model fits data
            while True:
                self.add(next(gen))
                new_model = self.fit()
                if new_model is None:
                    self.print([], color=2)
                    break
                if len(new_model.inliers) < len(model.inliers):
                    self.print(new_model.inliers, color=5)
                    break
                model = new_model

            self.print(model.inliers)

            # pop model data
            yield from self.pop(model.inliers[-1]+1, callback=ModelSetter(model, model.inliers))

