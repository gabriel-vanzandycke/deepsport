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

    def inliers(self, samples, condition):
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
        inliers = condition(p_error, d_error)
        assert inliers.shape == (len(samples),), f"inliers shape ({inliers.shape}) is not the same as samples ({len(samples)})"
        return inliers

class Fitter:
    def __init__(self, error_fct, optimizer='minimize', **optimizer_kwargs):
        self.optimizer = optimizer
        self.error_fct = error_fct
        self.optimizer_kwargs = optimizer_kwargs

    def __call__(self, samples):
        T0 = samples[0].timestamp
        P  = np.stack([s.calib.P for s in samples])
        RT = np.stack([np.hstack([s.calib.R, s.calib.T]) for s in samples])
        K  = np.stack([s.calib.K for s in samples])
        timestamps = np.array([s.timestamp for s in samples])
        points3D = Point3D([s.ball.center for s in samples])
        p_data = project_3D_to_2D(P, points3D)
        d_data = compute_length2D(K, RT, points3D, BALL_DIAMETER, points2D=p_data)

        def error(initial_condition):
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
        #return BallisticModel(initial_guess, T0)
        result = getattr(scipy.optimize, self.optimizer)(error, initial_guess, **self.optimizer_kwargs)
        result['initial_guess'] = initial_guess
        if not result.success:
            return None
        return BallisticModel(result['x'], T0)

repr_map = { # true, perd
    (True, True): 'ʘ',
    (True, False): '·',
    (False, True): 'O',
    (False, False): ' ',
}

class SlidingWindow:
    def __init__(self, condition, min_inliers_ratio=.8, min_inliers=5, display=False, **kwargs):
        self.window = []
        self.condition = condition
        self.fitter = Fitter(**kwargs)
        self.min_inliers_ratio = min_inliers_ratio    # min inliers ratio between first and last inliers
        self.min_inliers = min_inliers                # min inliers to consider the model valid
        self.popped = []
        self.stop = False
        self.display = display
        if self.display:
            for i, label in enumerate([
                "normal",
                "no model",
                "not enough inliers",
                "too many outliers",
                "proposed model",
                "fewer inliers than previous model",
                "first sample is not an inlier"
            ]):
                print(f"\x1b[3{i}m{i} - {label}\x1b[0m")

    def drop(self, count=1, callback=None):
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
                " ".join([repr_map[s.ball.state == BallState.FLYING, inliers[i] if i < len(inliers) else False] for i, s in enumerate(self.window)]) + \
                "|" + \
                "\x1b[0m"
            )

    def fit(self):
        model = self.fitter(self.window)
        if model is None:
            self.print(color=1)
            return None

        inliers = model.inliers(self.window, self.condition)
        if sum(inliers) < self.min_inliers:
            self.print(inliers, color=2)
            return None

        np.max(np.where(model.inliers))+1
        inliers_ratio = sum(inliers)/(np.ptp(np.where(inliers)) + 1)
        if inliers_ratio < self.min_inliers_ratio:
            self.print(inliers, color=3)
            return None

        if not inliers[0]:
            self.print(inliers, color=6)
            return None

        model.inliers = inliers
        return model


    def __call__(self, gen):
        while True:
            while len(self.window) < self.min_inliers:
                self.window.append(next(gen))

            # move window until model is found
            while (model := self.fit()) is None:
                yield from self.drop(1)
                self.window.append(next(gen))

            self.print(model.inliers, color=4)

            # grow window while model fits data
            while True:
                self.window.append(next(gen))
                new_model = self.fit()
                if new_model is None:
                    self.print([], color=1)
                    break
                if len(new_model.inliers) < len(model.inliers):
                    self.print(new_model.inliers, color=5)
                    break
                self.print(new_model.inliers, color=4)
                model = new_model

            self.print(model.inliers)

            # pop model data
            cb = lambda i, s: setattr(s.ball, 'model', model if model.inliers[i] else None)
            yield from self.drop(np.max(np.where(model.inliers))+1, callback=cb)

