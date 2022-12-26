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

def model_2D(weights=None, v_scale=1, p_scale=1, **kwargs):
    weights = np.vstack(weights)

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

        @classmethod
        def fit(cls, samples):
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
                p0 = Point3D(initial_condition[0:3])/p_scale
                v0 = Point3D(initial_condition[3:6])/v_scale
                initial_condition = p0.x, p0.y, p0.z, v0.x, v0.y, v0.z
                nonlocal p_error, d_error
                model = BallisticModel(initial_condition, T0)
                points3D = model(timestamps)

                p_pred = project_3D_to_2D(P, points3D)
                p_error = np.linalg.norm(p_data - p_pred, axis=0)

                d_pred = compute_length2D(K, RT, points3D, BALL_DIAMETER, points2D=p_pred)
                d_error = d_data - d_pred

                return p_error + np.linalg.norm(d_error, axis=0) # to delete
                return (np.stack([p_error, d_error])*weights).flatten()

            p0 = samples[0].ball.center
            v0 = (samples[-1].ball.center - samples[0].ball.center)/(samples[-1].timestamp - samples[0].timestamp)
            p0 = p0*p_scale
            v0 = v0*v_scale
            v0.z = (samples[1].ball.center.z - samples[0].ball.center.z)/(samples[1].timestamp - samples[0].timestamp)
            initial_guess = p0.x, p0.y, p0.z, v0.x, v0.y, v0.z
            result = scipy.optimize.least_squares(error, initial_guess, **kwargs)
            if not result.success:
                return None
            self = cls(result['x'], T0)
            #self.error = np.hsplit(result['fun'], 2)/weights
            #print(result['fun'])
            self.error = np.stack([p_error, d_error])
            self.samples = samples
            return self

    return BallisticModel


class ModelSetter:
    def __init__(self, model):
        self.model = model
    def __call__(self, index, sample):
        if self.model and self.model.inliers[0] <= index <= self.model.inliers[-1]:
            sample.ball.model = self.model

class SlidingWindow:
    def __init__(self, length, step, threshold, min_inliers_ratio=.8, min_inliers=5, weights=None, **kwargs):
        self.length = length
        self.step = step
        self.window = []
        self.Model = model_2D(weights=weights, **kwargs)
        self.threshold = threshold#np.array([[10], [10]]) # position error, diameter error
        self.min_inliers_ratio = min_inliers_ratio # min inliers ratio between first and last inliers
        self.min_inliers = min_inliers  # min inliers to consider the model valid
        self.popped = 0

    def pop(self, count, callback=None):
        for i in range(count):
            sample = self.window.pop(0)
            if callback is not None:
                callback(i, sample)
            yield sample
            self.popped += 1

    def __call__(self, gen):

        for _ in range(self.length - self.step):
            self.window.append(next(gen))

        model = None
        while True:
            if self.popped > 110:
                self.popped = 0
            #while len(self.window) < self.length:
            for _ in range(self.step):
                self.window.append(next(gen))

            new_model = self.Model.fit(self.window)
            print(" "*self.popped + "".join(["*" if s.ball.state == BallState.FLYING else "." for s in self.window]))
            if new_model is None:
                print("")
                yield from self.pop(len(self.window) - self.length + self.step, callback=ModelSetter(model))
                model = None
                continue

            inliers = np.where(new_model.error[0] < self.threshold)[0]
            print(" "*self.popped + "".join(["-" if i in inliers else " " for i in range(len(self.window))]))
            inliers_ratio = len(inliers)/(inliers[-1] - inliers[0] + 1)
            if len(inliers) < self.min_inliers or inliers_ratio < self.min_inliers_ratio:
                yield from self.pop(len(self.window) - self.length + self.step, callback=ModelSetter(model))
                model = None
                continue

            model = new_model
            model.inliers = inliers
            if len(self.window) - inliers[-1] > self.length:
                yield from self.pop(inliers[-1]+1, callback=ModelSetter(model))
                continue
