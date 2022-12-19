from abc import ABC, abstractmethod
from dataclasses import dataclass

from calib3d import Point3D, Point2D
import numpy as np
import scipy.optimize

from deepsport_utilities.court import BALL_DIAMETER

g = 9.81 * 100 /(1000*1000) # m/s² => cm/ms²

class BallisticModel():
    def __init__(self, x0, y0, z0, vx0, vy0, vz0, T0):
        self.p0 = Point3D(x0, y0, z0)
        self.v0 = Point3D(vx0, vy0, vz0)
        self.a0 = Point3D(0, 0, g)
        self.T0 = T0
    def __call__(self, t):
        t = t - self.T0
        return self.p0 + self.v0*t + self.a0*t**2/2

class Optimizer(ABC):
    @abstractmethod
    def fit(self, samples, T0):
        raise NotImplementedError()
    @abstractmethod
    def inliers(self, samples):
        raise NotImplementedError()
    @abstractmethod
    def error(self, samples):
        raise NotImplementedError()

@dataclass
class MSE3DOptimizer(Optimizer):
    threshold: float
    def fit(self, samples, T0):
        # For linear system Ax = b, least square error x = (inverse(A'A))(A'b)
        A = np.vstack([np.hstack([np.eye(3), np.eye(3)*(sample.timestamp - T0)]) for sample in samples])
        b = np.vstack([sample.ball.center - Point3D(0, 0, g*(sample.timestamp - T0)**2/2) for sample in samples])
        x = (np.linalg.inv(A.T@A)@A.T@b).flatten()
        self.model = BallisticModel(*x, T0)
    def inliers(self, samples):
        error = self.error(samples)
        return np.where(error < self.threshold)[0]
    def error(self, samples):
        return np.sqrt(np.sum(np.array([self.model(sample.timestamp) - sample.ball.center for sample in samples])**2, axis=1))

@dataclass
class MSE2DOptimizer(Optimizer):
    threshold: float
    def fit(self, samples, T0):
        def error(params):
            self.model = BallisticModel(*params, T0)
            return np.mean(self.error(samples))
        p0 = samples[0].ball.center
        v0 = samples[1].ball.center - samples[0].ball.center
        initial_guess = p0.x, p0.y, p0.z, v0.x, v0.y, v0.z
        result = scipy.optimize.minimize(error, initial_guess)
        self.model = BallisticModel(*result['x'], T0)
    def inliers(self, samples):
        error = self.error(samples)
        return np.where(error < self.threshold)[0]
    def error(self, samples):
        p_data = Point2D([s.calib.project_3D_to_2D(s.ball.center)           for s in samples])
        p_pred = Point2D([s.calib.project_3D_to_2D(self.model(s.timestamp)) for s in samples])
        return np.linalg.norm(p_data - p_pred, axis=0)


@dataclass
class MSE2DwithDiameterOptimizer(MSE2DOptimizer):
    alpha: float = 0.5
    def __post_init__(self):
        assert 0.0 <= self.alpha <= 1.0
    def position_error(self, samples):
        return super().error(samples)
    def diameter_error(self, samples):
        d_data = np.array([s.calib.compute_length2D(BALL_DIAMETER, s.ball.center)[0]           for s in samples])
        d_pred = np.array([s.calib.compute_length2D(BALL_DIAMETER, self.model(s.timestamp))[0] for s in samples])
        return np.linalg.norm(d_data - d_pred, axis=0)
    def error(self, samples):
        return self.alpha*self.position_error(samples) + (1-self.alpha)*self.diameter_error(samples)


@dataclass
class IterativeOptimizer(Optimizer):
    optimizer: Optimizer
    min_inliers: int = 5
    def fit(self, samples, T0):
        self.optimizer.fit(samples, T0)
        inliers = self.inliers(samples)
        if len(inliers) < self.min_inliers:
            self.model = None
            return
        if len(inliers) == len(samples):
            return
        samples = [s for i, s in enumerate(samples) if i in inliers]
        self.fit(samples, T0)
    def error(self, samples):
        return self.optimizer.error(samples)
    def inliers(self, samples):
        return self.optimizer.inliers(samples)
