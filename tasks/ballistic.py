from abc import ABC, abstractmethod
from dataclasses import dataclass

from calib3d import Point3D, Point2D
import numpy as np
import scipy.optimize

from deepsport_utilities.court import BALL_DIAMETER
from dataset_utilities.ds.raw_sequences_dataset import BallState


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

def model_2D(weights=(1, 1), **kwargs):
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

        @staticmethod
        def position_error(P, p_data, points3D):
            p_pred = project_3D_to_2D(P, points3D)
            return np.linalg.norm(p_data - p_pred, axis=0)

        @staticmethod
        def diameter_error(K, RT, d_data, points3D, points2D=None):
            d_pred = compute_length2D(K, RT, points3D, BALL_DIAMETER, points2D=points2D)
            return d_data - d_pred

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

            def error(initial_conditions):
                model = BallisticModel(initial_conditions, T0)
                points3D = model(timestamps)
                p_error = cls.position_error(P, p_data, points3D)
                d_error = cls.diameter_error(K, RT, d_data, points3D, points2D=project_3D_to_2D(P, points3D))
                return (np.stack([p_error, d_error])*weights).flatten()

            p0 = samples[0].ball.center
            v0 = samples[1].ball.center - samples[0].ball.center
            initial_guess = p0.x, p0.y, p0.z, v0.x, v0.y, v0.z
            result = scipy.optimize.least_squares(error, initial_guess, **kwargs)
            if not result.success:
                return None
            self = cls(result['x'], T0)
            self.error = np.hsplit(result['fun'], 2)/weights
            self.samples = samples
            return self

    return BallisticModel


class SlidingWindow:
    def __init__(self, length, step, weights=(1, 1), **kwargs):
        self.length = length
        self.step = step
        self.window = []
        self.Model = model_2D(weights=weights, **kwargs)
        self.threshold = np.array([[10], [10]]) # position error, diameter error
        self.min_inliers_ratio = .8 # min inliers ratio between first and last inliers
        self.min_inliers = 6  # min inliers to consider the model valid
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

        while True:
            #while len(self.window) < self.length:
            for _ in range(self.step):
                self.window.append(next(gen))
            print(" "*self.popped, "".join(["*" if s.ball.state == BallState.FLYING else "." for s in self.window]))

            model = self.Model.fit(self.window)
            if model is None:
                yield from self.pop(len(self.window) - self.length + self.step)
                continue

            inliers = np.where(np.all(model.error < self.threshold, axis=0))[0]
            inliers_ratio = (inliers[-1] - inliers[0] + 1)/len(inliers)
            if len(inliers) < self.min_inliers or inliers_ratio < self.min_inliers_ratio:
                yield from self.pop(len(self.window) - self.length + self.step)
                continue

            inliers = range(inliers[0], inliers[-1] + 1)
            if len(self.window) - inliers[-1] > self.length:
                callback = lambda i, s: setattr(s.ball, "center_new", model(s.timestamp) if i in inliers else None)
                yield from self.pop(inliers[-1], callback)
                continue


def _():
    window = 7

    samples = list(data)
    step = 3
    optimizer = MSE2DwithDiameterOptimizer(threshold=5, alpha=0.5)
    im = IterativeOptimizer(optimizer=optimizer)


    def smooth(samples):
        model = None
        popped = 0

        # initialization
        queue = [samples.pop(0) for _ in range(window - step)]

        while True:
            for _ in range(step):
                queue.append(samples.pop(0))
            if popped > 100:
                popped = 0
            print(" "*popped, "".join(["*" if s.ball.state == BallState.FLYING else "." for s in queue]))

            im.fit(queue, queue[0].timestamp)
            if im.model:
                inliers = im.inliers(queue)
                print(" "*popped, "".join(["o" if i in inliers else "." for i in range(len(queue))]))
                if inliers[-1] < len(queue) - step: # none of the new samples meet the model
                    for i in range(inliers[0], inliers[-1]+1):
                        queue[i].ball.center_old = queue[i].ball.center
                        queue[i].ball.center_new = im.model(queue[i].timestamp)
                    while len(queue) > window:
                        yield queue.pop(0)
                        popped += 1
                    im.model = None
                    model = None
                else:
                    model = im.model
            else:
                if model: # if recorded model, use it instead
                    im.model = model
                    inliers = im.inliers(queue)
                    for i in range(inliers[0], inliers[-1]+1):
                        queue[i].ball.center_old = queue[i].ball.center
                        queue[i].ball.center_new = model(queue[i].timestamp)
                while len(queue) > window:
                    yield queue.pop(0)
                    popped += 1
                im.model = None
                model = None


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
    alpha: float = 0.5
    @staticmethod
    def process_samples(samples):
        P  = np.stack([s.calib.P for s in samples])
        RT = np.stack([np.hstack([s.calib.R, s.calib.T]) for s in samples])
        K  = np.stack([s.calib.K for s in samples])
        timestamps = np.array([s.timestamp for s in samples])
        points3D = Point3D([s.ball.center for s in samples])
        p_data = project_3D_to_2D(P, points3D)
        d_data = compute_length2D(K, RT, points3D, BALL_DIAMETER, p_data)
        return P, K, RT, timestamps, p_data, d_data
    def compute_error(self, P, K, RT, timestamps, p_data, d_data, model):
        p_pred = project_3D_to_2D(P, model(timestamps))
        d_pred = compute_length2D(K, RT, model(timestamps), BALL_DIAMETER, p_pred)
        p_error = np.linalg.norm(p_data - p_pred, axis=0)
        d_error = np.linalg.norm(d_data - d_pred, axis=0)
        return (self.alpha)*p_error + (1-self.alpha)*d_error
    def fit(self, samples, T0):
        P, K, RT, timestamps, p_data, d_data = self.process_samples(samples)
        def error(params):
            model = BallisticModel(*params, T0)
            error = self.compute_error(P, K, RT, timestamps, p_data, d_data, model)
            return error
        p0 = samples[0].ball.center
        v0 = samples[1].ball.center - samples[0].ball.center
        initial_guess = p0.x, p0.y, p0.z, v0.x, v0.y, v0.z
        result = scipy.optimize.least_squares(error, initial_guess)
        print(result)
        self.model = BallisticModel(*result['x'], T0)
        raise
    def inliers(self, samples):
        error = self.error(samples)
        return np.where(error < self.threshold)[0]
    def error(self, samples):
        model = self.model
        P, K, RT, timestamps, p_data, d_data = self.process_samples(samples)
        return self.compute_error(P, K, RT, timestamps, p_data, d_data, model)


class Iterative2DOptimizer(MSE2DOptimizer):
    min_inliers: int = 5
    def fit(self, samples, T0):
        super().fit(samples, T0)
        inliers = self.inliers(samples)
        if len(inliers) < self.min_inliers:
            self.model = None
            return
        if len(inliers) == len(samples):
            return
        samples = [s for i, s in enumerate(samples) if i in inliers]
        self.fit(samples, T0)

def smooth(samples, window, step, im):
    model = None
    popped = 0

    # initialization
    queue = [samples.pop(0) for _ in range(window - step)]

    while True:
        for _ in range(step):
            queue.append(samples.pop(0))
        if popped > 100:
            popped = 0
        print(" "*popped, "".join(["*" if s.ball.state == BallState.FLYING else "." for s in queue]))

        im.fit(queue, queue[0].timestamp)
        if im.model:
            inliers = im.inliers(queue)
            print(" "*popped, "".join(["o" if i in inliers else "." for i in range(len(queue))]))
            if inliers[-1] < len(queue) - step: # none of the new samples meet the model
                for i in range(inliers[0], inliers[-1]+1):
                    queue[i].ball.center_old = queue[i].ball.center
                    queue[i].ball.center_new = im.model(queue[i].timestamp)
                while len(queue) > window:
                    yield queue.pop(0)
                    popped += 1
                im.model = None
                model = None
            else:
                model = im.model
        else:
            if model: # if recorded model, use it instead
                im.model = model
                inliers = im.inliers(queue)
                for i in range(inliers[0], inliers[-1]+1):
                    queue[i].ball.center_old = queue[i].ball.center
                    queue[i].ball.center_new = model(queue[i].timestamp)
            while len(queue) > window:
                yield queue.pop(0)
                popped += 1
            im.model = None
            model = None
