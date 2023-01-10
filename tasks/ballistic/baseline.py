
import numpy as np

from dataset_utilities.ds.raw_sequences_dataset import BallState

from .common import Fitter

np.set_printoptions(precision=3, linewidth=110)#, suppress=True)

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
            inliers_condition (Callable): given a vector of position errors and
                a vector of diameter errors, returns `True` for indices that
                should be considered inliers.
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
    def __init__(self, inliers_condition, min_distance=50, max_outliers_ratio=.8, min_inliers=5, display=False, **fitter_kwargs):
        self.window = []
        self.fitter = Fitter(inliers_condition=inliers_condition, **fitter_kwargs)
        self.max_outliers_ratio = max_outliers_ratio
        self.min_inliers = min_inliers
        self.min_distance = min_distance
        self.popped = []
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
                "".join([" " + repr_map[k[0], k[1]] for k in self.popped]) + \
                "|" + \
                f"\x1b[3{color}m" + \
                " ".join([repr_map[s.ball.state == BallState.FLYING, inliers[i] if i < len(inliers) else False] for i, s in enumerate(self.window)]) + \
                "\x1b[0m" + \
                "|"
            )

    def fit(self):
        model = self.fitter(self.window)
        if model is None:
            self.print(color=1)
            return None

        inliers = model.inliers
        if sum(inliers) < self.min_inliers:
            self.print(inliers, color=2)
            return None

        outliers_ratio = 1 - sum(inliers)/(np.ptp(np.where(inliers)) + 1)
        if outliers_ratio > self.max_outliers_ratio:
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

        self.print(model.inliers, color=4)
        return model


    def __call__(self, gen):
        while True:
            try:
                model = None
                while len(self.window) < self.min_inliers:
                    self.window.append(next(gen))

                # move window until a model is found
                while (model := self.fit()) is None:
                    yield from self.drop(1)
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
                    yield from self.drop(np.max(np.where(model.inliers))+1, callback=cb)
                yield from self.drop(len(self.window))
                return

            # pop model data
            cb = lambda i, s: setattr(s.ball, 'model', model if i in model.indices else None)
            yield from self.drop(np.max(np.where(model.inliers))+1, callback=cb)
