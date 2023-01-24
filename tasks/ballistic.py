from enum import IntEnum

from calib3d import Calib, Point3D, Point2D, ProjectiveDrawer
import cv2
import numpy as np
import scipy.optimize

from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.utils import setdefaultattr
from deepsport_utilities.ds.instants_dataset import InstantsDataset, BallState, Ball

np.set_printoptions(precision=3, linewidth=110)#, suppress=True)

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
        x0, y0, z0, vx0, vy0, vz0 = initial_condition
        self.p0 = Point3D(x0, y0, z0)
        self.v0 = Point3D(vx0, vy0, vz0)
        self.a0 = Point3D(0, 0, g)
        self.T0 = T0

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
        indices = [i for i, s in enumerate(samples) if hasattr(s, 'ball')]
        T0 = samples[indices[0]].ball.timestamp
        P  = np.stack([samples[i].calib.P for i in indices])
        RT = np.stack([np.hstack([samples[i].calib.R, samples[i].calib.T]) for i in indices])
        K  = np.stack([samples[i].calib.K for i in indices])
        timestamps = np.array([samples[i].ball.timestamp for i in indices])
        points3D = Point3D([samples[i].ball.center for i in indices])
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
        model = BallisticModel(result['x'], T0)
        model.inliers = np.array([False]*len(samples))
        model.inliers[indices] = self.inliers_condition(p_error, d_error)

        model.indices = np.arange(np.min(np.where(model.inliers)), np.max(np.where(model.inliers))+1) if np.any(model.inliers) else np.array([])
        return model


class ModelFit(IntEnum):
    NO_MODEL = 1
    ACCEPTED = 2
    DISCARDED = 3
    PROPOSED = 4
    LESS_INLIERS = 5

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
    def __init__(self, min_distance=50, max_outliers_ratio=.8, min_inliers=5,
                 display=False, window_size=None, max_inliers_decrease=.1,
                 **fitter_kwargs):
        self.window = []
        self.fitter = Fitter(**fitter_kwargs)
        self.max_outliers_ratio = max_outliers_ratio
        self.max_inliers_decrease = max_inliers_decrease
        self.min_inliers = min_inliers
        self.min_distance = min_distance
        self.popped = 0
        self.display = display
        self.window_size = window_size or min_inliers
        if self.display:
            for label in ModelFit:
                print(f"\x1b[3{label}m{label} -", label, "\x1b[0m")

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

    def print(self, inliers, color=0, label=None):
        if self.display:
            if sum(inliers)>1 or sum([s.ball_state == BallState.FLYING for s in self.window]) > 1:
                popped = self.popped# % 1000
                print(
                    "  "*popped + f"|\x1b[3{color}m" + \
                    " ".join([repr_map[s.ball_state == BallState.FLYING, inliers[i] if i < len(inliers) else False] for i, s in enumerate(self.window)]) + \
                    "\x1b[0m| " + (label or "")
                )
            else:
                self.popped = 0

    def fit(self):
        raise NotImplementedError

    def __call__(self, gen):
        empty = False
        while not empty:
            try:
                model = None # required if `next(gen)` raises `StopIteration`
                while len(self.window) < self.window_size:
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
                        break

                    if (sum(new_model.inliers) - sum(model.inliers))/len(model.inliers) < -self.max_inliers_decrease:
                        self.print(new_model.inliers, color=ModelFit.LESS_INLIERS)
                        break
                    model = new_model

                self.print([i in model.indices for i in range(len(self.window))], color=ModelFit.ACCEPTED, label='(model)')

            except StopIteration:
                empty = True # empty generator raises `StopIteration`

            # pop model data
            if model: # required if `StopIteration` is raised before `model` is assigned
                # def callback(i, sample):
                #     if i in model.indices:
                #         if not hasattr(sample, 'ball'):
                #             sample.ball = Ball({
                #                 'origin': 'model',
                #                 'center': model(sample.timestamps[0]).tolist(),
                #                 'timestamp' : sample.timestamps[0],
                #                 'image': None
                #             })
                #         sample.ball.model = model
                callback = lambda i, s: setattr(setdefaultattr(s, 'ball', Ball({
                    'origin': 'model',
                    'center': model(s.timestamps[0]).tolist(),
                    'timestamp' : s.timestamps[0],
                    'image': None
                })), 'model', model if i in model.indices else None)
                yield from self.pop(np.max(np.where(model.inliers))+1, callback=callback)

        yield from self.pop(len(self.window))

class NaiveSlidingWindow(SlidingWindow):
    def __init__(self, *args, min_distance_2D=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_distance_2D = min_distance_2D

    def fit(self):
        model = self.fitter(self.window)
        if model is None:
            self.print([False]*len(self.window), color=ModelFit.NO_MODEL)
            return None

        inliers = model.inliers
        if sum(inliers) < self.min_inliers:
            self.print(inliers, color=ModelFit.DISCARDED, label="not enough inliers")
            return None

        if self.outliers_ratio(inliers) > self.max_outliers_ratio:
            self.print(inliers, color=ModelFit.DISCARDED, label="too many outliers")
            return None

        if not inliers[0]:
            self.print(inliers, color=ModelFit.DISCARDED, label="first sample is not an inlier")
            return None

        timestamps = np.array([self.window[i].ball.timestamp for i in model.indices])
        points3D = model(timestamps)
        if points3D.z.max() > 0: # ball is under the ground (z points down)
            self.print(model.inliers, color=ModelFit.DISCARDED, label="z < 0")
            return None

        distances3D = np.linalg.norm(points3D[:, 1:] - points3D[:, :-1], axis=0)
        if distances3D.sum() < self.min_distance:
            self.print(model.inliers, color=ModelFit.DISCARDED, label="3D curve too short")
            return None

        position = lambda sample_index, point_index: self.window[sample_index].calib.project_3D_to_2D(points3D[:, point_index])
        distances2D = [np.linalg.norm(position(sample_index, point_index) - position(sample_index, point_index+1)) for point_index, sample_index in enumerate(model.indices[:-1])]
        if distances2D.sum() < self.min_distance_2D:
            self.print(model.inliers, color=ModelFit.DISCARDED, label="2D curve too short")
            return None

        self.print(model.inliers, color=ModelFit.PROPOSED)
        return model

class BallStateSlidingWindow(SlidingWindow):
    def __init__(self, *args, min_flyings, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_flyings = min_flyings

    def fit(self):
        flyings = [hasattr(s, 'ball') and s.ball.state == BallState.FLYING for s in self.window]
        if sum(flyings) < self.min_flyings:
            self.print(flyings, color=ModelFit.DISCARDED, label="not enough flying balls")
            return None

        if self.outliers_ratio(flyings) > self.max_outliers_ratio:
            self.print(flyings, color=ModelFit.DISCARDED, label="too many outliers")
            return None

        model = self.fitter(self.window)
        if model is None:
            self.print(flyings, color=ModelFit.NO_MODEL)
            return None

        inliers = model.inliers
        if sum(inliers) < self.min_inliers:
            self.print(inliers, color=ModelFit.DISCARDED, label="not enough inliers")
            return None

        self.print(model.inliers, color=ModelFit.PROPOSED, label='(inliers)')
        self.print(flyings, color=ModelFit.PROPOSED, label='(flyings)')
        return model


class Trajectory:
    def __init__(self, samples, trajectory_id):
        self.start_key = samples[0].key
        self.end_key = samples[-1].key
        self.samples = samples
        self.trajectory_id = trajectory_id
    def __lt__(self, other): # self < other
        return self.end_key < other.start_key
    def __gt__(self, other): # self > other
        return self.start_key > other.end_key
    def __eq__(self, other):
        raise NotImplementedError
    def __sub__(self, other):
        return min(self.end_key.timestamp,   other.end_key.timestamp) \
             - max(self.start_key.timestamp, other.start_key.timestamp)

class MatchTrajectories:
    def __init__(self, TP_cb=None, FP_cb=None, FN_cb=None):
        self.TP = []
        self.FP = []
        self.FN = []
        self.dist_T0 = []
        self.dist_TN = []
        self.TP_cb = TP_cb or (lambda a, p: None)
        self.FP_cb = FP_cb or (lambda a, p: None)
        self.FN_cb = FN_cb or (lambda a, p: None)
        self.annotations = []
        self.predictions = []

    def update_metrics(self, a: Trajectory, p: Trajectory):
        self.dist_T0.append(p.start_key.timestamp - a.start_key.timestamp)
        self.dist_TN.append(p.end_key.timestamp - a.end_key.timestamp)

    def extract_annotated_trajectories(self, gen):
        trajectory_id = 1
        samples = []
        for sample in gen:
            if sample.ball_state == BallState.FLYING:
                self.annotations.append(trajectory_id)
                samples.append(sample)
            else:
                self.annotations.append(0)
                if samples:
                    yield Trajectory(samples, trajectory_id)
                    trajectory_id += 1
                samples = []

    def extract_predicted_trajectories(self, gen):
        trajectory_id = 1
        model = None
        samples = []
        for sample in gen:
            if hasattr(sample, 'ball') and (new_model := getattr(sample.ball, 'model', None)) != model:
                if model:
                    yield Trajectory(samples, trajectory_id)
                    trajectory_id += 1
                samples = []
                model = new_model
            if model is not None:
                samples.append(sample)
                self.predictions.append(trajectory_id)
            else:
                self.predictions.append(0)

    def __call__(self, agen, pgen):
        pgen = self.extract_predicted_trajectories(pgen)
        agen = self.extract_annotated_trajectories(agen)
        try:
            p = next(pgen)
            a = next(agen)
            while True:
                while a < p:
                    self.FN.append(a.trajectory_id)
                    self.FN_cb(a, None)
                    a = next(agen)
                while p < a:
                    self.FP.append(p.trajectory_id)
                    self.FP_cb(None, p)
                    p = next(pgen)
                if a.end_key.timestamp in range(p.start_key.timestamp, p.end_key.timestamp+1):
                    while (a2 := next(agen)) - p > a - p:
                        self.FN.append(a.trajectory_id)
                        self.FN_cb(a, p)
                        a = a2
                    self.TP.append((a.trajectory_id, p.trajectory_id))
                    self.TP_cb(a, p)
                    self.update_metrics(a, p)
                    p = next(pgen)
                    a = a2
                elif p.end_key.timestamp in range(a.start_key.timestamp, a.end_key.timestamp+1):
                    while a - (p2 := next(pgen)) > a - p:
                        self.FP.append(p.trajectory_id)
                        self.FP_cb(a, p)
                        p = p2
                    self.TP.append((a.trajectory_id, p.trajectory_id))
                    self.TP_cb(a, p)
                    self.update_metrics(a, p)
                    a = next(agen)
                    p = p2
                else:
                    pass # no match, move-on to next annotated trajectory
        except StopIteration:
            # Consume remaining trajectories
            try:
                while (p := next(pgen)):
                    self.FP.append(p.trajectory_id)
                    self.FP_cb(None, p)
            except StopIteration:
                pass
            try:
                while (a := next(agen)):
                    self.FN.append(a.trajectory_id)
                    self.FN_cb(a, None)
            except StopIteration:
                pass


class TrajectoryRenderer():
    def __init__(self, ids: InstantsDataset, margin: int=0):
        self.ids = ids
        self.margin = margin

    def draw_ball(self, pd, image, ball, color=None, label=None):
        color = color or pd.color
        ground3D = Point3D(ball.center.x, ball.center.y, 0)
        pd.polylines(image, Point3D([ball.center, ground3D]), lineType=cv2.LINE_AA, color=color)
        pd.draw_line(image, ground3D+Point3D(100,0,0), ground3D-Point3D(100,0,0), lineType=cv2.LINE_AA, thickness=1, color=color)
        pd.draw_line(image, ground3D+Point3D(0,100,0), ground3D-Point3D(0,100,0), lineType=cv2.LINE_AA, thickness=1, color=color)
        center = pd.calib.project_3D_to_2D(ball.center).to_int_tuple()
        radius = pd.calib.compute_length2D(ball.center, BALL_DIAMETER/2)

        cv2.circle(image, center, int(radius), color, 1)
        if label is not None:
            cv2.putText(image, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def __call__(self, trajectory: Trajectory):
        for sample in trajectory.samples:
            key = sample.key
            instant = self.ids.query_item(key)

            for image, calib in zip(instant.images, instant.calibs):
                pd = ProjectiveDrawer(calib, (0, 120, 255), segments=1)

                if hasattr(sample, "ball"):
                    ball = sample.ball
                    self.draw_ball(pd, image, ball, label=str(ball.state))

                    if hasattr(ball, "model") and ball.model is not None:
                        pd.color = (250, 195, 0)
                        model = ball.model
                        timestamps = np.array([sample.ball.timestamp for sample in trajectory.samples if hasattr(sample, 'ball') and hasattr(sample.ball, 'timestamp')])
                        points3D = model(timestamps)
                        ground3D = Point3D(points3D.x, points3D.y, np.zeros_like(points3D.x))
                        start = Point3D(np.vstack([points3D[:, 0], ground3D[:, 0]]).T)
                        stop  = Point3D(np.vstack([points3D[:, -1], ground3D[:, -1]]).T)
                        for line in [points3D, ground3D, start, stop]:
                            pd.polylines(image, line, lineType=cv2.LINE_AA)

                # draw ball annotation if any
                if sample.ball_annotations:
                    ball = sample.ball_annotations[0]
                    if ball.center.z < -0.01: # z pointing down
                        color = (0, 255, 20)
                        self.draw_ball(pd, image, ball, color=color)
                cv2.putText(image, str(sample.ball_state), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 20), lineType=cv2.LINE_AA)
            yield np.hstack(instant.images)


    # model = None
    # canvas = []
    # def __init__(self, f=25, thickness=2, display=False):
    #     self.thickness = thickness
    #     self.f = f
    #     self.display = display
    # def __call__(self, images, calibs: Iterable[Calib], sample=None, color=(0, 120, 255)):
    #     new_model = False

    #     # If new model starts or existing model has ended
    #     if hasattr(sample, 'ball') and (model := getattr(sample.ball, 'model', None)) != self.model:
    #         self.model = model
    #         new_model = model is not None
    #         if model is None:
    #             if self.display: # display model that just ended
    #                 pass # TODO
    #                 # w = self.canvas.shape[1]
    #                 # h = int(w/16*9)
    #                 # offset = 320
    #                 # plt.imshow(self.canvas[offset:offset+h, 0:w])
    #                 # plt.show()
    #             self.canvas = []

    #     # Draw model
    #     if self.model and self.model.T0 <= timestamp <= self.model.TN:
    #         num  = int((self.model.TN - self.model.T0)*self.f/1000)
    #         points3D = self.model(np.linspace(self.model.T0, self.model.TN, num))
    #         ground3D = Point3D(points3D)
    #         ground3D.z = 0
    #         start = Point3D(np.vstack([points3D[:, 0], ground3D[:, 0]]).T)
    #         stop  = Point3D(np.vstack([points3D[:, -1], ground3D[:, -1]]).T)
    #         for image, calib in zip(images, calibs):
    #             pd = ProjectiveDrawer(calib, (255,255,0), thickness=self.thickness, segments=1)
    #             pd.polylines(image, points3D, lineType=cv2.LINE_AA)
    #             pd.polylines(image, ground3D, lineType=cv2.LINE_AA)
    #             pd.polylines(image, start, lineType=cv2.LINE_AA)
    #             pd.polylines(image, stop, lineType=cv2.LINE_AA)

    #     # Draw detected position
    #     if hasattr(sample, 'ball'):
    #         ground3D = Point3D(sample.ball.center.x, sample.ball.center.y, 0)
    #         if new_model: # use current image as canvas
    #             self.canvas = copy.deepcopy(images)
    #         for image_list in [images, self.canvas]:
    #             for image, calib in zip(image_list, calibs):
    #                 if hasattr(sample, "ball"):
    #                     ball = sample.ball
    #                     color = (0, 120, 255) if hasattr(ball, "model") and ball.model is not None else (200, 40, 100)
    #                     pd = ProjectiveDrawer(calib, color, thickness=self.thickness, segments=1)
    #                     pd.polylines(image, Point3D([ball.center, ground3D]), lineType=cv2.LINE_AA)
    #                     pd.draw_line(image, ground3D+Point3D(100,0,0), ground3D-Point3D(100,0,0), lineType=cv2.LINE_AA, thickness=self.thickness)
    #                     pd.draw_line(image, ground3D+Point3D(0,100,0), ground3D-Point3D(0,100,0), lineType=cv2.LINE_AA, thickness=self.thickness)
    #                     center = calib.project_3D_to_2D(ball.center).to_int_tuple()
    #                     radius = calib.compute_length2D(ball.center, BALL_DIAMETER/2)
    #                     cv2.circle(image, center, int(radius), color, 1)
    #                     cv2.putText(image, str(ball.state), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    #                 cv2.putText(image, str(sample.ball_state), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 30), self.thickness, lineType=cv2.LINE_AA)



class SelectBall:
    def __init__(self, origin):
        self.origin = origin
    def __call__(self, key, item):
        try:
            item.ball = max([d for d in item.ball_detections if d.origin == self.origin], key=lambda d: d.value)
            item.ball.timestamp = item.timestamps[item.ball.camera]
            item.calib = item.calibs[item.ball.camera]
        except ValueError:
            pass
        return item
