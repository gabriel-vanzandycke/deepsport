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
        self.optimizer_kwargs.setdefault('tol', 1)

    def __call__(self, samples):
        indices = [i for i, s in enumerate(samples) if hasattr(s, 'ball')]
        if not indices:
            return None
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
        try:
            initial_guess = (np.linalg.inv(A.T@A)@A.T@b).flatten()
        except np.linalg.LinAlgError:
            return None

        result = getattr(scipy.optimize, self.optimizer)(error, initial_guess, **self.optimizer_kwargs)
        if not result.success:
            return None
        model = BallisticModel(result['x'], T0)
        model.inliers = np.array([False]*len(samples))
        model.inliers[indices] = self.inliers_condition(p_error, d_error)
        camera = samples[np.argmax(model.inliers)].ball.camera if np.any(model.inliers) else None # camera index of first inlier
        model.cameras = np.array([(camera := samples[i].ball.camera if model.inliers[i] else camera) for i in range(len(samples))])
        model.indices = np.arange(np.min(np.where(model.inliers)), np.max(np.where(model.inliers))+1) if np.any(model.inliers) else np.array([])
        return model


class ModelFit(IntEnum):
    NO_MODEL = 1
    ACCEPTED = 2
    DISCARDED = 3
    PROPOSED = 4
    LESS_INLIERS = 5
    SKIPPED = 6

repr_map = { # true, perd
    (True, True): 'ʘ',
    (True, False): '·',
    (False, True): 'O',
    (False, False): ' ',
}

RECOVERED_BALL_ORIGIN = 'fitting'

class Window(list):
    @property
    def duration(self):
        return self[-1].key.timestamp - self[0].key.timestamp if self else 0

class SlidingWindow:
    """ Detect ballistic motion by sliding a window over successive frames and
        fitting a ballistic model to the detections.
        Arguments:
            min_window_length (int): minimum window length (in miliseconds).
            min_inliers (int): trajectories with less than `min_inliers` inliers
                are discarded.
            max_outliers_ratio (float): trajectories with more than
                `max_outliers_ratio` outliers (counted between first and last
                inliers) are discarded.
            max_inliers_decrease (float): threshold above which, if the ratio
                of inliers decreases, the trajectory is discarded.
            min_distance_cm (int): trajectories shorter than `min_distance_cm`
                (in world coordinates) are discarded.
            min_distance_px (int): trajectories shorter than `min_distance_px`
                (in image space) are discarded.
            display (bool): if `True`, display the trajectory in the terminal.
            fitter_kwargs (dict): keyword arguments passed to model `Fitter`.
    """
    def __init__(self, min_window_length, *, min_inliers, max_outliers_ratio=.4,
            max_inliers_decrease=.1, min_distance_cm=50, min_distance_px=50,
            display=False, **fitter_kwargs):
        self.min_window_length = min_window_length
        self.window = Window()
        self.fitter = Fitter(**fitter_kwargs)
        self.max_outliers_ratio = max_outliers_ratio
        self.max_inliers_decrease = max_inliers_decrease
        self.min_inliers = min_inliers
        self.min_distance_cm = min_distance_cm
        self.min_distance_px = min_distance_px
        self.display = display
        if self.display:
            for label in ModelFit:
                print(f"\x1b[3{label}m{label} -", label, "\x1b[0m")
    popped = 0
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
        model = self.fitter(self.window)
        if model is None:
            self.print([False]*len(self.window), color=ModelFit.NO_MODEL)
            return None

        inliers = model.inliers
        if not inliers[0]:
            self.print(inliers, color=ModelFit.DISCARDED, label="first sample is not an inlier")
            return None

        if sum(inliers) < self.min_inliers:
            self.print(inliers, color=ModelFit.DISCARDED, label="not enough inliers")
            return None

        outliers_ratio = 1 - sum(inliers)/(np.ptp(np.where(inliers)) + 1)
        if outliers_ratio > self.max_outliers_ratio:
            self.print(inliers, color=ModelFit.DISCARDED, label="too many outliers")
            return None

        timestamps = np.array([self.window[i].ball.timestamp for i in model.indices if hasattr(self.window[i], 'ball')])
        points3D = model(timestamps)
        if points3D.z.max() > 0: # ball is under the ground (z points down)
            self.print(model.inliers, color=ModelFit.DISCARDED, label="z < 0")
            return None

        distances_cm = np.linalg.norm(points3D[:, 1:] - points3D[:, :-1], axis=0)
        if distances_cm.sum() < self.min_distance_cm:
            self.print(model.inliers, color=ModelFit.DISCARDED, label="3D curve too short")
            return None

        position = lambda sample_index, point_index: self.window[sample_index].calib.project_3D_to_2D(points3D[:, point_index:point_index+1])
        distances_px = [np.linalg.norm(position(sample_index, point_index) - position(sample_index, point_index+1)) for point_index, sample_index in enumerate(model.indices[:-1]) if hasattr(self.window[sample_index], 'ball')]
        if sum(distances_px) < self.min_distance_px:
            self.print(model.inliers, color=ModelFit.DISCARDED, label="2D curve too short")
            return None

        self.print(model.inliers, color=ModelFit.PROPOSED, label='(inliers)')
        return model

    """
        Inputs: generator of `Sample`s (containing ball detections or not)
        Outputs: generator of `Sample`s (with added balls where a model was found)
    """
    def __call__(self, gen):
        empty = False
        while not empty:
            try:
                model = None # required if `next(gen)` raises `StopIteration`
                while self.window.duration < self.min_window_length:
                    self.window.append(next(gen))

                # move window forward until a model is found
                while (model := self.fit()) is None:
                    yield from self.pop(1)
                    while self.window.duration < self.min_window_length:
                        self.window.append(next(gen))

                # grow window while model fits data
                while True:
                    self.window.append(next(gen))
                    new_model = self.fit()
                    if new_model is None:
                        break

                    inliers_increase = (sum(new_model.inliers) - sum(model.inliers))/len(model.inliers)
                    if inliers_increase >= 0:
                        model = new_model
                    elif inliers_increase < -self.max_inliers_decrease:
                        self.print(new_model.inliers, color=ModelFit.LESS_INLIERS)
                        break
                    else: # inliers decreased but not too much
                        self.print(new_model.inliers, color=ModelFit.SKIPPED)
                        continue # keep previous model and test a new model with additional data

                self.print([i in model.indices for i in range(len(self.window))], color=ModelFit.ACCEPTED, label='(model)')

            except StopIteration:
                empty = True # empty generator raises `StopIteration`

            # pop model data
            if model: # required if `StopIteration` is raised before `model` is assigned
                model.start_timestamp = self.window[model.indices[0]].ball.timestamp
                model.end_timestamp = self.window[model.indices[-1]].ball.timestamp
                callback = lambda i, s: setattr(setdefaultattr(s, 'ball', Ball({
                    'origin': RECOVERED_BALL_ORIGIN,
                    'center': model(s.timestamps[model.cameras[i]]).tolist(),
                    'timestamp' : s.timestamps[model.cameras[i]],
                    'image': model.cameras[i],
                })), 'model', model if i in model.indices else None)
                yield from self.pop(np.max(np.where(model.inliers))+1, callback=callback)

        yield from self.pop(len(self.window))

class BallStateSlidingWindow(SlidingWindow):
    """
        Arguments:
            min_flyings (int): number of flying balls required to fit a model.
            max_nonflyings_ratio (float): maximum ratio of non-flying balls,
                computed between the first and last sample classified as flying.
    """
    def __init__(self, *, min_flyings, max_nonflyings_ratio, **kwargs):
        super().__init__(**kwargs)
        self.min_flyings = min_flyings
        self.max_nonflyings_ratio = max_nonflyings_ratio

    def fit(self):
        flyings = [hasattr(s, 'ball') and s.ball.state == BallState.FLYING for s in self.window]
        if sum(flyings) < self.min_flyings:
            self.print(flyings, color=ModelFit.DISCARDED, label="not enough flying balls")
            return None

        nonflyings_ratio = 1 - sum(flyings)/(np.ptp(np.where(flyings)) + 1) if np.any(flyings) else 0
        if nonflyings_ratio > self.max_nonflyings_ratio:
            self.print(flyings, color=ModelFit.DISCARDED, label="too many non-flyings")
            return None

        model = super().fit()
        if model and (self.min_flyings or self.max_nonflyings_ratio):
            self.print(flyings, color=ModelFit.PROPOSED, label='(flyings)')
        return model


def model(*, min_window_length, min_inliers, max_outliers_ratio,
    max_inliers_decrease, min_distance_cm, min_distance_px,
    min_flyings, max_nonflyings_ratio,
    d_error_weight, p_error_threshold, scale, **kwargs):

    return BallStateSlidingWindow(
        min_window_length=min_window_length,
        min_inliers=min_inliers,
        max_outliers_ratio=max_outliers_ratio,
        max_inliers_decrease=max_inliers_decrease,
        min_distance_cm=min_distance_cm,
        min_distance_px=min_distance_px,
        min_flyings=min_flyings,
        max_nonflyings_ratio=max_nonflyings_ratio,
        inliers_condition = lambda p_error, d_error: p_error < p_error_threshold * (1 + scale*np.sin(np.linspace(0, np.pi, len(p_error)))),
        error_fct = lambda p_error, d_error: np.linalg.norm(p_error) + d_error_weight*np.linalg.norm(d_error),
        **kwargs,
    )


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
    def __len__(self):
        return self.end_key.timestamp - self.start_key.timestamp


def compute_projection_error(true: Point3D, pred: Point3D):
    difference = true - pred
    difference.z = 0 # set z coordinate to 0 to compute projection error on the ground
    return np.linalg.norm(difference, axis=0)

class MatchTrajectories:
    def __init__(self, min_duration=250, callback=None):
        self.TP = []
        self.FP = []
        self.FN = []
        self.dist_T0 = []
        self.dist_TN = []
        self.callback = callback or (lambda a, p, t: None)
        self.annotations = []
        self.predictions = []
        self.min_duration = min_duration
        self.detections_MAPE = []
        self.ballistic_MAPE = []
        self.recovered = []
        self.splitted_predicted_trajectories = 0
        self.splitted_annotated_trajectories = 0
        self.overlap = []

    def TP_callback(self, a, p):
        self.TP.append((a.trajectory_id, p.trajectory_id))
        self.dist_T0.append(p.start_key.timestamp - a.start_key.timestamp)
        self.dist_TN.append(p.end_key.timestamp - a.end_key.timestamp)
        self.recovered.append(len([s for s in p.samples if s.ball.origin == RECOVERED_BALL_ORIGIN]))
        self.overlap.append(a - p)

        # compute MAPE, MARE, MADE if ball 3D position was annotated
        if any([s.ball_annotations and np.abs(s.ball_annotations[0].center.z) > 0.1 for s in a.samples]):
            annotated_trajectory_samples = {s.key: s for s in a.samples if s.ball_annotations}
            predicted_trajectory_samples = {s.key: s for s in p.samples if s.ball.origin is not RECOVERED_BALL_ORIGIN}
            keys = set(annotated_trajectory_samples.keys()) & set(predicted_trajectory_samples.keys())
            detected_ball3D =  Point3D([predicted_trajectory_samples[k].ball.center for k in keys])
            annotated_ball3D = Point3D([annotated_trajectory_samples[k].ball_annotations[0].center for k in keys])
            ballistic_ball3D = Point3D([predicted_trajectory_samples[k].ball.model(predicted_trajectory_samples[k].ball.timestamp) for k in keys])
            self.detections_MAPE.extend(compute_projection_error(annotated_ball3D, detected_ball3D))
            self.ballistic_MAPE.extend(compute_projection_error(annotated_ball3D, ballistic_ball3D))

        self.callback(a, p, 'TP')

    def FN_callback(self, a, p):
        self.splitted_annotated_trajectories += (1 if p is not None else 0)
        if len(a) < self.min_duration:
            return
        self.FN.append(a.trajectory_id)
        self.callback(a, p, 'FN')

    def FP_callback(self, a, p):
        self.splitted_predicted_trajectories += (1 if a is not None else 0)
        if len(p) < self.min_duration:
            return
        self.FP.append(p.trajectory_id)
        self.callback(a, p, 'FP')

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
            # skip samples without ball model
            if not hasattr(sample, 'ball') or not hasattr(sample.ball, 'model') or sample.ball.model is None:
                self.predictions.append(0)
                continue

            # if model changed, yield previous trajectory
            if sample.ball.model != model and model is not None:
                yield Trajectory(samples, trajectory_id)
                trajectory_id += 1
                samples = []

            model = sample.ball.model
            samples.append(sample)
            self.predictions.append(trajectory_id)

    def __call__(self, agen, pgen):
        pgen = self.extract_predicted_trajectories(pgen)
        agen = self.extract_annotated_trajectories(agen)
        try:
            p = next(pgen)
            a = next(agen)
            while True:
                # skip annotated and predicted trajectories until an overlap is found
                while a < p:
                    self.FN_callback(a, None)
                    a = next(agen)
                while p < a:
                    self.FP_callback(None, p)
                    p = next(pgen)

                if p.start_key <= a.end_key <= p.end_key:
                    # keep the annotated trajectory that maximizes the overlap with the predicted trajectory
                    while (a2 := next(agen)).start_key <= p.end_key:
                        if a2 - p > a - p:
                            self.FN_callback(a, p)
                            a = a2
                        else:
                            self.FN_callback(a2, p)
                    self.TP_callback(a, p)
                    p = next(pgen)
                    a = a2
                elif a.start_key <= p.end_key <= a.end_key:
                    # keep the predicted trajectory that maximizes the overlap with the annotated trajectory
                    while (p2 := next(pgen)).start_key <= a.end_key:
                        if p2 - a > p - a:
                            self.FP_callback(a, p)
                            p = p2
                        else:
                            self.FP_callback(a, p2)
                    self.TP_callback(a, p)
                    a = next(agen)
                    p = p2
                else:
                    pass # no match, move-on to next annotated trajectory
        except StopIteration:
            # Consume remaining trajectories
            try:
                while (p := next(pgen)):
                    self.FP_callback(None, p)
            except StopIteration:
                pass
            try:
                while (a := next(agen)):
                    self.FN_callback(a, None)
            except StopIteration:
                pass

    @property
    def metrics(self):
        mean = lambda x: np.mean(x) if np.any(x) else np.nan
        return {
            'TP': len(self.TP),
            'FP': len(self.FP),
            'FN': len(self.FN),
            'recovered': sum(self.recovered),
            'overlap': sum(self.overlap),
            'mean_dist_T0': mean(self.dist_T0),
            'dist_T0': mean(np.abs(self.dist_T0)),
            'mean_dist_TN': mean(self.dist_TN),
            'dist_TN': mean(np.abs(self.dist_TN)),
            'precision': len(self.TP) / (len(self.TP) + len(self.FP)) if len(self.TP) + len(self.FP) > 0 else 0,
            'recall': len(self.TP) / (len(self.TP) + len(self.FN)) if len(self.TP) + len(self.FN) > 0 else 0,
            'splitted_predicted_trajectories': self.splitted_predicted_trajectories,
            'splitted_annotated_trajectories': self.splitted_annotated_trajectories,
            'ballistic_MAPE': mean(self.ballistic_MAPE),
            'detections_MAPE': mean(self.detections_MAPE),
        }


class InstantRenderer():
    def __init__(self, ids: InstantsDataset, min_duration: int):
        self.ids = ids
        self.min_duration = min_duration

    def draw_ball(self, pd, image, ball, color=None, label=None):
        color = color or pd.color
        ground3D = Point3D(ball.center.x, ball.center.y, 0)
        pd.draw_line(image, ball.center,               ground3D,                  lineType=cv2.LINE_AA, color=color)
        pd.draw_line(image, ground3D+Point3D(100,0,0), ground3D-Point3D(100,0,0), lineType=cv2.LINE_AA, thickness=1, color=color)
        pd.draw_line(image, ground3D+Point3D(0,100,0), ground3D-Point3D(0,100,0), lineType=cv2.LINE_AA, thickness=1, color=color)
        center = pd.calib.project_3D_to_2D(ball.center).to_int_tuple()
        radius = pd.calib.compute_length2D(ball.center, BALL_DIAMETER/2)

        cv2.circle(image, center, int(radius), color, 1)
        if label is not None:
            cv2.putText(image, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def __call__(self, sample):
        instant = self.ids.query_item(sample.key)
        for image, calib in zip(instant.images, instant.calibs):
            pd = ProjectiveDrawer(calib, (0, 120, 255), segments=1)

            if ball := getattr(sample, "ball", None):
                if model := getattr(ball, "model", None):
                    self.draw_ball(pd, image, ball, label=str(ball.state))
                    duration = model.end_timestamp - model.start_timestamp
                    pd.color = (250, 195, 0) if duration > self.min_duration else (250, 200, 30)
                    timestamps = np.linspace(model.start_timestamp, model.end_timestamp, int(np.ceil(duration/10)))
                    points3D = model(timestamps)
                    ground3D = Point3D(points3D.x, points3D.y, np.zeros_like(points3D.x))
                    start = Point3D(np.vstack([points3D[:, 0], ground3D[:, 0]]).T)
                    stop  = Point3D(np.vstack([points3D[:, -1], ground3D[:, -1]]).T)
                    for line in [points3D, ground3D, start, stop]:
                        pd.polylines(image, line, lineType=cv2.LINE_AA)
                elif sample.ball_state == BallState.FLYING:
                    self.draw_ball(pd, image, ball, color=(255, 0, 0), label=str(ball.state))

            # draw ball annotation if any
            if sample.ball_annotations:
                ball = sample.ball_annotations[0]
                if ball.center.z < -0.01: # z pointing down
                    color = (0, 255, 20)
                    self.draw_ball(pd, image, ball, color=color)
            cv2.putText(image, str(sample.ball_state), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 20), lineType=cv2.LINE_AA)
        return np.hstack(instant.images)

class TrajectoryRenderer(InstantRenderer):
    def __call__(self, annotated: Trajectory, predicted: Trajectory):
        annotated_trajectory_samples = {sample.key: sample for sample in annotated.samples} if annotated else {}
        predicted_trajectory_samples = {sample.key: sample for sample in predicted.samples} if predicted else {}
        samples = {**annotated_trajectory_samples, **predicted_trajectory_samples} # prioritize predicted trajectory samples
        for key, sample in samples.items():
            instant = self.ids.query_item(key)
            yield super().__call__(instant, sample)



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


