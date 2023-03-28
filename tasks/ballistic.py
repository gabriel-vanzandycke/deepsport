from dataclasses import dataclass
import warnings

from calib3d import Point3D, ProjectiveDrawer
import cv2
import numpy as np

from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.ds.instants_dataset import InstantsDataset, BallState
from models.ballistic import FITTED_BALL_ORIGIN


np.set_printoptions(precision=3, linewidth=110)#, suppress=True)



def compute_projection_error(true_center: Point3D, pred_center: Point3D):
    difference = true_center - pred_center
    difference.z = 0 # set z coordinate to 0 to compute projection error on the ground
    return np.linalg.norm(difference, axis=0)

@dataclass
class ComputeSampleMetrics:
    min_duration: int = 250 # ballistic model shorter than `min_duration` (in miliseconds) won't be counted as FP
    min_duration_TP: bool = False # if True, ballistic model shorter than `min_duration` (in miliseconds) won't be counted as TP
    def __post_init__(self):
        self.TP = self.FP = self.FN = self.TN = self.interpolated = 0
        self.ballistic_all_MAPE = []         # MAPE between GT and ballistic models or detections if no ballistic model is found
        self.ballistic_restricted_MAPE = []  # MAPE between GT and detected ballistic models
        self.detections_MAPE = []            # MAPE between GT and all original detections
    def __call__(self, gen):
        for i, sample in enumerate(gen):
            if hasattr(sample, 'ball'):
                if hasattr(sample.ball, 'model'):
                    if sample.ball_state == BallState.FLYING \
                     and (not self.min_duration_TP or sample.ball.model.window.duration >= self.min_duration): # new here
                        self.TP += 1
                    elif sample.ball_state != BallState.NONE \
                     and sample.ball.model.window.duration >= self.min_duration:
                        self.FP += 1
                    if sample.ball.origin == FITTED_BALL_ORIGIN:
                        self.interpolated += 1
                else:
                    if sample.ball_state == BallState.FLYING:
                        self.FN += 1
                    elif sample.ball_state != BallState.NONE:
                        self.TN += 1
                true_balls = [a for a in getattr(sample, 'ball_annotations', []) if a.origin in ['interpolation', 'annotation']]
                if true_balls:
                    true_center = true_balls[0].center
                    if hasattr(sample.ball, 'model'):
                        error = compute_projection_error(true_center, sample.ball.model(sample.timestamp))
                        self.ballistic_restricted_MAPE.append(error)
                    else:
                        error = compute_projection_error(true_center, sample.ball.center)
                    self.ballistic_all_MAPE.append(error)
                    if sample.ball.origin != FITTED_BALL_ORIGIN:
                        self.detections_MAPE.append(compute_projection_error(true_center, sample.ball.center))
            yield sample
    @property
    def metrics(self):
        return {
            "TP": self.TP,
            "FP": self.FP,
            "FN": self.FN,
            "TN": self.TN,
            "ballistic_all_MAPE": np.mean(self.ballistic_all_MAPE),
            "ballistic_restricted_MAPE": np.mean(self.ballistic_restricted_MAPE),
            "detections_MAPE": np.mean(self.detections_MAPE),
            "interpolated": self.interpolated,
            's_precision': self.TP / (self.TP + self.FP) if self.TP > 0 else 0,
            's_recall': self.TP / (self.TP + self.FN) if self.TP > 0 else 0,
        }


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
    def __add__(self, other):
        return max(self.end_key.timestamp,   other.end_key.timestamp) \
             - min(self.start_key.timestamp, other.start_key.timestamp)
    def __len__(self):
        return self.end_key.timestamp - self.start_key.timestamp




class MatchTrajectories:
    # compute MAPE, MARE, MADE if ball 3D position was annotated for FP when splitted as well.")
    def __init__(self, min_duration=250, callback=None):
        warnings.warn("not implemented: compute MAPE, MARE, MADE if ball 3D position was annotated for FP when splitted as well.")
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
        self.union = []
        self.intersection = []
        self.s_TP = 0
        self.s_FP = 0
        self.s_FN = 0


    def compute_samples_TP_FP_FN(self, a, p):
        samples = {sample.key: sample for sample in p.samples} if p else {}

        annotated_trajectory_samples = set([sample.key for sample in a.samples]) if a else set([])
        predicted_trajectory_samples = set([sample.key for sample in p.samples]) if p else set([])
        self.s_TP += len(annotated_trajectory_samples.intersection(predicted_trajectory_samples))
        self.s_FN += len(annotated_trajectory_samples.difference(predicted_trajectory_samples))
        s_FP = predicted_trajectory_samples.difference(annotated_trajectory_samples)
        self.s_FP += len([k for k in s_FP if samples[k].ball_state != BallState.NONE])

    def TP_callback(self, a, p):
        self.TP.append((a.trajectory_id, p.trajectory_id))
        self.dist_T0.append(p.start_key.timestamp - a.start_key.timestamp)
        self.dist_TN.append(p.end_key.timestamp - a.end_key.timestamp)
        self.recovered.append(len([s for s in p.samples if s.ball.origin == FITTED_BALL_ORIGIN]))
        self.overlap.append(a - p)
        self.intersection.append(a - p)
        self.union.append(a + p)
        self.compute_samples_TP_FP_FN(a, p)

        # compute MAPE, MARE, MADE if ball 3D position was annotated
        if any([s.ball_annotations and np.abs(s.ball_annotations[0].center.z) > 0.1 for s in a.samples]):
            annotated_trajectory_samples = {s.key: s for s in a.samples if s.ball_annotations}
            predicted_trajectory_samples = {s.key: s for s in p.samples if s.ball.origin is not FITTED_BALL_ORIGIN}
            keys = set(annotated_trajectory_samples.keys()) & set(predicted_trajectory_samples.keys())
            detected_ball3D =  Point3D([predicted_trajectory_samples[k].ball.center for k in keys])
            annotated_ball3D = Point3D([annotated_trajectory_samples[k].ball_annotations[0].center for k in keys])
            ballistic_ball3D = Point3D([predicted_trajectory_samples[k].ball.model(predicted_trajectory_samples[k].timestamp) for k in keys])
            self.detections_MAPE.extend(compute_projection_error(annotated_ball3D, detected_ball3D))
            self.ballistic_MAPE.extend(compute_projection_error(annotated_ball3D, ballistic_ball3D))

        self.callback(a, p, 'TP')

    def FN_callback(self, a, p):
        if p is not None:
            self.intersection.append(a - p)
            self.union.append(a + p)
        else:
            self.union.append(len(a))
        self.splitted_annotated_trajectories += (1 if p is not None else 0)
        self.compute_samples_TP_FP_FN(a, p)
        if len(a) < self.min_duration:
            return
        self.FN.append(a.trajectory_id)
        self.callback(a, p, 'FN')

    def FP_callback(self, a, p):
        if a is not None:
            self.intersection.append(a - p)
            self.union.append(a + p)
        else:
            self.union.append(len(p))
        self.splitted_predicted_trajectories += (1 if a is not None else 0)
        self.compute_samples_TP_FP_FN(a, p)
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
            # skip samples without valid ball model
            if not hasattr(sample, 'ball') \
            or not hasattr(sample.ball, 'model') \
            or sample.ball.model is None \
            or not isinstance(getattr(sample.ball.model, "mark", ModelMarkAccepted()), ModelMarkAccepted):
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
            'IoU': sum(self.intersection)/sum(self.union),
            's_TP': self.s_TP,
            's_FP': self.s_FP,
            's_FN': self.s_FN,
            's_precision': self.s_TP / (self.s_TP + self.s_FP) if self.s_TP + self.s_FP > 0 else 0,
            's_recall': self.s_TP / (self.s_TP + self.s_FN) if self.s_TP + self.s_FN > 0 else 0,
        }


class InstantRenderer():
    def __init__(self, ids: InstantsDataset):
        self.ids = ids
        self.font_size = .8

    def draw_ball(self, pd, image, ball, color=None, label=None):
        color = color or pd.color
        ground3D = Point3D(ball.center.x, ball.center.y, 0)
        pd.draw_line(image, ball.center,               ground3D,                  lineType=cv2.LINE_AA, color=color)
        pd.draw_line(image, ground3D+Point3D(100,0,0), ground3D-Point3D(100,0,0), lineType=cv2.LINE_AA, thickness=1, color=color)
        pd.draw_line(image, ground3D+Point3D(0,100,0), ground3D-Point3D(0,100,0), lineType=cv2.LINE_AA, thickness=1, color=color)
        radius = pd.calib.compute_length2D(ball.center, BALL_DIAMETER/2)
        x, y = pd.calib.project_3D_to_2D(ball.center).to_int_tuple()

        cv2.circle(image, (x, y), int(radius), color, 1)
        if label is not None:
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, color, 2, lineType=cv2.LINE_AA)

    def draw_model(self, pd, image, model, color, label):
        start_timestamp = model.window[0].timestamp
        end_timestamp = model.window[-1].timestamp
        duration = end_timestamp - start_timestamp
        timestamps = np.linspace(start_timestamp, end_timestamp, int(np.ceil(duration/10)+1))
        points3D = model(timestamps)
        ground3D = Point3D(points3D.x, points3D.y, np.zeros_like(points3D.x))
        start = Point3D(np.vstack([points3D[:, 0], ground3D[:, 0]]).T)
        stop  = Point3D(np.vstack([points3D[:, -1], ground3D[:, -1]]).T)
        for line in [points3D, ground3D, start, stop]:
            pd.polylines(image, line, color=color, lineType=cv2.LINE_AA)

        points3D = model(model.window.timestamps)
        points2D = pd.calib.project_3D_to_2D(points3D)
        radii = pd.calib.compute_length2D(points3D, BALL_DIAMETER/2)
        for i, (point3D, point2D, radius) in enumerate(zip(points3D, points2D, radii)):
            pd.draw_line(image, point3D, model.window[i].ball.center, (200, 200, 200), 1)
            radius = 2
            cv2.circle(image, point2D.to_int_tuple(), int(radius), color, -1)

        # Write model mark
        point3D = Point3D(points3D[:, 0])
        point3D.z = 0
        x, y = pd.calib.project_3D_to_2D(point3D).to_int_tuple()
        cv2.putText(image, label, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (250, 20, 30), 2, lineType=cv2.LINE_AA)


    def __call__(self, sample):
        instant = self.ids.query_item(sample.key)
        for image, calib in zip(instant.images, instant.calibs):
            pd = ProjectiveDrawer(calib, (0, 120, 255), segments=1)

            if ball := getattr(sample, "ball", None):
                for model in getattr(sample, "models", []):
                    self.draw_model(pd, image, model, color=(255, 0, 20), label=model.message)
                if model := getattr(ball, "model", None):
                    self.draw_model(pd, image, model, color=(250, 195, 0), label="")
                color = (150, 150, 150) if hasattr(ball, 'model') else ((255, 0, 0) if sample.ball_state == BallState.FLYING else (0, 120, 255))
                label = f"{ball.value:0.2f} - {str(ball.state)}" if ball.value else f"{str(ball.state)}"
                self.draw_ball(pd, image, ball, color=color, label=label)

            # draw annotations
            color = (0, 255, 20)
            if sample.ball_annotations:
                ball = sample.ball_annotations[0]
                if ball.origin in ['annotation', 'interpolation']:
                    self.draw_ball(pd, image, ball, color=color)
            cv2.putText(image, str(sample.ball_state), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, color, 2, lineType=cv2.LINE_AA)

        return np.hstack(instant.images)

class TrajectoryRenderer(InstantRenderer):
    def __call__(self, annotated: Trajectory, predicted: Trajectory):
        annotated_trajectory_samples = {sample.key: sample for sample in annotated.samples} if annotated else {}
        predicted_trajectory_samples = {sample.key: sample for sample in predicted.samples} if predicted else {}
        samples = {**annotated_trajectory_samples, **predicted_trajectory_samples} # prioritize predicted trajectory samples
        for key, sample in samples.items():
            instant = self.ids.query_item(key)
            yield from super().__call__(instant, sample)


class SelectBall:
    def __init__(self, origin):
        self.origin = origin
    def __call__(self, key, item):
        try:
            item.ball = max([d for d in item.ball_detections if d.origin == self.origin], key=lambda d: d.value)
            item.timestamp = item.timestamps[item.ball.camera]
            item.calib = item.calibs[item.ball.camera]
        except ValueError:
            pass
        item.models = []
        return item


