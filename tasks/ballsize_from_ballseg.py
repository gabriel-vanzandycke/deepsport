from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
import pandas

import numpy as np
import tensorflow as tf

from experimentator import Callback, ExperimentMode, build_experiment
from experimentator.utils import find
from experimentator.tf2_chunk_processors import ChunkProcessor
from experimentator.tf2_experiment import TensorflowExperiment

from dataset_utilities.court import Court
from dataset_utilities.calib import Calib, Point3D, Point2D

from .detection import ComputeTopkMetrics
from .ballsize import BALL_DIAMETER

class BallSizeFromBallSegExperiment(TensorflowExperiment):
    batch_inputs_names = ["batch_target", "batch_input_image", "batch_ball_size"]
    batch_metrics_names = ["batch_heatmap", "batch_input_image", "topk_indices",
        "target_is_ball", "predicted_is_ball", "batch_ball_size", "predicted_diameter",
        "ballseg_topk_TP", "ballseg_topk_FP", "ballseg_P", "ballseg_N"]
    batch_outputs_names = ["predicted_diameter", "predicted_is_ball"]
    def save_weights(self, filename):
        self.train_model.get_layer('ballsize').save_weights(filename)
    def load_weights(self, filename):
        super().load_weights(filename) # sets the `self.weights_file` variable
        self.train_model.get_layer('ballsize').load_weights(self.weights_file)
        self.weights_file = None


@dataclass
class ReapeatData(Callback):
    before = ["ComputeDiameterError"]
    when = ExperimentMode.EVAL
    def __init__(self, names, k):
        self.names = names
        self.k = k
    def on_batch_end(self, **state):
        for name in self.names:
            state[name] = [v for v in state[name] for k in range(self.k)]


@dataclass
class ComputeBallSegTopkMetrics(ComputeTopkMetrics):
    def on_batch_end(self, state, **_):
        ballseg_state = {name: state[f"ballseg_{name}"] for name in ["topk_TP", "topk_FP", "P", "N"]}
        super().on_batch_end(ballseg_state)
    def on_cycle_end(self, state, **_):
        ballseg_state = {}
        super().on_cycle_end(ballseg_state)
        state.update(**{f"ballseg_{name}": ballseg_state[name] for name in ballseg_state})


@dataclass
class HoughCircle():
    size_min: int = None
    size_max: int = None
    sigma: float = .2
    normalize: bool = False
    step: float = .2
    discard_extremities: bool = True
    def __call__(self, mask, size_min=None, size_max=None):
        mask = np.uint8(mask*255)
        assert np.max(mask) <= 255, "Expecting a mask in [0,1]"

        # Load picture and detect edges
        edges = skimage.feature.canny(mask, sigma=self.sigma, low_threshold=10, high_threshold=20)

        # Test different radius
        size_min = size_min or self.size_min
        size_max = size_max or self.size_max
        min_radius = int(self.size_min/2) # radius = diameter/2
        max_radius = int(self.size_max/2) # radius = diameter/2
        radii = np.arange(min_radius, max_radius+1, self.step)

        H = skimage.transform.hough_circle(edges, radii, normalize=self.normalize)

        # Select the most prominent N circles
        _, cxs, cys, radius = skimage.transform.hough_circle_peaks(H, radii, total_num_peaks=1)

        if not np.any(radius) or (self.discard_extremities and radius in [min_radius, max_radius]):
            return np.nan, np.nan, np.nan
        return cxs[0], cys[0], radius[0]*2

def expected_range(calib: Calib, ball: Point2D, court: Court=None, max_ball_height=500):
    court = court or Court('FIBA')
    bz = calib.project_2D_to_3D(ball, Z=-BALL_DIAMETER/2)
    by = calib.project_2D_to_3D(ball, Y=-100)
    bx = calib.project_2D_to_3D(ball, X=-200 if bz.x < court.w/2 else court.w-200)
    size_min = max(calib.compute_length2D(BALL_DIAMETER, Point3D(np.hstack([bx, by, bz]))))
    bz = calib.project_2D_to_3D(ball, Z=-max_ball_height)
    by = calib.project_2D_to_3D(ball, Y=court.h+100)
    size_max = min(calib.compute_length2D(BALL_DIAMETER, Point3D(np.hstack([by, bz]))))
    return size_min, size_max

@dataclass
class ComputeDiameterErrorFromPatchHeatmap(Callback):
    def __post_init__(self):
        self.hc = HoughCircle()
    def init(self, exp):
        self.batch_size = exp.batch_size
        self.k = exp.cfg.get('k', 1)
    def on_cycle_begin(self, **_):
        self.acc = defaultdict(lambda: [])
    def on_batch_end(self, batch_heatmap, batch_ball_size, batch_ball, batch_calib, **_):
        assert self.batch_size*self.k == len(batch_heatmap)
        for i, (heatmap, ball_size) in enumerate(zip(batch_heatmap, batch_ball_size)):#, batch_ball, batch_calib):
            if np.isnan(ball_size): # ignore patches without ball
                continue

            ball = Point3D(batch_ball[i])
            calib = batch_calib[i]

            size_min, size_max = expected_range(calib, calib.project_3D_to_2D(ball))

            true_diameter = ball_size
            _, _, predicted_diameter = self.hc(heatmap, size_min=size_min, size_max=size_max)
            diameter_error = predicted_diameter - true_diameter
            projection_error = compute_projection_error(calib, ball, diameter_error)[0]
            relative_error = compute_relative_error(calib, ball, predicted_diameter)

            self.acc["true_diameter"].append(true_diameter)
            self.acc["predicted_diameter"].append(predicted_diameter)
            self.acc["diameter_error"].append(diameter_error)
            self.acc["projection_error"].append(projection_error)
            self.acc["relative_error"].append(relative_error)

    def on_cycle_end(self, state, **_): # state in R/W mode
        try:
            df = pandas.DataFrame(np.vstack(list(self.acc.values())).T, columns=self.acc.keys())
            state["ballseg_ball_size_metrics"] = df
            state["ballseg_MADE"] = np.mean(np.abs(df['diameter_error']))
            state["ballseg_MAPE"] = np.mean(np.abs(df['projection_error']))
            state["ballseg_MARE"] = np.mean(np.abs(df['relative_error']))
        except ValueError:
            state["ballseg_ball_size_metrics"] = None
            for name in ["MADE", "MAPE", "MARE"]:
                state[name] = np.nan




class BallSegModel(ChunkProcessor):
    def __init__(self, config=None, **kwargs):
        self.config = config or "configs/ballseg.py"
        self.kwargs = kwargs
    @cached_property
    def model(self):
        exp = build_experiment(find(self.config), **self.kwargs)
        exp.load_weights()
        exp.metrics.update(exp.outputs)
        model = exp.eval_model
        model._name = "ballseg"
        model.trainable = False
        return model
    def __call__(self, chunk):
        inputs = {name: chunk[name] for name in chunk if name in ["batch_input_image", "batch_target"]}
        chunk.update(self.model(inputs))

class BallSegCandidates(ChunkProcessor):
    def __init__(self, side_length, oracle=False):
        self.side_length = side_length
        self.oracle = oracle

    def __call__(self, chunk):
        ks = range(chunk['topk_indices'].get_shape()[-2])
        if not self.oracle:
            # (0,0) means using first keypoint type and lowest threshold
            offsets = lambda chunk, k: chunk['topk_indices'][:,0,0,k]
        else:
            assert len(ks) == 1, "Oracle detection can only work when k=1 (as placeholders are constructed using k)"
            # take mean position on the target mask
            offsets = lambda chunk, _: tf.stack([tf.reduce_mean(tf.where(t), axis=0) for t in chunk["batch_target"]])

        # extracting k patches from input image using `offsets`
        chunk["batch_input_image"] = tf.stack([
            tf.image.extract_glimpse(
                tf.cast(chunk['batch_input_image'], tf.float32),
                size=(self.side_length, self.side_length),
                offsets=tf.cast(offsets(chunk, k), tf.float32) - self.side_length//2,
                centered=False, normalized=False, noise='zero'
            ) for k in ks
        ], 1)
        chunk["batch_input_image"] = tf.reshape(chunk["batch_input_image"], [-1, self.side_length, self.side_length, 3])

        # extracting k patches from output heatmap using `offsets`
        chunk["batch_heatmap"] = tf.stack([
            tf.image.extract_glimpse(
                tf.cast(chunk['batch_heatmap'], tf.float32),
                size=(self.side_length, self.side_length),
                offsets=tf.cast(offsets(chunk, k), tf.float32) - self.side_length//2,
                centered=False, normalized=False, noise='zero'
            ) for k in ks
        ], 1)
        chunk["batch_heatmap"] = tf.reshape(chunk["batch_heatmap"], [-1, self.side_length, self.side_length])

        # batch_ball_size stays the same when using oracle detections
        if self.oracle:
            pass
        elif 'topk_targets' in chunk:
            chunk["batch_ball_size"] = tf.cast(
                tf.stack([chunk['topk_targets'][:,0,0,k] for k in ks], 1),
                tf.float64
            ) * chunk["batch_ball_size"][:,tf.newaxis]
            # chunk["batch_ball_size"] = tf.cast(tf.stack([
            #     tf.gather_nd(chunk['batch_target'], chunk['topk_indices'][:,0,0,k], batch_dims=1)
            #     for k in ks
            # ], 1), tf.float64) * chunk["batch_ball_size"][:,tf.newaxis]
            chunk["batch_ball_size"] = tf.where(
                chunk["batch_ball_size"]==0,
                chunk["batch_ball_size"]*np.nan,
                chunk["batch_ball_size"]
            )
            chunk["batch_ball_size"] = tf.reshape(chunk["batch_ball_size"], [-1])


        #chunk["batch_ballseg_value"] =
        for name in ['P', 'N', 'topk_TP', 'topk_FP']:
            if name in chunk:
                chunk["ballseg_P"] = chunk['P']
                chunk["ballseg_N"] = chunk['N']
                chunk["ballseg_topk_TP"] = chunk['topk_TP']
                chunk["ballseg_topk_FP"] = chunk['topk_FP']


class CNNModel(ChunkProcessor):
    def __init__(self, config=None, **kwargs):
        self.config = config or "configs/ballsize.py"
        self.kwargs = kwargs
    @cached_property
    def model(self):
        exp = build_experiment(find(self.config), **self.kwargs)
        exp.load_weights()
        model = exp.eval_model
        model._name = "ballsize"
        return model
    def __call__(self, chunk):
        inputs = {name: chunk[name] for name in chunk if name in ['batch_input_image', 'batch_ball_size']}
        chunk.update(self.model(inputs))