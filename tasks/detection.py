from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas
import tensorflow as tf

from tf_layers import AvoidLocalEqualities, PeakLocalMax, ComputeElementaryMetrics

from experimentator import Callback, ChunkProcessor, ExperimentMode
from experimentator.tf2_experiment import TensorflowExperiment

class HeatmapDetectionExperiment(TensorflowExperiment):
    batch_inputs_names = ["batch_target", "batch_input_image"]
    @cached_property
    def metrics(self):
        metrics = ["TP", "FP", "TN", "FN", "topk_TP", "topk_FP", "P", "N"]
        return {
            name:self.chunk[name] for name in metrics if name in self.chunk
        }
    @cached_property
    def outputs(self):
        outputs = ["batch_heatmap", "topk_indices", "topk_outputs", "topk_targets"]
        return {
            name:self.chunk[name] for name in outputs if name in self.chunk
        }


def divide(num: np.ndarray, den: np.ndarray):
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den>0)

@dataclass
class ComputeMetrics(Callback):
    before = ["AuC", "GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    thresholds: (int, np.ndarray, list, tuple) = np.linspace(0,1,51)
    class_index: int = 0
    def on_cycle_begin(self, **_):
        self.acc = {}
    def on_batch_end(self, state, **_): # 'state' argument in R/W
        for name in ["TP", "FP", "TN", "FN"]:
            if name in state:
                value = np.sum(state[name], axis=0) # sums the batch dimension
                self.acc[name] = self.acc.setdefault(name, np.zeros_like(value)) + value
    def on_cycle_end(self, state, **_):
        TP = self.acc["TP"][self.class_index]
        FP = self.acc["FP"][self.class_index]
        TN = self.acc["TN"][self.class_index]
        FN = self.acc["FN"][self.class_index]

        data = {
            "thresholds": self.thresholds,
            "accuracy": (TP+TN)/(TP+TN+FP+FN),
            "precision": divide(TP, TP+FP),
            "recall": divide(TP, TP+FN),
            "TP rate": divide(TP, TP+FN),
            "FP rate": divide(FP, FP+TN),
        }
        state["metrics"] = pandas.DataFrame(np.vstack([data[name] for name in data]).T, columns=list(data.keys()))

@dataclass
class ComputeTopkMetrics(Callback):
    """
        Arguments:
            k  list of top-k of interest (e.g. [1,3,10] for top-1, top-3 and top-10)
    """
    before = ["AuC", "GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    k: (tuple, list, np.ndarray)
    thresholds: (int, np.ndarray, list, tuple) = np.linspace(0,1,51)
    class_index: int = 0
    def on_cycle_begin(self, **_):
        self.acc = {}
    def on_batch_end(self, state, **_): # 'state' argument in R/W
        for name in ["topk_TP", "topk_FP", "P", "N"]:
            if name in state:
                value = np.sum(state[name], axis=0) # sums the batch dimension
                self.acc[name] = self.acc.setdefault(name, np.zeros_like(value)) + value
    def on_cycle_end(self, state, **_):
        for k in self.k:
            FP = np.sum(self.acc["topk_FP"][self.class_index, :, 0:k], axis=1)
            TP = np.sum(self.acc["topk_TP"][self.class_index, :, 0:k], axis=1)
            P = self.acc["P"][np.newaxis]
            N = self.acc["N"][np.newaxis]
            data = {
                "thresholds": self.thresholds,
                "FP rate": divide(FP, P + N),  # #-possible cases is the number of images
                "TP rate": divide(TP, P),      # #-possible cases is the number of images on which there's a ball to detect
                "precision": divide(TP, TP + FP),
                "recall": divide(TP, P),
            }
            state[f"top{k}_metrics"] = pandas.DataFrame(np.vstack([data[name] for name in data]).T, columns=list(data.keys()))

        # TODO: handle multple classes cases (here only class index is picked and the rest is discarded)

@dataclass
class AuC(Callback):
    after = ["ComputeTopkMetrics", "ComputeMetrics", "HandleMetricsPerBallSize"]
    before = ["GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    name: str
    table_name: str
    x_label: str = "FP rate"
    y_label: str = "TP rate"
    x_lim: int = 1
    close_curve: bool = True
    def on_cycle_end(self, state, **_):
        x = state[self.table_name][self.x_label][::-1]
        y = state[self.table_name][self.y_label][::-1]

        state[self.name] = self(x,y)
    def __call__(self, x, y):
        auc = 0
        for xi1, yi1, xi2, yi2 in zip(x, y, x[1:], y[1:]):
            if xi1 == xi2:
                continue
            if xi2 >= self.x_lim: # last trapezoid
                auc += (yi1+yi2)*(self.x_lim-xi1)/2
                break
            auc += (yi1+yi2)*(xi2-xi1)/2

        if self.close_curve:
            auc += (1-xi2)*yi2 # pylint: disable=undefined-loop-variable
        return auc


class ComputeKeypointsDetectionHitmap(ChunkProcessor):
    def __init__(self, non_max_suppression_pool_size=50, threshold=np.linspace(0,1,51)):
        if isinstance(threshold, np.ndarray):
            thresholds = threshold
        elif isinstance(threshold, list):
            thresholds = np.array(threshold)
        elif isinstance(threshold, float):
            thresholds = np.array([threshold])
        else:
            raise ValueError(f"Unsupported type for input argument 'threshold'. Recieved {threshold}")
        assert len(thresholds.shape) == 1, "'threshold' argument should be 1D-array (a scalar is also accepted)."

        # Saved here for 'config' property
        self.non_max_suppression_pool_size = non_max_suppression_pool_size
        self.threshold = threshold

        self.avoid_local_eq = AvoidLocalEqualities()
        self.peak_local_max = PeakLocalMax(min_distance=non_max_suppression_pool_size//2, thresholds=thresholds)

    def __call__(self, chunk):
        chunk["batch_hitmap"] = self.peak_local_max(self.avoid_local_eq(chunk["batch_heatmap"])) # B,H,W,C,T [bool]

class ComputeKeypointsDetectionMetrics(ChunkProcessor):
    def __init__(self):
        self.compute_metric = ComputeElementaryMetrics()

    def __call__(self, chunk):
        batch_hitmap = tf.cast(chunk["batch_hitmap"], tf.int32) # B,H,W,C,T
        batch_target = tf.cast(chunk["batch_target"], tf.int32)[..., tf.newaxis] # B,H,W,C,T

        batch_metric = self.compute_metric(batch_hitmap=batch_hitmap, batch_target=batch_target)
        chunk["TP"] = batch_metric["batch_TP"] # B x K x C
        chunk["FP"] = batch_metric["batch_FP"]
        chunk["TN"] = batch_metric["batch_TN"]
        chunk["FN"] = batch_metric["batch_FN"]

class ConfidenceHitmap(ChunkProcessor):
    def __call__(self, chunk):
        chunk["batch_confidence_hitmap"] = tf.cast(chunk["batch_hitmap"], tf.float32)*chunk["batch_heatmap"][..., tf.newaxis]

class ComputeTopK(ChunkProcessor):
    def __init__(self, k):
        """ From a `confidence_hitmap` tensor where peaks are identified with non-zero pixels whose
            value correspnod to the peaks intensity, compute the `topk_indices` holding (x,y) positions
            and `topk_outputs` holding the intensity of the `k` highest peaks.
            Inputs:
                batch_confidence_hitmap - a (B,H,W,C,N) tensor where C is the number of keypoint types
                                          and N is the threshold dimension where only peaks above the
                                          corresponding threshold are reported.
            Outputs:
                topk_outputs - a (B,C,N,K) tensor where values along the K dimensions are sorted by
                               peak intensity.
                topk_indices - a (B,C,N,K,S) tensor where x coordinates are located in S=0 and y
                               coordinates are located in S=1.
        """
        self.k = np.max(k)
    def __call__(self, chunk):
        # Flatten hitmap to feed `top_k`
        _, H, W, C, N = [tf.shape(chunk["batch_confidence_hitmap"])[d] for d in range(5)]
        shape = [-1, C, N, H*W]
        flatten_hitmap = tf.reshape(tf.transpose(chunk["batch_confidence_hitmap"], perm=[0,3,4,1,2]), shape=shape)
        topk_values, topk_indices = tf.math.top_k(flatten_hitmap, k=self.k, sorted=True)

        chunk["topk_outputs"] = topk_values # B, C, K
        chunk["topk_indices"] = tf.stack(((topk_indices // W), (topk_indices % W)), -1) # B, C, K, D
        
class ComputeKeypointsTopKDetectionMetrics(ChunkProcessor):
    def __call__(self, chunk):
        assert len(chunk["batch_target"].get_shape()) == 3 or chunk["batch_target"].get_shape()[3] == 1, \
            "Only one keypoint type is allowed. If 'batch_target' is one_hot encoded, it needs to be compressed before."
        batch_target = tf.cast(chunk["batch_target"], tf.int32)
        batch_target = batch_target[..., 0] if len(batch_target.shape) == 4 else batch_target
        chunk["topk_targets"] = tf.gather_nd(batch_target, chunk["topk_indices"], batch_dims=1)

        chunk["P"] = tf.cast(tf.reduce_any(batch_target!=0, axis=[1,2]), tf.int32)
        chunk["N"] = 1-chunk["P"]
        chunk["topk_TP"] = tf.cast(tf.cast(tf.math.cumsum(chunk["topk_targets"], axis=-1), tf.bool), tf.int32)
        chunk["topk_FP"] = tf.cast(tf.cast(chunk["topk_outputs"], tf.bool), tf.int32)-chunk["topk_targets"]

