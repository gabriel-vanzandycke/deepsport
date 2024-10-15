import abc
import os
import re
import glob
import itertools

from aleatorpy import pseudo_random, method
import numpy as np
import cv2
import tensorflow as tf

from functools import cached_property
from experimentator import Stage
from experimentator.tf2_experiment import TensorflowExperiment
from experimentator.callbacked_experiment import Callback
from dataset_utilities.transforms import Transform
from deepsport_utilities.utils import setdefaultattr
from dataset_utilities.court import Court
from dataset_utilities.calib import Point3D
from experimentator import ChunkProcessor

from .ball import BallSegmentationExperiment

def duplicator(f):
    def duplicated_keys():
        keys = f()
        return list(itertools.chain.from_iterable(zip(keys, keys))) # duplicates each key
    return duplicated_keys

class ContrastiveLearningExperiment(TensorflowExperiment):
    @cached_property
    def subsets(self):
        subsets = super().subsets
        for subset in subsets:
            subset.shuffled_keys = duplicator(subset.shuffled_keys)
            subset.dataset.query_item.loop = (subset.dataset.query_item.loop or 1)*2
        return subsets
    @cached_property
    def outputs(self):
        return {"batch_output": self.chunk["embeddings"]}
    @cached_property
    def metrics(self):
        return {**super().metrics, "accuracy": self.chunk["accuracy"]}

class RandomBallByPairs(Transform):
    def __init__(self, debug=False):
        self.debug = debug
    @method
    @pseudo_random(input_dependent=True, repeat=2)
    def random_ball(self, *_): # necessary unused argument to control randomness
        x = np.random.beta(2, 2)
        y = np.random.beta(2, 2)
        return x, y
    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError()

class SetRandomBallPositionViewTransform(RandomBallByPairs):
    def __call__(self, view_key, view):
        court = setdefaultattr(view, "court", Court(getattr(view, "rule_type", "FIBA")))
        for idx, ball in enumerate([a for a in view.annotations if a.type == "ball" and a.camera == view_key.camera]):
            top_edge = list(court.visible_edges(view.calib))[0]
            start = top_edge[0][0][0]
            stop = top_edge[1][0][0]
            x, y = self.random_ball(view_key, idx)
            x = x*(stop-start)+start
            y = y*court.h/2+court.h/4
            z = -100
            ball.center = Point3D(x,y,z) # Setting random ball position instead of real ball position !
            ball.visible = False         # Setting ball visibility to False
            if self.debug:
                cv2.circle(view.image, view.calib.project_3D_to_2D(ball.center).to_int_tuple(), 10, (0,255,0), 2)
        return view

class SetRandomBallPositionItemTransform(RandomBallByPairs):
    def __call__(self, key, item):
        h, w, _ = item.image.shape
        x, y = self.random_ball(key)
        item.annotation['x'] = x*w - item.annotation['width']/2
        item.annotation['y'] = y*h - item.annotation['height']/2

        if self.debug:
            cv2.circle(item.image, (int(x*w), int(y*h)), 10, (0,255,0), 3, -1)
        return item


class SumAccuracyCallback(Callback):
    before = ["GatherCycleMetrics"]
    when = Stage.EVAL
    def on_cycle_begin(self, **_):
        self.TP = 0
        self.FP = 0
    def on_batch_end(self, accuracy, **_):
        TP = np.sum(accuracy)
        FP = len(accuracy) - TP
        self.TP += TP
        self.FP += FP
    def on_cycle_end(self, state, **_):
        state["precision"] = self.TP/(self.TP+self.FP) if (self.TP+self.FP) > 0 else 0


ICNET_WEIGHTS_SUFFIX = "_icnet_weights"

class BallDetectionFromContrastiveLearning(BallSegmentationExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        experiment_location = self.cfg["weights_folder"]
        if experiment_location is not None:
            filenames = [f for f in glob.glob(os.path.join(experiment_location,"*")) if f"{ICNET_WEIGHTS_SUFFIX}.index" in f]
            # match epoch number in the beginning of the filename to take the highest
            filename = max([(int(re.match(r"^(\d+)_.*", os.path.basename(f)).group(1)),f) for f in filenames], key=lambda kv: kv[0])[1]
            icnet_model = [layer for layer in self.train_model.layers if "ic_net_model" in layer.name][-1]
            print(f"Loading ICNet weights from '{filename}'.")
            icnet_model.load_weights(filename.replace(".index",""))


class SaveICNetWeights(Callback):
    max_precision = None
    after = ["AverageMetrics"]
    validation_cycle_name: str = "validation"
    def init(self, exp):
        self.exp = exp
        self.folder = exp.folder
    @cached_property
    def icnet_model(self):
        return [layer for layer in self.exp.train_model.layers if "ic_net_model" in layer.name][0]
    def on_cycle_end(self, cycle_name, precision, epoch, **_):
        if cycle_name == self.validation_cycle_name:
            if self.max_precision is None or precision > self.max_precision:
                self.max_precision = precision
                self.icnet_model.save_weights(os.path.join(self.folder, f"{epoch}{ICNET_WEIGHTS_SUFFIX}"))

class MLP(ChunkProcessor):
    @cached_property
    def mpl(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation=tf.keras.activations.relu, use_bias=False),
            tf.keras.layers.Dense(2048, use_bias=False)
        ])
    def __call__(self, chunk):
        chunk["embeddings"] = self.mpl(chunk["batch_logits"])
