from dataclasses import dataclass
import typing

import numpy as np
import pandas

from experimentator import Callback, ExperimentMode, ConfusionMatrix
import sklearn.metrics

from tasks.detection import divide

@dataclass
class OneHotEncode(Callback):
    when = ExperimentMode.EVAL
    name: str
    num_classes: int
    def on_batch_end(self, state, **_): # state in R/W mode
        state[self.name] = np.squeeze(np.eye(self.num_classes)[state[self.name].reshape(-1)])

@dataclass
class ComputeClassifactionMetrics(Callback):
    after = ["OneHotEncode"]
    before = ["GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    logits_key: str = "batch_output"
    target_key: str = "batch_target"
    name: str = "classification"
    def on_cycle_begin(self, **_):
        self.acc = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    def on_batch_end(self, **state):
        output = state[self.logits_key] == np.max(state[self.logits_key], axis=1)[..., np.newaxis]
        target = state[self.target_key] # must be one hot encoded
        assert target.shape == output.shape, f"target shape {target.shape} != output shape {output.shape}"

        self.acc['TP'] += np.sum(  target   *   output  , axis=0) # sum over batch dimension
        self.acc['FP'] += np.sum((1-target) *   output  , axis=0)
        self.acc['FN'] += np.sum(  target   * (1-output), axis=0)
        self.acc['TN'] += np.sum((1-target) * (1-output), axis=0)
    def on_cycle_end(self, state, **_):
        FP = self.acc["FP"]
        TP = self.acc["TP"]
        FN = self.acc["FN"]
        TN = self.acc["TN"]
        data = {
            "FP rate": divide(FP, TN + FP),
            "TP rate": divide(TP, TP + FN),
            "precision": divide(TP, TP + FP),
            "recall": divide(TP, TP + FN),
        }
        state[f"{self.name}_metrics"] = pandas.DataFrame(np.vstack([data[name] for name in data]).T, columns=list(data.keys()))


@dataclass
class ExtractClassificationMetrics(Callback):
    before = ["GatherCycleMetrics"]
    after = ["ComputeClassifactionMetrics", "ComputeConfusionMatrix"]
    when = ExperimentMode.EVAL
    class_name: str
    class_index: int
    name: str = "classification"
    def on_cycle_end(self, state, **_):
        for metric in ['precision', 'recall']:
            state[f"{self.class_name}_{metric}"] = state[f'{self.name}_metrics'][metric].iloc[self.class_index]


@dataclass
class ComputeConfusionMatrix(Callback):
    after = ["OneHotEncode"]
    before = ["GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    classes: typing.Tuple[list, tuple]
    logits_key: str = "batch_output"
    target_key: str = "batch_target"
    def on_cycle_begin(self, **_):
        n = len(self.classes)
        self.cm = np.zeros((n, n))
    def on_batch_end(self, **state):
        self.cm += sklearn.metrics.confusion_matrix(
            np.argmax(state[self.target_key], axis=1),
            np.argmax(state[self.logits_key], axis=1),
            labels=self.classes
        )
    def on_cycle_end(self, state, **_):
        state['confusion_matrix'] = ConfusionMatrix(self.cm)# / np.sum(self.cm)
