from dataclasses import dataclass
import typing

import numpy as np
import pandas

from experimentator import Callback, ExperimentMode, ConfusionMatrix
import sklearn.metrics

from tasks.detection import divide



@dataclass
class ComputeClassifactionMetrics(Callback):
    before = ["AuC", "GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    def on_cycle_begin(self, **_):
        self.acc = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    def on_batch_end(self, batch_output, batch_target, **_):
        target = batch_target # must be one hot encoded
        output = batch_output == np.max(batch_output, axis=1)[..., np.newaxis]
        assert target.shape == output.shape

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
        state["classification_metrics"] = pandas.DataFrame(np.vstack([data[name] for name in data]).T, columns=list(data.keys()))


@dataclass
class ExtractClassificationMetrics(Callback):
    before = ["GatherCycleMetrics"]
    after = ["ComputeClassifactionMetrics", "ComputeConfusionMatrix"]
    when = ExperimentMode.EVAL
    class_name: str
    class_index: int
    def on_cycle_end(self, state, **_):
        for metric in ['precision', 'recall']:
            state[f"{self.class_name}_{metric}"] = state['classification_metrics'][metric].iloc[self.class_index]


@dataclass
class ComputeConfusionMatrix(Callback):
    before = ["GatherCycleMetrics"]
    when = ExperimentMode.EVAL
    classes: typing.Tuple[list, tuple]
    def on_cycle_begin(self, **_):
        n = len(self.classes)
        self.cm = np.zeros((n, n))
    def on_batch_end(self, batch_output, batch_target, **_):
        self.cm += sklearn.metrics.confusion_matrix(
            np.argmax(batch_target, axis=1),
            np.argmax(batch_output, axis=1),
            labels=self.classes
        )
    def on_cycle_end(self, state, **_):
        state['confusion_matrix'] = ConfusionMatrix(self.cm)# / np.sum(self.cm)
