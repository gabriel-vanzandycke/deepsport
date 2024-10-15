import pytest

from experimentator import build_experiment


@pytest.mark.skip(reason="Dummy")
def test_build_empty_experiment():
    assert build_experiment("""
import experimentator
import experimentator.tf2_experiment
experiment_type=[experimentator.BaseExperiment, experimentator.tf2_experiment.TensorflowExperiment]
""", load_weights=False)

def test_train_dummy_experiment():
    exp = build_experiment("""
import experimentator
import experimentator.tf2_experiment
import mlworkflow
import tensorflow as tf
experiment_type=[experimentator.BaseExperiment, experimentator.tf2_experiment.TensorflowExperiment]
subsets = [experimentator.Subset(
                            "subset_name",
                            experimentator.Stage.EVAL,
                            mlworkflow.DictDataset({
                                "k1":{"a":1, "b":2},
                                "k2":{"a":2, "b":4}
                           }))]
batch_size = 1
chunk_processors = [lambda chunk: chunk.update({"loss": chunk["batch_a"]})]
optimizer = tf.keras.optimizers.Adam(learning_rate=1)
""", load_weights=False)
    exp.train(1)

