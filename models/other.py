from functools import cached_property

import tensorflow as tf

from experimentator import ChunkProcessor, ExperimentMode
from tf_layers import GammaColorAugmentation


class GammaAugmentation(ChunkProcessor):
    mode = ExperimentMode.TRAIN
    def __init__(self, tensor_name):
        """ Performs Gama Data-Augmentation on `tensor_name`.
            One single name is allowed because the same random augmentation should be applied to dependant items.
        """
        # create random gamma for RGB with R in [0.96, 1.04], G in [0.98, 1.02] and B in [0.96, 1.04]
        self.layer = GammaColorAugmentation([0.02, 0.01, 0.02])
        self.tensor_name = tensor_name
    def color_augmentation(self, chunk):
        assert chunk[self.tensor_name].shape[3] == 3, "GamaAugmentation must be done on RGB images"
        chunk[self.tensor_name] = self.layer(chunk[self.tensor_name])
    def __call__(self, chunk):
        self.color_augmentation(chunk)


class LeNetHead(ChunkProcessor):
    def __init__(self, output_features):
        self.output_features = output_features
    @cached_property
    def model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(self.output_features),
        ])
    def __call__(self, chunk):
        chunk["batch_logits"] = self.model(chunk["batch_logits"])
