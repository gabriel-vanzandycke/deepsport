from functools import cached_property

import tensorflow as tf

from experimentator import ChunkProcessor, ExperimentMode
from tf_layers import GammaColorAugmentation, AlexKendalMultiTaskLossWeighting


class CombineLosses(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __init__(self, names, weights):
        self.weights = weights
        self.names = names
    def __call__(self, chunk):
        losses = tf.stack([chunk[name]*w for name, w in zip(self.names, self.weights)])
        mask = tf.math.logical_not(tf.math.is_nan(losses))
        chunk["losses"] = losses
        chunk["loss"] = tf.reduce_sum(losses[mask])

class AlexKendallCombineLosses(CombineLosses):
    @cached_property
    def layer(self):
        return AlexKendalMultiTaskLossWeighting()
    def __call__(self, chunk):
        super().__call__(chunk)
        losses = chunk['losses']
        mask = tf.math.logical_and(
            tf.math.logical_not(tf.math.is_nan(losses)),
            tf.cast(losses, tf.bool)
        )
        losses = tf.where(mask, losses, 0)
        losses = self.layer(losses)
        chunk["loss"] = tf.reduce_sum(losses[mask])


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

class ExtractCenterFeatures(ChunkProcessor):
    def __init__(self, proportion, tensor_name='batch_logits'):
        self.tensor_name = tensor_name
        self.proportion = proportion
    def __call__(self, chunk):
        if self.proportion == 1:
            return
        _, H, W, _ = chunk[self.tensor_name].shape
        x_slice = slice(int((1-self.proportion)*W/2), int((1+self.proportion)*W/2))
        y_slice = slice(int((1-self.proportion)*H/2), int((1+self.proportion)*H/2))
        chunk[self.tensor_name] = chunk[self.tensor_name][:,y_slice,x_slice,:]

class LeNetHead(ChunkProcessor):
    def __init__(self, output_features, name='batch', input_tensor='batch_logits'):
        self.output_features = output_features
        self.input_tensor = input_tensor
        self.name = name
    @cached_property
    def model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(self.output_features),
        ], name=f"{self.name}_head")
    def __call__(self, chunk):
        chunk[f"{self.name}_logits"] = self.model(chunk[self.input_tensor])

class CropBlockDividable(ChunkProcessor):
    def __init__(self, tensor_names, block_size=16):
        self.tensor_names = tensor_names
        self.block_size = block_size
    def __call__(self, chunk):
        for name in chunk:
            if name in self.tensor_names:
                height, width = tf.shape(chunk[name])[1:3]
                w = width//self.block_size*self.block_size
                h = height//self.block_size*self.block_size
                chunk[name] = chunk[name][:,:h,0:w]

class HuberLoss(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __init__(self, y_true, y_pred, name="regression", delta=1.0):
        self.delta = delta # required to print config
        self.loss = tf.keras.losses.Huber(delta=delta, name='huber_loss')
        self.y_true = y_true
        self.y_pred = y_pred
        self.name = name
    def __call__(self, chunk):
        mask = tf.math.logical_not(tf.math.is_nan(chunk[self.y_true]))
        loss = self.loss(y_true=chunk[self.y_true][mask], y_pred=chunk[self.y_pred][mask])
        chunk[f"{self.name}_loss"] = loss # Huber loss does reduce_mean by default

class BinaryCrossEntropyLoss(ChunkProcessor):
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL
    def __init__(self, y_true, y_pred, from_logits=True, name='bce'):
        self.y_true = y_true
        self.y_pred = y_pred
        self.name = name
        self.from_logits = from_logits
    def __call__(self, chunk):
        # TODO: check if binary crossentropy fits the unconfident targets
        chunk[f"{self.name}_loss"] = tf.keras.losses.binary_crossentropy(y_true=chunk[self.y_true], y_pred=chunk[self.y_pred], from_logits=self.from_logits)
