import tensorflow as tf
import numpy as np

from experimentator.tf2_chunk_processors import ChunkProcessor



class TensorflowBackbone(ChunkProcessor):
    def __init__(self, model_name, *args, include_top=False, **kwargs):
        self.model_name = model_name
        self.include_top = include_top
        self.args = args
        self.kwargs = kwargs
    def init_network(self, input_shape):
        model_str = f"tf.keras.applications.{self.model_name}"
        print(f"Initializing '{model_str}' with {input_shape} input")
        self.network = eval(model_str)(input_shape=input_shape, include_top=self.include_top, *self.args, **self.kwargs) # pylint: disable=eval-used
        #print(self.network.summary())

    def __call__(self, chunk):
        if getattr(self, "network", None) is None:
            self.init_network(chunk["batch_input"].get_shape()[1:4])
        chunk["batch_logits"] = self.network(chunk["batch_input"], training=True)


class SixChannelsTensorflowBackbone(TensorflowBackbone):
    def init_network(self, input_shape):
        H, W, C = input_shape
        super().init_network((H, W, 3))
        if C != 3:
            assert C == 6, f"Expected 3 or 6 channels, got {C}"
            # Retrieve first layer specifications
            layer = self.network.layers[1]
            weights, biaises = layer.weights
            config = layer.get_config()

            # Duplicate first layer channels
            weights = tf.concat([weights]*2, 2)

            # Recreate first layer
            layer = layer.__class__.from_config(config)
            layer.build(input_shape[1:])
            layer.set_weights((np.array(weights), np.array(biaises)))

            # Recreate model
            self.network = tf.keras.Sequential([
                layer,
                *self.network.layers[2:]
            ], self.network.name)
