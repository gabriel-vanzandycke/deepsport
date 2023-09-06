import tensorflow as tf
import numpy as np

from experimentator.tf2_chunk_processors import ChunkProcessor



class TensorflowBackbone(ChunkProcessor):
    def __init__(self, model_name, *args, include_top=False, **kwargs):
        self.model_name = model_name
        self.include_top = include_top
        self.args = args
        self.kwargs = kwargs
    def init_model(self, input_shape):
        model_str = f"tf.keras.applications.{self.model_name}"
        print(f"Initializing '{model_str}' with {input_shape} input")
        self.model = eval(model_str)(input_shape=input_shape, include_top=self.include_top, *self.args, **self.kwargs) # pylint: disable=eval-used
        #print(self.model.summary())
    def __call__(self, chunk):
        if getattr(self, "model", None) is None:
            self.init_model(chunk["batch_input"].get_shape()[1:4])
        chunk["batch_logits"] = self.model(chunk["batch_input"], training=True)


class SixChannelsTensorflowBackbone(TensorflowBackbone):
    def init_model(self, input_shape):
        H, W, C = input_shape
        super().init_model((H, W, 3))
        if C != 3:
            assert C == 6, f"Expected 3 or 6 channels, got {C}"
            print(f"Re-initializing model with {C} input channels")

            # Retrieve first layer specifications
            layer = self.model.layers[1]
            weights, biaises = layer.weights
            config = layer.get_config()

            # Dupldicate first layer channels
            weights = tf.concat([weights]*2, 2)

            # Recreate first layer
            layer = layer.__class__.from_config(config)
            layer.build(input_shape[1:])
            layer.set_weights((np.array(weights), np.array(biaises)))

            # Recreate model
            self.model = tf.keras.Sequential([
                layer,
                *self.model.layers[2:]
            ], self.model.name)

class SkipConnectionCroppedInputsModel(tf.keras.Model):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def build(self, input_shape):
        C = input_shape[-1]
        self._layers = []
        first_weights = None
        for layer in self.model.layers[1:]:
            self._layers.append(layer.__class__.from_config(layer.get_config()))
            if isinstance(self._layers[-1], tf.keras.layers.Conv2D):
                weights, biaises = layer.weights
                _, h, w, c = layer.input_shape
                if first_weights is None:
                    first_weights = weights
                else:
                    weights = tf.concat([weights, weights[:,:,0:C]], axis=2)
                    c = c+C

                self._layers[-1].build((h, w, c))
                self._layers[-1].set_weights((np.array(weights), np.array(biaises)))

    def call(self, input_tensor):
        _, H, W, _ = input_tensor.shape
        x = input_tensor
        for i, layer in enumerate(self._layers):
            if isinstance(layer, tf.keras.layers.Conv2D) and i != 0: # Skip first layer as well
                _, h, w, c = x.shape
                x_slice = slice((W-w)//2, (W-w)//2+w, None)
                y_slice = slice((H-h)//2, (H-h)//2+h, None)
                x = layer(tf.concat([x, input_tensor[:, y_slice, x_slice]], axis=-1))
            else:
                x = layer(x)
        return x


class SkipConnectionCroppedInputsModelSixCannels(SixChannelsTensorflowBackbone):
    def init_model(self, input_shape):
        return SkipConnectionCroppedInputsModel(super().init_model(input_shape))
