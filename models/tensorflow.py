import tensorflow as tf

from experimentator.tf2_chunk_processors import ChunkProcessor



class TensorflowBackbone(ChunkProcessor):
    def __init__(self, model_name, *args, include_top=False, **kwargs):
        self.model_name = model_name
        self.include_top = include_top
        self.args = args
        self.kwargs = kwargs
    def __call__(self, chunk):
        if getattr(self, "network", None) is None:
            input_shape = chunk["batch_input"].get_shape()[1:4]
            model_str = f"tf.keras.applications.{self.model_name}"
            print(f"Initializing '{model_str}' with input", chunk["batch_input"])
            self.network = eval(model_str)(input_shape=input_shape, include_top=self.include_top, *self.args, **self.kwargs) # pylint: disable=eval-used
            #print(self.network.summary())
        chunk["batch_logits"] = self.network(chunk["batch_input"], training=True)

