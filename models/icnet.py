from functools import cached_property

import tensorflow as tf

from experimentator import ChunkProcessor
from icnet_tf2 import ICNetModel

class ICNetBackbone(ChunkProcessor):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    @cached_property
    def network(self):
        return ICNetModel(*self.args, **self.kwargs)
    def __call__(self, chunk):
        """ Outputs 3 tensors
        """
        #  B x H/8 x W/8 x 256  ,      B x H/4 x W/4 x 128  ,      B x H/2 x W/2 x 128
        chunk["conv5_4_interp"] , chunk["sub24_sum_interp"] , chunk["sub12_sum_interp"] = self.network(chunk["batch_input"])

class ICNetHead(ChunkProcessor):
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
    def __call__(self, chunk):
        num_classes = self.num_classes or chunk["batch_target"].get_shape().as_list()[-1]
        chunk["sub4_out"] = tf.keras.layers.Conv2D(num_classes, 1, 1, activation=None, name='sub4_out')(chunk["conv5_4_interp"])
        chunk["sub24_out"] = tf.keras.layers.Conv2D(num_classes, 1, 1, activation=None, name='sub24_out')(chunk["sub24_sum_interp"])
        chunk["sub124_out"] = tf.keras.layers.Conv2D(num_classes, 1, 1, activation=None, name='conv6_cls')(chunk["sub12_sum_interp"])

        shape = chunk["batch_input"].shape if chunk["batch_input"].get_shape().as_list()[1] is not None else tf.shape(chunk["batch_input"])
        chunk["batch_logits"] = tf.image.resize(chunk["sub124_out"], method=tf.image.ResizeMethod.BILINEAR, size=(shape[1],shape[2]))

