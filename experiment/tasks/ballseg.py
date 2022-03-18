import tensorflow as tf
import numpy as np
from .common import ChunkProcessor


def gaussian_kernel(size: int, std: float):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(std))
    return kernel / np.sum(kernel)

class BallSegMetrics(ChunkProcessor):
    def __init__(self, top_k=3, threshold=0.5):
        self.gaussian_kernel = gaussian_kernel(9, 2.0)
        self.top_k = top_k
    def __call__(self, chunk):
        batch_size = tf.shape(chunk["batch_target"])[0]
        targets = chunk["batch_target"] # shape [b, h, w]
        predictions = chunk["batch_softmax"] # shape [b, h, w]
        # give more tolerence to target size

        kernel = tf.ones((5,5,1))
        targets = tf.nn.dilation2d(targets[..., tf.newaxis], filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), padding="SAME", data_format="NHWC")
        targets = targets[...,0]-1

        if k==1:
            max_pred_in_target = tf.reduce_max(tf.where(targets > 0.5, predictions, tf.zeros_like(targets)), axis=[1, 2])
            max_pred_out_target = tf.reduce_max(tf.where(targets < 0.5, predictions, tf.zeros_like(targets)), axis=[1, 2])
            accuracy = tf.reduce_sum(tf.cast(tf.greater(max_pred_in_target, max_pred_out_target), tf.int32))
            return accuracy

        chunk["BallSeg_TP"] = 
        # dilate and erode
        kernel = tf.ones((5,5,1))
        x = tf.nn.dilation2d(predictions[..., tf.newaxis], filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), padding="SAME", data_format="NHWC")
        x = tf.nn.erosion2d(x, filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), padding="SAME", data_format="NHWC")

        # smooth result
        kernel = self.gaussian_kernel[..., tf.newaxis, tf.newaxis]
        x = tf.nn.conv2d(x, filters=kernel, strides=(1,1,1,1), padding='SAME')/tf.reduce_max(x)

        # locate maximas
        pool = tf.nn.max_pool2d(x, strides=1, ksize=7, padding='SAME')
        maxima = tf.reshape(tf.where(pool-x < 0.00001, x, tf.zeros_like(x)), (batch_size, -1))
        chunk["maxima"] = maxima

        topk_values, topk_indices = tf.nn.top_k(maxima, k=self.top_k, sorted=True)
        d["topk_values"] = topk_values
        d["topk_indices"] = topk_indices

        targets = tf.reshape(tf.where(targets > 0.5, tf.ones_like(targets), tf.zeros_like(targets)), (batch_size, -1))
        d["thresholded_targets"] = targets
        ranks = tf.where(topk_values > threshold, tf.cast(tf.batch_gather(targets, topk_indices), tf.float32), tf.zeros_like(topk_values))
        d["ranks"] = ranks
        # only pay attention to the first candidate on the ball
        ranks = tf.cast(tf.logical_and(tf.equal(ranks,tf.cumsum(ranks,axis=1)),tf.equal(ranks,tf.ones_like(ranks))), tf.int32)
        d["ranks2"] = ranks
