
import abc
from functools import cached_property
import os
import re
import tempfile
import time

import numpy as np
import tensorflow as tf

from calib3d import Point3D, Point2D
from experimentator import ChunkProcessor, ExperimentMode
# allow import from here to simplify config file
from experimentator.tf2_chunk_processors import CastFloat, Normalize, DatasetStandardize, StopGradients # pylint: disable=unused-import

from deepsport_utilities.transforms import Transform
from deepsport_utilities.court import BALL_DIAMETER

from .detection import HeatmapDetectionExperiment

BALL_SIGMA_FACTOR = 0.2
DEFAULT_NEIGHBOURS = 4

class PIFExperiment(HeatmapDetectionExperiment):
    batch_inputs_names = ["batch_target", "batch_confidence_target", "batch_scales_target", "batch_xs_target", "batch_ys_target",
        "batch_input_image", "batch_input_image2"]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.cfg.get("ignore_eval_warning", False):
            assert all([i not in self.cfg.get("eval_epochs", [i]) for i in range(10)]), \
                "Current implementation doesn't support evaluation on epochs < 10." \
                "Decoding algorithms loops for an eternity when confidence has not been trained."
    @cached_property
    def metrics(self):
        metrics = super().metrics
        for name in ["confidence_loss", "localization_loss", "scale_loss"]:
            metrics[name] = self.chunk[name]
        return metrics

class CreatePIFTarget(abc.ABC, Transform):
    def __init__(self, stride, neighbours=DEFAULT_NEIGHBOURS, scale_factor=BALL_SIGMA_FACTOR):
        """ Creates the PIF (part-intensity-field from PifPaf (S.Kreiss et al., PifPaf, CVPR 2019)) target given a set
            of keypoints (e.g. ball, shoulder, knee, ...)
        """
        self.stride = stride # backbone reduction factor (ResNet50 is 32)
        self.neighbours = neighbours
        self.scale_factor = scale_factor # ball size affected by gaussians sigma

    @abc.abstractmethod
    def get_keypoints(self, key, item):
        raise NotImplementedError()

    def __call__(self, key, item):
        input_shape, keypoints, scales = self.get_keypoints(key, item)
        input_width, input_height = input_shape

        output_width = input_width//self.stride
        output_height = input_height//self.stride
        keypoints = keypoints/self.stride
        scales = scales*self.scale_factor/self.stride

        if keypoints.size == 0:
            return {name: np.zeros((output_height, output_width), dtype=np.float32) for name in ["scales_target", "confidence_target", "xs_target", "ys_target"]}

        shape = (output_height, output_width, keypoints.shape[1])

        xs, ys = np.meshgrid(np.arange(output_width), np.arange(output_height))
        xs = keypoints.x - xs[...,np.newaxis]
        ys = keypoints.y - ys[...,np.newaxis]

        scales = np.ones(shape)*np.array(scales)[np.newaxis, np.newaxis]
        indices = np.argmin(xs**2+ys**2, axis=-1)

        # use closest keypoint for each cell (keypoints are in the last dimension)
        xs = np.take_along_axis(xs, indices[...,np.newaxis], axis=-1)[...,0]
        ys = np.take_along_axis(ys, indices[...,np.newaxis], axis=-1)[...,0]
        scales = np.take_along_axis(scales, indices[...,np.newaxis], axis=-1)[...,0]
        confidence = np.where(np.logical_and(np.abs(xs)<self.neighbours/2, np.abs(ys)<self.neighbours/2), 1.0, 0.0)
        xs[confidence==0] = 0
        ys[confidence==0] = 0
        scales[confidence==0] = 0
        return {
            "scales_target": scales.astype(np.float32),
            "confidence_target": confidence.astype(np.float32),
            "xs_target": xs.astype(np.float32),
            "ys_target": ys.astype(np.float32)
        }

class AddPIFBallTargetSynergyFactory(CreatePIFTarget):
    def get_keypoints(self, key, item):
        height, width, _ = item.image.shape
        x = item.annotation['x'] + item.annotation['width']/2
        y = item.annotation['y'] + item.annotation['height']/2
        r = (item.annotation['width'] + item.annotation['height'])/4
        return (width, height), Point2D(x,y), np.array([r])

class AddPIFBallTargetViewFactory(CreatePIFTarget):
    def get_keypoints(self, key, item):
        view = item
        calib = view.calib
        balls_3D = Point3D([a.center for a in view.annotations if a.type == "ball" and calib.projects_in(a.center)])
        keypoints = calib.project_3D_to_2D(balls_3D)
        scales = calib.compute_length2D(BALL_DIAMETER/2, balls_3D)*self.scale_factor
        return (calib.width, calib.height), keypoints, scales

class CustomResnet50(ChunkProcessor):
    """ This gives me exactly the same number of trainable parameters as the pytorch version
    """
    def __init__(self, weight_decay=1e-5, weights="tensorflow"):
        self.weight_decay = weight_decay
        self.weights = weights
        base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(None, None, 3))
        if weights in ["pytorch", "torch"]:
            self.set_pytorch_weights(base_model)
        elif weights not in ["tensorflow", "tf"]:
            raise NotImplementedError

        # Remove first 7x7 pooling layer
        head_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[4].output, name="ResNet-head")
        tail_input = tf.keras.Input(tensor=base_model.layers[7].input)
        tail_model = tf.keras.Model(inputs=tail_input, outputs=base_model.output, name="ResNet-tail")

        # Adding weight decay regularization
        # Setting different batch norm parameters
        # Removing bias from conv layers
        regularizer = tf.keras.regularizers.l2(weight_decay)
        for layer in head_model.layers + tail_model.layers:
            if hasattr(layer, "kernel_regularizer"):
                setattr(layer, "kernel_regularizer", regularizer)
            if hasattr(layer, "bias_regularizer"):
                setattr(layer, "bias_regularizer", regularizer)
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = 0.99 # /!\ pytorch_momentum = (1 - tensorflow_momentum)
                layer.epsilon = 1e-4
            if isinstance(layer, tf.keras.layers.Conv2D):
                delattr(layer, "bias")
                layer.use_bias = False

        model = tf.keras.Sequential([head_model, tail_model], name="Custon-ResNet")
        self.model = self.reload_model(model)

    @staticmethod
    def reload_model(model):
        # Save the weights and config
        model_json = model.to_json()
        state = np.random.get_state()
        np.random.seed(time.clock_gettime_ns(time.CLOCK_MONOTONIC)%1000)
        tmp_weights_path = os.path.join(tempfile.gettempdir(), f"tmp_weights_{np.random.randint(0,10e5):05d}.h5")
        np.random.set_state(state)
        model.save_weights(tmp_weights_path)

        # Load model from config and weights
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(tmp_weights_path, by_name=True)
        return model

    @staticmethod
    def set_pytorch_weights(resnet_tf):
        import torchvision
        def match_pytorch_name(layer_name):
            m = re.match(r"^conv1_(?P<type>\w*)$", layer_name)
            if m:
                return "{type}1".format(**m.groupdict())
            m = re.match(r"^conv(?P<layer>\d*)_block(?P<block>\d*)_(?P<index>\d*)_(?P<type>\w*)$", layer_name)
            if m:
                match = m.groupdict()
                match['layer'] = int(match['layer']) - 1
                match['block'] = int(match['block']) - 1
                if match['index'] == '0':
                    if match['type'] == 'conv':
                        match['type'] = 0
                    elif match['type'] == 'bn':
                        match['type'] = 1
                    else:
                        raise ValueError('Unexpected type: {}'.format(match['type']))
                    return "layer{layer}.{block}.downsample.{type}".format(**match)
                return "layer{layer}.{block}.{type}{index}".format(**match)
        resnet_torch = torchvision.models.resnet50(pretrained=True)
        resnet_torch_state_dict = resnet_torch.state_dict()

        for tf_layer in resnet_tf.layers:
            torch_name = match_pytorch_name(tf_layer.name)

            # address only batch-normalization and convolution layers
            if re.match(r".*conv$", tf_layer.name):
                weights = resnet_torch_state_dict[torch_name+'.weight'].numpy()
                weights_list = [weights.transpose((2, 3, 1, 0))]
                bias_name = torch_name+'.bias'
                if bias_name in resnet_torch_state_dict:
                    bias = resnet_torch_state_dict[bias_name].numpy()
                    weights_list.append(bias)
                else:
                    weights_list.append(np.zeros_like(tf_layer.weights[1]))
                tf_layer.set_weights(weights_list)
            elif re.match(r".*bn$", tf_layer.name):
                gamma = resnet_torch_state_dict[torch_name+'.weight'].numpy()
                beta = resnet_torch_state_dict[torch_name+'.bias'].numpy()
                mean = resnet_torch_state_dict[torch_name+'.running_mean'].numpy()
                var = resnet_torch_state_dict[torch_name+'.running_var'].numpy()
                bn_list = [gamma, beta, mean, var]
                tf_layer.set_weights(bn_list)

    def __call__(self, chunk):
        chunk["batch_logits"] = self.model(chunk["batch_input"])

class CompositeFieldFused(ChunkProcessor):
    def __init__(self, dropout=0.0, depth_to_space=2):
        self.dropout = dropout
        self.depth_to_space = depth_to_space
        self.model = tf.keras.Sequential([
            tf.keras.layers.SpatialDropout2D(dropout),
            tf.keras.layers.Conv2D(filters=5*depth_to_space*depth_to_space, kernel_size=1),
            tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, depth_to_space))
        ])
    def __call__(self, chunk):
        logits = self.model(chunk["batch_logits"])

        chunk["confidence_logits"] = logits[...,0]
        chunk["xs"]                = logits[...,1]
        chunk["ys"]                = logits[...,2]
        chunk["logb"]              = logits[...,3]
        chunk["scales_logits"]     = logits[...,4]

        chunk["confidence"] = tf.sigmoid(chunk["confidence_logits"])
        chunk["scales"]     = tf.exp(chunk["scales_logits"])

class CompositeLoss(ChunkProcessor):
    def __init__(self, lambdas=None):
        self.lambdas = lambdas or [1,1,1]
        self.confidence_loss = ConfidenceLoss()
        self.localization_loss = LocalizationLoss(weight=0.1)
        self.scale_loss = ScaleLoss()
    def __call__(self, chunk):
        target_confidence = chunk["batch_confidence_target"]
        chunk["confidence_loss"] = self.confidence_loss(batch_logits=chunk["confidence_logits"], batch_target=target_confidence)
        chunk["localization_loss"] = self.localization_loss(chunk["xs"], chunk["ys"], chunk["logb"], chunk["batch_xs_target"], chunk["batch_ys_target"], mask=target_confidence)
        chunk["scale_loss"] = self.scale_loss(output_scales=chunk["scales_logits"], target_scales=chunk["batch_scales_target"], mask=target_confidence)
        chunk["loss"] = self.lambdas[0]*chunk["confidence_loss"] + self.lambdas[1]*chunk["localization_loss"] + self.lambdas[2]*chunk["scale_loss"]

class ConfidenceLoss(ChunkProcessor):
    r"""
        $$ w = \right\{
            \begin{align}
                (1.0 + e^{-\ell})^\gamma  && \text{background}\\
                (1.0 + e^{\ell})^\gamma   && \text{foreground}
            \end{align}
            \left.
            \quad
            \mathcal{L}_c = w \cdot \left[ t \cdot \log \ell + (1 - t) \cdot \log (1 - \ell) \right]
        $$
    """
    def __init__(self, background_weight=1.0, focal_gamma=1.0, clamp_loss=True, divison_factor=1000):
        self.background_weight = background_weight
        self.focal_gamma = focal_gamma
        self.clamp_loss = [0.02, 5.0] if clamp_loss is True else clamp_loss
        self.division_factor = divison_factor

    def __call__(self, batch_logits, batch_target):
        weight_map = 1 # will be broadcasted if unused
        loss_map = tf.keras.losses.binary_crossentropy(y_true=batch_target[...,tf.newaxis], y_pred=batch_logits[...,tf.newaxis], from_logits=True)
        if self.clamp_loss:
            loss_map = tf.clip_by_value(loss_map, self.clamp_loss[0], self.clamp_loss[1])

        if self.focal_gamma != 0.0: # will use focal loss with gamma=focal_gamma
            proba_map = tf.exp(-loss_map) # loss := - log(proba) => proba = exp(-loss)
            weight_map = tf.pow(1.0 - proba_map, self.focal_gamma)
            loss_map = loss_map * weight_map

        if self.background_weight != 1.0:
            weight_map = tf.ones_like(batch_target)
            weight_map[batch_target == 0] = self.background_weight
            loss_map = loss_map * weight_map

        loss = tf.reduce_sum(loss_map, axis=[1,2])
        loss = tf.reduce_mean(loss, axis=0)/self.division_factor
        return loss

class LocalizationLoss():
    def __init__(self, weight=None, norm_low_clip=None, division_factor=100.0, logb_constraint="tanh"):
        self.weight = weight
        self.norm_low_clip = norm_low_clip
        self.division_factor = division_factor
        self.logb_constraint = logb_constraint

    def laplace_loss(self, x1, x2, logb, t1, t2):
        """ Loss based on Laplace Distribution.
            Loss for a single two-dimensional vector (x1, x2) with radial
            spread b and true (t1, t2) vector.
        """
        # left derivative of sqrt at zero is not defined, so prefer torch.norm():
        # https://github.com/pytorch/pytorch/issues/2421
        # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
        norm = tf.norm(tf.stack((x1, x2)) - tf.stack((t1, t2)), axis=0)
        if self.norm_low_clip is not None:
            norm = tf.clip_by_value(norm, self.norm_low_clip, 5.0)

        # Constrain range of logb
        # low range constraint: prevent strong confidence when overfitting
        # high range constraint: force some data dependence
        if self.logb_constraint == "tanh":
            logb = 3.0 * tf.tanh(logb/3.0)
        elif self.logb_constraint == "clamp":
            logb = tf.where(logb < -3.0, -3.0, logb)
        else:
            raise NotImplementedError()

        # loss = logb + (norm + 0.1) * tf.exp(-logb)    # Note: previous version of openpifpaf code
        loss = logb + 0.694 + norm * tf.exp(-logb)      # Note:

        if self.weight is not None:
            loss = loss * self.weight
        return loss

    def __call__(self, x1, x2, logb, t1, t2, mask):
        indices = tf.where(mask != 0)

        x1 = tf.gather_nd(x1, indices)
        x2 = tf.gather_nd(x2, indices)
        logb = tf.gather_nd(logb, indices)
        t1 = tf.gather_nd(t1, indices)
        t2 = tf.gather_nd(t2, indices)

        loss = self.laplace_loss(x1, x2, logb, t1, t2)

        batch_size = tf.cast(tf.shape(mask)[0], tf.float32)
        loss = tf.reduce_sum(loss)/batch_size
        return loss/self.division_factor

class ScaleLoss():
    def __init__(self, clip_by_value=None, division_factor=100.0):
        self.division_factor = division_factor
        self.clip_by_value = clip_by_value # [0.0, 5.0] in some version of PIFPAF code

    def __call__(self, output_scales, target_scales, mask):
        indices = tf.where(mask != 0)

        output_scales = tf.gather_nd(output_scales, indices)
        target_scales = tf.gather_nd(target_scales, indices)

        loss = tf.abs(output_scales - tf.math.log(target_scales))
        if self.clip_by_value is not None:
            loss = tf.clip_by_value(loss, self.clip_by_value[0], self.clip_by_value[1])

        batch_size = tf.cast(tf.shape(mask)[0], tf.float32)
        loss = tf.reduce_sum(loss)/batch_size
        return loss/self.division_factor

class DecodePif(ChunkProcessor):
    mode = ExperimentMode.EVAL
    def __init__(self, stride, neighbours=DEFAULT_NEIGHBOURS, max_value=1.0, **kwargs):
        self.stride = stride
        self.max_value = max_value
        self.neighbours = neighbours
        self.kwargs = kwargs
        self.layer = DecodePIFLayer(stride=stride, **self.kwargs)

    def __call__(self, chunk):
        height, width = chunk["confidence"].get_shape().as_list()[1:3]
        posx, posy = tf.meshgrid(tf.range(0, width, dtype=tf.float32), tf.range(0, height, dtype=tf.float32))
        mean = tf.stack((posx[tf.newaxis]+chunk["xs"], posy[tf.newaxis]+chunk["ys"]), axis=-1)*self.stride
        variance  = tf.maximum(1.0, chunk["scales"]*self.stride/2)**2
        confidence = chunk["confidence"]

        chunk["batch_heatmap"] = tf.math.minimum(self.max_value, self.layer(mean, variance, confidence)/(self.neighbours**2))
        print(chunk['batch_heatmap'].shape)

class DecodePIFLayer(tf.keras.layers.Layer):
    def __init__(self, stride, *args, min_confidence=0.1, min_variance=None, impl='__dynamic_loop_over_cells_with_max_implementation', **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = stride
        self.min_confidence = min_confidence
        self.min_variance = min_variance
        self.call = getattr(self, impl.replace("__", f"_{self.__class__.__name__}__"))

    def build(self, input_shape):
        _, self.height, self.width, _ = input_shape
        self.positions = tf.stack(tf.meshgrid(
            tf.range(0,self.width*self.stride, dtype=tf.float32),
            tf.range(0,self.height*self.stride, dtype=tf.float32)
        ), axis=-1)

    def __full_tensor_implementaiton(self, mean, variance, confidence, **_):
        raise NotImplementedError("This implementation requires to create a tensor far too large.")
        return tf.reduce_sum(
            tf.exp(
                -tf.reduce_sum(
                    (self.positions[tf.newaxis,:,:,tf.newaxis,:] - tf.reshape(mean, [-1, 1, 1, self.width*self.height, 2]))**2,
                    axis=-1
                )/(2*tf.reshape(variance, [-1, 1, 1, self.width*self.height]))
            )*tf.reshape(confidence, [-1, 1, 1, self.width*self.height]),
            axis=-1
        )

    def __dynamic_loop_over_cells_with_max_implementation(self, mean, variance, confidence, **_):
        # The simpler (and actually faster) implementation `acc = tf.zeros_like(self.positions)[tf.newaxis,:,:,0]`
        # to construct `acc` cannot be used because of the while loop...
        batch_size = tf.shape(confidence)[0]
        height, width, _ = self.positions.get_shape().as_list()
        acc = tf.zeros((batch_size, height, width))

        # Identify relevent cells
        indices = tf.where(confidence > self.min_confidence)
        if self.min_variance is not None:
            raise NotImplementedError
            #indices = tf.sets.intersection(indices, tf.where(variance>self.min_variance))
        mean = tf.gather_nd(mean, indices)
        variance = tf.gather_nd(variance, indices)
        confidence = tf.gather_nd(confidence, indices)

        # Accumulate over cells
        i = tf.constant(0)
        def cond(i, acc): # pylint: disable=unused-argument
            return i < len(confidence)
        def body(i, acc):
            x = tf.exp(
                -tf.reduce_sum(
                    (self.positions - mean[i])**2,
                    axis=-1
                )/(2*variance[i])
            )*confidence[i]
            acc = tf.tensor_scatter_nd_add(acc, [[indices[i,0]]], updates=x[tf.newaxis])
            return i+1, acc
        i, acc = tf.while_loop(cond, body, [i, acc])
        return acc

    def __dynamic_loop_over_batch_implementation(self, mean, variance, confidence, **_):
        raise NotImplementedError("Re-implemented needed by sampling mean, variance and confidence based on the confidence threshold")
        batch_size = tf.shape(mean)[0]
        index = tf.constant(0)
        acc = tf.zeros_like(self.positions)[tf.newaxis,:,:,0]
        def cond(index, acc): # pylint: disable=unused-argument
            return index < batch_size
        def body(index, acc):
            b = index
            acc[b] = tf.reduce_sum(
                tf.exp(
                    -tf.reduce_sum(
                        (self.positions[:,:,tf.newaxis,:] - tf.reshape(mean[b], [1, 1, self.width*self.height, 2]))**2,
                        axis=-1
                    )/(2*tf.reshape(variance[b], [1, 1, self.width*self.height]))
                )*tf.reshape(confidence[b], [1, 1, self.width*self.height]),
                axis=-1
            )
            return index, acc
        index, acc = tf.while_loop(cond, body, [index, acc])
        return acc

    def __dynamic_loop_over_batch_with_max_implementation(self, mean, variance, confidence, **_):
        raise NotImplementedError("Implementation is probably wrong... check required")
        # The simpler (and actually faster) implementation `acc = tf.zeros_like(self.positions)[tf.newaxis,:,:,0]`
        # cannot be used because of the while loop
        batch_size = tf.shape(mean)[0]
        height, width, _ = self.positions.get_shape().as_list()
        acc = tf.zeros((batch_size, height, width))

        # Identify relevent cells
        indices = tf.where(confidence>self.min_confidence)
        if self.min_variance is not None:
            raise NotImplementedError
            #indices = tf.sets.intersection(indices, tf.where(variance>self.min_variance))
        mean = tf.gather_nd(mean, indices)
        variance = tf.gather_nd(variance, indices)
        confidence = tf.gather_nd(confidence, indices)

        # Accumulate over batches
        i = tf.constant(0)
        def cond(index, acc): # pylint: disable=unused-argument
            return index < batch_size
        def body(index, acc):
            x = tf.exp(
                -tf.reduce_sum(
                    (self.positions - mean[i])**2,
                    axis=-1
                )/(2*variance[i])
            )*confidence[i]
            acc = tf.tensor_scatter_nd_add(acc, [[indices[i,0]]], updates=x[tf.newaxis])
            return i+1, acc
        i, acc = tf.while_loop(cond, body, [i, acc])
        return acc

    def __static_loop_over_batch_with_acc_implementation(self, mean, variance, confidence, **_):
        batch_size = tf.shape(mean)[0]
        height, width, _ = self.positions.get_shape().as_list()
        acc = tf.zeros((batch_size, height, width))

        for b in range(len(confidence)): # pylint: disable=consider-using-enumerate
            x = tf.reduce_sum(
                tf.exp(
                    -tf.reduce_sum(
                        (self.positions[:,:,tf.newaxis,:] - tf.reshape(mean[b], [1, 1, self.width*self.height, 2]))**2,
                        axis=-1
                    )/(2*tf.reshape(variance[b], [1, 1, self.width*self.height]))
                )*tf.reshape(confidence[b], [1, 1, self.width*self.height]),
                axis=-1
            )
            acc = tf.tensor_scatter_nd_update(acc, [[b]], updates=x[tf.newaxis])
        return acc
