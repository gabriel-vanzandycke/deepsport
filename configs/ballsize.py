import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator.utils import find
import experimentator.tf2_experiment
import deepsport_utilities.ds.instants_dataset
import tasks.ballsize
import models.other
import models.tensorflow

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.ballsize.BallSizeEstimation
]

batch_size = 16

# Dataset parameters
side_length = 64
output_shape = (side_length, side_length)

# DeepSport Dataset
dataset_name = "ball_views.pickle"
size_min = 14
size_max = 45
max_shift = 10
transforms = [
    tasks.ballsize.BallRandomCropperTransform(
        output_shape=output_shape,
        size_min=size_min,
        size_max=size_max,
        margin=side_length//2-max_shift
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSizeFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallPositionFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
    )
]

dataset_splitter = deepsport_utilities.ds.instants_dataset.DeepSportDatasetSplitter(additional_keys_usage="skip")
dataset = mlwf.TransformedDataset(mlwf.PickledDataset(find(dataset_name)), transforms)
subsets = dataset_splitter(dataset)

callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.LearningRateDecay(start=range(50,101,10), duration=2, factor=.5),
    tasks.ballsize.ComputeDiameterError(),
    tasks.ballsize.ComputeDetectionMetrics(),
]

alpha = 0.5
globals().update(locals()) # required to use locals() in lambdas
chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image"]),
    lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"]}),
    models.other.GammaAugmentation("batch_input"),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    models.tensorflow.TensorflowBackbone("vgg16.VGG16", include_top=False),
    models.other.LeNetHead(output_features=2),
    tasks.ballsize.NamedOutputs(),
    tasks.ballsize.ClassificationLoss(),
    tasks.ballsize.RegressionLoss(),
    lambda chunk: chunk.update({"loss": alpha*chunk["classification_loss"] + (1-alpha)*chunk["regression_loss"]}),
    lambda chunk: chunk.update({"predicted_is_ball": tf.nn.sigmoid(chunk["predicted_is_ball"])}),
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
