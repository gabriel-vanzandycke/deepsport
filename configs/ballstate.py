import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator.utils import find
import experimentator.tf2_experiment
import experimentator.wandb_experiment
import deepsport_utilities.ds.instants_dataset
from dataset_utilities.ds.raw_sequences_dataset import BallState
import tasks.ballstate
import tasks.classification
import models.other
import models.tensorflow

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.ballstate.BallStateClassification
]

batch_size = 16

# Dataset parameters
side_length = 114
output_shape = (side_length, side_length)

# DeepSport Dataset
dataset_name = "ball_states_dataset.pickle"
scale_min = 0.75
scale_max = 1.25
max_shift = 0
transforms = [
    tasks.ballstate.BallCropperTransform(
        output_shape=output_shape,
        scale_min=scale_min,
        scale_max=scale_max,
        margin=side_length//2-max_shift
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        tasks.ballstate.AddBallStateFactory(),
    )
]

dataset_splitter = experimentator.BasicDatasetSplitter()
dataset = mlwf.TransformedDataset(mlwf.PickledDataset(find(dataset_name)), transforms)
subsets = dataset_splitter(dataset)

classes = [BallState(i) for i in range(4)]

callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    tasks.classification.ComputeClassifactionMetrics(),
    tasks.classification.ConfusionMatrix(classes=classes),
    experimentator.wandb_experiment.LogStateWandB(),
    experimentator.LearningRateDecay(start=range(50,101,10), duration=2, factor=.5),
]

globals().update(locals()) # required to use locals() in lambdas
chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image"]),
    lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"]}),
    models.other.GammaAugmentation("batch_input"),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    models.tensorflow.TensorflowBackbone("vgg16.VGG16", include_top=False),
    models.other.LeNetHead(output_features=len(classes)),
    lambda chunk: chunk.update({"batch_target": tf.one_hot(chunk['batch_ball_state'], len(classes))}),
    lambda chunk: chunk.update({"loss": tf.keras.losses.binary_crossentropy(chunk["batch_target"], chunk["batch_logits"], from_logits=True)}),
    lambda chunk: chunk.update({"loss": tf.reduce_mean(chunk["loss"])}),
    lambda chunk: chunk.update({"batch_output": tf.nn.softmax(chunk["batch_logits"])})
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
