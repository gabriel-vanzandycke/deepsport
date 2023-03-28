import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator import find
import experimentator.tf2_experiment
import deepsport_utilities.ds.instants_dataset
import models.other
import models.tensorflow
import experimentator.wandb_experiment
import tasks.ballsize
import tasks.detection

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
size_max = 37
max_shift = 0
transforms = [
    deepsport_utilities.ds.instants_dataset.BallViewRandomCropperTransform(
        output_shape=output_shape,
        size_min=size_min,
        size_max=size_max,
        margin=side_length//2-max_shift,
        on_ball=True,
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSizeFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallPositionFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
        tasks.ballsize.AddIsBallTargetFactory(),
    )
]

dataset_splitter_str = 'deepsport'
validation_pc = 15
testing_arena_labels = []
dataset = mlwf.PickledDataset(find(dataset_name))
globals().update(locals()) # required to use locals() in lambdas
dataset = mlwf.TransformedDataset(dataset, transforms) # CachedDataset fails for whatever reason
fold = 0
dataset_splitter = {
    "deepsport": deepsport_utilities.ds.instants_dataset.DeepSportDatasetSplitter(additional_keys_usage='testing2', validation_pc=validation_pc),
    "arenas_specific": deepsport_utilities.ds.instants_dataset.dataset_splitters.TestingArenaLabelsDatasetSplitter(testing_arena_labels, validation_pc=validation_pc),
}[dataset_splitter_str]
subsets = dataset_splitter(dataset, fold)
testing_arena_labels = dataset_splitter.testing_arena_labels

callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.LearningRateDecay(start=range(50,101,10), duration=2, factor=.5),
    tasks.ballsize.ComputeDiameterError(),
    tasks.ballsize.ComputeDetectionMetrics(),
    tasks.detection.AuC("top1-AuC", "top1_metrics"),
    experimentator.wandb_experiment.LogStateWandB("validation_MAPE", False),
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
