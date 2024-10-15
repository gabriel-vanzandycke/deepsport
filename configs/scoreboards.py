import deepsport_utilities.dataset
import tensorflow as tf
import experimentator
from experimentator import find
import experimentator.tf2_experiment
import dataset_utilities.ds.scoreboards_dataset
import deepsport_utilities.ds.instants_dataset
import deepsport_utilities.transforms
import models.other
import models.tensorflow
import experimentator.wandb_experiment
import tasks.scoreboards
import models.icnet

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.scoreboards.DetectScoreboardsExperiment
]

batch_size = 16

# Dataset parameters
output_shape = (512, 512)
dataset_name = "scoreboards_dataset_full.pickle"
dataset = experimentator.CachedPickledDataset(find(dataset_name))

scale = .5
transforms = [
    dataset_utilities.ds.scoreboards_dataset.RandomScalingCropperTransform(output_shape, scale=scale),
    dataset_utilities.ds.scoreboards_dataset.FillBackgroundTransform(dataset),
    deepsport_utilities.transforms.DataExtractorTransform(
        dataset_utilities.ds.scoreboards_dataset.AddImageFactory(),
        dataset_utilities.ds.scoreboards_dataset.AddNumbersHeatmapFactory()
    ),
]

dataset = experimentator.dataset.DataAugmentationDataset(dataset, transforms)
subsets = deepsport_utilities.ds.instants_dataset.KFoldsArenaLabelsTestingDatasetSplitter(evaluation_sets_repetitions=1)(dataset)

callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    #experimentator.LearningRateDecay(start=range(50,101,10), duration=2, factor=.5),
    experimentator.wandb_experiment.LogStateWandB(),
    #experimentator.tf2_experiment.ProfileCallback(),
]

globals().update(locals()) # required to use locals() in lambdas

chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image", "batch_target"]),
    lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"]}),
    models.other.GammaAugmentation("batch_input"),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    models.icnet.ICNetBackbone(),
    models.icnet.ICNetHead(num_classes=1),
    experimentator.tf2_chunk_processors.SigmoidCrossEntropyLoss(),
    lambda chunk: chunk.update({"batch_heatmap": tf.nn.sigmoid(chunk["batch_logits"])}),
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
