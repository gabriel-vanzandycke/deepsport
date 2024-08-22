import deepsport_utilities.dataset
import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator import find
import experimentator.tf2_experiment
import deepsport_utilities.ds.scoreboards_dataset
import models.other
import models.tensorflow
import experimentator.wandb_experiment
import tasks.scoreboards

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.scoreboards.DetectScoreboardsExperiment
]

batch_size = 16

# Dataset parameters
output_shape = (512, 256)
dataset_name = "scoreboards_dataset.pickle"

scale = 1
size_min = 0.8
size_max = 1.2
max_shift = 50

transforms = [
    deepsport_utilities.ds.scoreboards_dataset.RandomScalingCropperTransform(output_shape, size_min, size_max, max_shift),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.scoreboards_dataset.AddImageFactory(),
        deepsport_utilities.ds.scoreboards_dataset.AddNumbersHeatmapFactory()
    ),
]

dataset = mlwf.PickledDataset(find(dataset_name))
dataset = mlwf.TransformedDataset(dataset, transforms)
subsets = deepsport_utilities.dataset.BasicDatasetSplitter()(dataset)

callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    #experimentator.SaveWeights(),
    #experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    #experimentator.LogStateDataCollector(),
    #experimentator.LearningRateDecay(start=range(50,101,10), duration=2, factor=.5),
    #experimentator.wandb_experiment.LogStateWandB(),
]

chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image"]),
    lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"]}),
    models.other.GammaAugmentation("batch_input"),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    models.tensorflow.TensorflowBackbone("vgg16.VGG16", include_top=False),
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
