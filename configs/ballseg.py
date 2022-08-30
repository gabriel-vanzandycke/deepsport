import numpy as np
import tensorflow as tf
import mlworkflow as mlwf
import experimentator
import experimentator.wandb_experiment
from experimentator.utils import find

import experimentator.tf2_experiment
import experimentator.tf2_chunk_processors
import deepsport_utilities.ds.instants_dataset
import models.other
import models.icnet
import tasks.detection

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.detection.HeatmapDetectionExperiment
]

# Dataset parameters
output_shape = (640, 640)


with_diff = True

# DeepSport Dataset
#dataset_name = "camera_with_ball_visible_views.pickle"
dataset_name = "gva/camera_with_ball_visible_views.pickle"
size_min = 11
size_max = 28
transforms = [
    deepsport_utilities.ds.instants_dataset.views_transforms.BallViewRandomCropperTransform(
        output_shape=output_shape,
        size_min=size_min,
        size_max=size_max
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddDiffFactory() if with_diff else None,
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSegmentationTargetViewFactory(),
    )
]

#dataset_splitter = deepsport_utilities.ds.instants_dataset.DeepSportDatasetSplitter(additional_keys_usage="skip")
dataset_splitter = deepsport_utilities.ds.instants_dataset.TestingArenaLabelsDatasetSplitter(validation_pc=0, testing_arena_labels=['KS-FR-GRAVELINES', 'KS-FR-STRASBOURG'])
dataset = mlwf.TransformedDataset(mlwf.PickledDataset(find(dataset_name)), transforms)
subsets = dataset_splitter(dataset)



# Training parameters
batch_size      = 16

k = [1]
decay_start = (100, 200, 300)
callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    tasks.detection.ComputeTopkMetrics(k=k),
    tasks.detection.AuC("top1-AuC", "top1_metrics"),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.wandb_experiment.LogStateWandB(criterion_metric="validation_top1-AuC"),
    experimentator.LearningRateDecay(start=decay_start, duration=10, factor=0.1),
]

globals().update(locals()) # required to use 'tf' in lambdas
chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image", "batch_input_image2", "batch_target"]),
    lambda chunk: chunk.update({'batch_input_diff': tf.subtract(chunk["batch_input_image"], chunk["batch_input_image2"])}),
    models.other.GammaAugmentation("batch_input_image"),
    #lambda chunk: chunk.update({'batch_input': chunk['batch_input_image']}),
    lambda chunk: chunk.update({'batch_input': tf.concat((chunk["batch_input_image"], chunk["batch_input_diff"]), axis=3)}),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    models.icnet.ICNetBackbone(),
    models.icnet.ICNetHead(num_classes=1),
    experimentator.tf2_chunk_processors.SigmoidCrossEntropyLoss(),
    lambda chunk: chunk.update({"batch_heatmap": tf.nn.sigmoid(chunk["batch_logits"])}),
    tasks.detection.ComputeKeypointsDetectionHitmap(non_max_suppression_pool_size=int(size_max*1.1)),
    tasks.detection.ConfidenceHitmap(),
    tasks.detection.ComputeTopK(k=k),
    tasks.detection.EnlargeTarget(int(size_min/2)),
    tasks.detection.ComputeKeypointsTopKDetectionMetrics()
]

learning_rate   = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
