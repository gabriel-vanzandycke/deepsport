import tensorflow as tf
import mlworkflow as mlwf
import experimentator
import experimentator.wandb_experiment
from experimentator import find

import experimentator.tf2_experiment
import experimentator.tf2_chunk_processors
import deepsport_utilities.ds.instants_dataset
import models.other
import tasks.pifball

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.pifball.PIFExperiment
]

# Dataset parameters
output_shape = (640, 640)

# Backbone parameters
depth_to_space = 2
backbone_stride = 16

# DeepSport Dataset
dataset_name = "camera_with_ball_visible_views.pickle"
size_min = 11
size_max = 28
transforms = [
    deepsport_utilities.ds.instants_dataset.views_transforms.BallViewRandomCropperTransform(
        output_shape=output_shape,
        size_min=size_min,
        size_max=size_max,
        on_ball=.5,
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        tasks.pifball.AddPIFBallTargetViewFactory(stride=backbone_stride//depth_to_space),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSegmentationTargetViewFactory()
    )
]

#dataset_splitter = deepsport_utilities.ds.instants_dataset.TestingArenaLabelsDatasetSplitter(validation_pc=0, testing_arena_labels=['KS-FR-GRAVELINES', 'KS-FR-STRASBOURG'])
fold = 0
dataset_splitter = deepsport_utilities.ds.instants_dataset.dataset_splitters.KFoldsArenaLabelsTestingDatasetSplitter(8, 0, 1)
dataset = experimentator.dataset.DataAugmentationDataset(mlwf.PickledDataset(find(dataset_name)), transforms)
subsets = dataset_splitter(dataset, fold=fold)
testing_arena_labels = dataset_splitter.testing_arena_labels



# Training parameters
batch_size = 4

k = [1]
callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    tasks.detection.ComputeTopkMetrics(k=k),
    tasks.detection.AuC("top1-AuC", "top1_metrics"),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.wandb_experiment.LogStateWandB(criterion_metric="validation_top1-AuC"),
    experimentator.LearningRateWarmUp(start=0, duration=1, factor=0.0001),
    #experimentator.StopFailedTraining(),
]


lambdas = [30.0, 2.0, 2.0]
globals().update(locals()) # required to use '' in lambdas
chunk_processors = [
    tasks.pifball.CastFloat(tensor_names=["batch_input_image", "batch_target"]),
    models.other.GammaAugmentation(tensor_name="batch_input_image"),
    lambda chunk: chunk.update({'batch_input': chunk["batch_input_image"]}),
    tasks.pifball.Normalize(tensor_names="batch_input"),
    experimentator.tf2_chunk_processors.DatasetStandardize(tensor_names="batch_input", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    tasks.pifball.CustomResnet50(),
    tasks.pifball.CompositeFieldFused(depth_to_space=depth_to_space),
    tasks.pifball.CompositeLoss(lambdas=lambdas),
    tasks.pifball.DecodePif(stride=backbone_stride//depth_to_space),
    lambda chunk: chunk.update({'batch_output': chunk['batch_heatmap']}),
    experimentator.tf2_chunk_processors.ExpandDims(tensor_names=["batch_heatmap"]),
    tasks.detection.ComputeKeypointsDetectionHitmap(non_max_suppression_pool_size=int(size_max*1.1)),
    tasks.detection.ConfidenceHitmap(),
    tasks.detection.ComputeTopK(k=k),
    tasks.detection.EnlargeTarget(int(size_min/2)),
    tasks.detection.ComputeKeypointsTopKDetectionMetrics(),
]

learning_rate = 0.001
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
