import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator.utils import find
import experimentator.tf2_experiment
import deepsport_utilities.ds.instants_dataset
import models.other
import tasks.ballsize
import tasks.detection
import tasks.ballsize_from_ballseg


experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.ballsize_from_ballseg.BallSizeFromBallSegExperiment
]

batch_size = 2

# Dataset parameters
output_shape = (640, 640)

# DeepSport Dataset
dataset_name = "camera_views_with_ball_visible.pickle"
size_min = 14
size_max = 37
on_ball = False
transforms = [
    deepsport_utilities.ds.instants_dataset.views_transforms.BallViewRandomCropperTransform(
        output_shape=output_shape,
        size_min=size_min,
        size_max=size_max,
        on_ball=on_ball,
        margin=100
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSizeFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallPositionFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSegmentationTargetViewFactory(),
    )
]
dataset_splitter = deepsport_utilities.ds.instants_dataset.DeepSportDatasetSplitter(additional_keys_usage="skip")
dataset = mlwf.TransformedDataset(mlwf.PickledDataset(find(dataset_name)), transforms)
subsets = dataset_splitter(dataset)


k = 4

# Training parameters
callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.LearningRateDecay(start=range(50,101,10), duration=2, factor=0.5),
    tasks.ballsize_from_ballseg.ReapeatData(names=['batch_calib', 'batch_ball'], k=k),
    tasks.ballsize.ComputeDiameterError(),
    tasks.ballsize.ComputeDetectionMetrics(),
    tasks.detection.AuC("top1-AuC", "top1_metrics"),
    tasks.ballsize_from_ballseg.ComputeBallSegTopkMetrics(k=[1]),
    tasks.detection.AuC("ballseg_top1-AuC", "ballseg_top1_metrics"),
    tasks.ballsize_from_ballseg.ComputeDiameterErrorFromPatchHeatmap(),
]


oracle = False
side_length = 64

ballseg_config = None
globals().update(locals()) # required to use 'tf' in lambdas
chunk_processors = [
    tasks.ballsize_from_ballseg.BallSegModel(config=ballseg_config, batch_size=batch_size, k=k, output_shape=output_shape),
    tasks.ballsize_from_ballseg.BallSegCandidates(side_length, oracle=oracle),
    tasks.ballsize_from_ballseg.CNNModel(batch_size=batch_size*k, side_length=side_length)
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
