import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator.utils import find
import experimentator.tf2_experiment
import experimentator.tf2_chunk_processors
import dataset_utilities.ds.instants_dataset
import experiment.models.custom
import experiment.models.icnet
import experiment.tasks.detection

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    experiment.tasks.detection.HeatmapDetectionExperiment
]

# Dataset parameters
output_shape = (640, 640)

# DeepSport Dataset
dataset_name = "camera_views_with_ball_visible.pickle"
scaled = True
size_min = 14
size_max = 37
globals().update(locals()) # required to use 'tf' in lambdas
transforms = [
    dataset_utilities.ds.instants_dataset.views_transforms.BallViewRandomCropperTransform(
        output_shape=output_shape,
        size_min=size_min,
        size_max=size_max
    ),
    dataset_utilities.transforms.DataExtractorTransform(
        dataset_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        dataset_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        dataset_utilities.ds.instants_dataset.views_transforms.AddBallSegmentationTargetViewFactory(),
    )
]

dataset_splitter = dataset_utilities.ds.instants_dataset.DeepSportDatasetSplitter(additional_keys_usage="skip")
dataset = mlwf.TransformedDataset(mlwf.PickledDataset(find(dataset_name)), transforms)
subsets = dataset_splitter(dataset)



# Training parameters
batch_size      = 16

k = [1]
decay_start = (100, 200, 300)
callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experiment.tasks.detection.ComputeTopkMetrics(k=k),
    experiment.tasks.detection.AuC("top1-AuC", "top1_metrics"),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.LearningRateDecay(start=decay_start, duration=10, factor=0.1),
]

chunk_processors = [
    #experiment.chunk_processors.CropBlockDividable(tensor_names=["batch_input_image", "batch_target"]),
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image", "batch_target"]),
    experiment.models.custom.GammaAugmentation("batch_input_image"),
    lambda chunk: chunk.update({'batch_input': chunk['batch_input_image']}),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    experiment.models.icnet.CustomICNet(),
    experiment.models.icnet.ICNetHead(num_classes=1),
    experimentator.tf2_chunk_processors.SigmoidCrossEntropyLoss(),
    lambda chunk: chunk.update({"batch_output": tf.nn.sigmoid(chunk["batch_logits"])}),
    lambda chunk: chunk.update({"batch_heatmap": chunk["batch_output"][:,:,:,0]}),
    #experiment.tasks.ball_keemotion.HeatmapToPoints(window_size=50, top_k=10, threshold=0.2),
    lambda chunk: chunk.update({"batch_target": tf.nn.max_pool2d(chunk["batch_target"][..., tf.newaxis], int(size_min/2), strides=1, padding='SAME')}), # enlarge target
    experiment.tasks.detection.ComputeKeypointsDetectionHitmap(non_max_suppression_pool_size=int(size_max*1.1)),
    experiment.tasks.detection.ConfidenceHitmap(),
    experiment.tasks.detection.ComputeKeypointsTopKDetectionMetrics(k=k)
]

learning_rate   = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
