import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator import find
import experimentator.tf2_experiment
import experimentator.wandb_experiment
import deepsport_utilities.ds.instants_dataset
from dataset_utilities.ds.raw_sequences_dataset import BallState
import deepsport_utilities.ds.instants_dataset.dataset_splitters
import tasks.ballstate
import tasks.classification
import models.other
import tasks.ballsize
import models.tensorflow

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.ballsize.BallSizeEstimation
    #tasks.ballstate.BallStateAndBallSizeExperiment,
]

batch_size = 16

with_diff = True

# Dataset parameters
side_length = 224
output_shape = (side_length, side_length)

# DeepSport Dataset
dataset_name = "ids_balls_dataset.pickle"
size_min = 14 #9   # 14
size_max = 37# 28  # 37
scale_min = 0.75
scale_max = 1.25
max_shift = 5

globals().update(locals()) # required for using locals in lambda
dataset = experimentator.CachedPickledDataset(find(dataset_name))

# transformation should be a function of a scale factor if I want to evaluate on
# strasbourg and gravelines sequences (in order to evaluate on the scale range trained on)
dataset = mlwf.TransformedDataset(dataset, [
    tasks.ballstate.BallViewRandomCropperTransformCompat(
        output_shape=output_shape,
        scale_min=scale_min,
        scale_max=scale_max,
        size_min=size_min,
        size_max=size_max,
        margin=side_length//2-max_shift,
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddNextImageFactory() if with_diff else None,
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallPositionFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
        #deepsport_utilities.ds.instants_dataset.views_transforms.AddBallStateFactory(),
        tasks.ballstate.AddBallSizeFactory(),
        tasks.ballstate.AddIsBallTargetFactory(),
    )
])

testing_arena_labels = []
dataset_splitter_str = 'deepsport'
validation_pc = 15
fold = 0
dataset_splitter = {
    "deepsport": deepsport_utilities.ds.instants_dataset.DeepSportDatasetSplitter(additional_keys_usage='testing2', validation_pc=validation_pc),
    "arenas_specific": deepsport_utilities.ds.instants_dataset.dataset_splitters.TestingArenaLabelsDatasetSplitter(testing_arena_labels, validation_pc=validation_pc),
}[dataset_splitter_str]
subsets = dataset_splitter(dataset, fold)
testing_arena_labels = dataset_splitter.testing_arena_labels

## add ballistic dataset
#dataset = mlwf.CachedDataset(mlwf.TransformedDataset(mlwf.PickledDataset(find("ballistic_ball_views.pickle")), transforms(.5)))
#subsets.append(Subset("ballistic", SubsetType.EVAL, dataset))


globals().update(locals()) # required to use 'BallState' in list comprehention
#classes = [BallState(i) for i in range(state_max+1)]
decay_start = 20
warmup = False
callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    #experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.LearningRateDecay(start=range(decay_start,101,10), duration=2, factor=.5) if not warmup else None,
    tasks.ballsize.ComputeDiameterError(),
#    tasks.classification.ComputeClassifactionMetrics(),
#    tasks.classification.ComputeConfusionMatrix(classes=classes),
#    experimentator.wandb_experiment.LogStateWandB(),
    experimentator.LearningRateWarmUp() if warmup else None,
#    tasks.classification.ExtractClassificationMetrics(class_name=str(BallState(1)), class_index=1),
#    tasks.classification.ExtractClassificationMetrics(class_name=str(BallState(2)), class_index=2),
    tasks.ballstate.ComputeDetectionMetrics(origin='ballseg'),
    tasks.detection.AuC("top1-AuC", "top1_metrics"),
    tasks.detection.AuC("top2-AuC", "top2_metrics"),
    tasks.detection.AuC("top4-AuC", "top4_metrics"),
    tasks.detection.AuC("top8-AuC", "top8_metrics"),
    tasks.detection.AuC("initial_top1-AuC", "initial_top1_metrics"),
    experimentator.wandb_experiment.LogStateWandB("validation_MAPE", False),
    tasks.ballstate.TopkNormalizedGain([1, 2, 4, 8]),
]

skip = True
backbone = {
    True: models.tensorflow.SkipConnectionCroppedInputsModelSixCannels,
    False: models.tensorflow.SixChannelsTensorflowBackbone,
}[skip]

alpha = .5
globals().update(locals()) # required to use locals() in lambdas
chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image", "batch_input_image2"]),
    lambda chunk: chunk.update({'batch_input_diff': tf.subtract(chunk["batch_input_image"], chunk["batch_input_image2"])}) if with_diff else None,
    models.other.GammaAugmentation("batch_input_image"),
    lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"] if not with_diff else tf.concat((chunk["batch_input_image"], chunk["batch_input_diff"]), axis=3)}),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    backbone("vgg16.VGG16", include_top=False),
    models.other.LeNetHead(output_features=2),#len(classes)),
    tasks.ballsize.NamedOutputs(),
    tasks.ballsize.ClassificationLoss(),
    tasks.ballsize.RegressionLoss(),
    #lambda chunk: chunk.update({"batch_target": tf.one_hot(chunk['batch_ball_state'], len(classes))}),
    #lambda chunk: chunk.update({"loss": tf.keras.losses.binary_crossentropy(chunk["batch_target"], chunk["batch_logits"], from_logits=True)}),
    #lambda chunk: chunk.update({"loss": tf.reduce_mean(chunk["loss"])}),
    lambda chunk: chunk.update({"loss": alpha*chunk["classification_loss"] + (1-alpha)*chunk["regression_loss"]}),
    lambda chunk: chunk.update({"predicted_is_ball": tf.nn.sigmoid(chunk["predicted_is_ball"])}),
    #lambda chunk: chunk.update({"batch_output": tf.nn.softmax(chunk["batch_logits"])})
]

learning_rate = 2e-5
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
