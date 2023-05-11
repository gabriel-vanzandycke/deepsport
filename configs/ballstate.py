import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator import find
import experimentator.tf2_experiment
import experimentator.wandb_experiment
from deepsport_utilities.dataset import Subset, SubsetType
import deepsport_utilities.ds.instants_dataset
from dataset_utilities.ds.raw_sequences_dataset import BallState
import deepsport_utilities.ds.instants_dataset.dataset_splitters
import tasks.ballstate
import tasks.classification
import models.other
import tasks.ballsize
import models.tensorflow

nstates = 1

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.ballstate.BallStateAndBallSizeExperiment if nstates else tasks.ballsize.BallSizeEstimation,
]

globals().update(locals()) # required to use 'BallState' in list comprehention
batch_size = 16

with_diff = True

# Dataset parameters
side_length = 224
output_shape = (side_length, side_length)

# DeepSport Dataset
dataset_name = "ids-deepsportradar_balls_dataset.pickle"
size_min = 9   # 14
size_max = 28  # 37
scale_min = 0.75
scale_max = 1.25
max_shift = 5

dataset = experimentator.CachedPickledDataset(find(dataset_name))


state_mapping = {
    0: {
        BallState.FLYING:     [1],
        BallState.CONSTRAINT: [0],
        BallState.DRIBBLING:  [0],
    },
    1: {
        BallState.FLYING:     [1],
        BallState.CONSTRAINT: [0],
        BallState.DRIBBLING:  [0],
    },
    2: {
        BallState.FLYING:     [0, 1, 0],
        BallState.CONSTRAINT: [0, 0, 1],
        BallState.DRIBBLING:  [0, 0, 1],
    },
    3: {
        BallState.FLYING:     [0, 1, 0, 0],
        BallState.CONSTRAINT: [0, 0, 1, 0],
        BallState.DRIBBLING:  [0, 0, 0, 1],
    }
}[nstates]
FLYING_index = 1 if nstates > 1 else 0

# transformation should be a function of a scale factor if I want to evaluate on
# strasbourg and gravelines sequences (in order to evaluate on the scale range trained on)
transforms = [
    tasks.ballstate.BallViewRandomCropperTransformCompat(
        output_shape=output_shape,
        scale_min=scale_min,
        scale_max=scale_max,
        size_min=size_min,
        size_max=size_max,
        margin=side_length//2-max_shift,
        padding=side_length//2,
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddNextImageFactory() if with_diff else None,
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallPositionFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallStateFactory(state_mapping),
        tasks.ballstate.AddBallSizeFactory(),
        tasks.ballstate.AddIsBallTargetFactory(),
    )
]
dataset = mlwf.TransformedDataset(dataset, transforms)
globals().update(locals()) # required for using locals in lambda
#dataset = mlwf.FilteredDataset(dataset, lambda k,v: v['ball_state'] in list(state_mapping.keys()), cache=True)

testing_arena_labels = ('KS-FR-STRASBOURG', 'KS-FR-GRAVELINES', 'KS-FR-BOURGEB', 'KS-FR-EVREUX')
dataset_splitter_str = 'arenas_specific'
validation_pc = 0 if nstates == 0 else 10
fold = 0
dataset_splitter = {
    "deepsport": deepsport_utilities.ds.instants_dataset.DeepSportDatasetSplitter(additional_keys_usage='testing2', validation_pc=validation_pc),
    "arenas_specific": deepsport_utilities.ds.instants_dataset.dataset_splitters.TestingArenaLabelsDatasetSplitter(testing_arena_labels, validation_pc=validation_pc),
}[dataset_splitter_str]
subsets = dataset_splitter(dataset, fold)
if nstates == 0:
    subsets.append(Subset("testing2", SubsetType.EVAL, mlwf.TransformedDataset(experimentator.CachedPickledDataset(find("ids-private_balls_dataset.pickle")), transforms)))

## add ballistic dataset
#dataset = mlwf.CachedDataset(mlwf.TransformedDataset(mlwf.PickledDataset(find("ballistic_ball_views.pickle")), transforms(.5)))
#subsets.append(Subset("ballistic", SubsetType.EVAL, dataset))


decay_start = 20
decay_step = 10
warmup = False
save_weights = False
callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experimentator.SaveWeights() if save_weights else None,
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.LearningRateDecay(start=range(decay_start,101,decay_step), duration=2, factor=.5) if not warmup else None,
    tasks.ballsize.ComputeDiameterError(),
    experimentator.wandb_experiment.LogStateWandB(),
    experimentator.LearningRateWarmUp() if warmup else None,
    tasks.classification.ComputeClassifactionMetrics(logits_key="predicted_state", target_key="batch_ball_state", name="state_classification") if nstates else None,
    tasks.classification.ExtractClassificationMetrics(class_name="BallState.FLYING", class_index=FLYING_index, name="state_classification") if nstates else None,
    tasks.ballstate.StateFLYINGMetrics(),
    tasks.ballstate.ComputeDetectionMetrics(origin='ballseg') if dataset_name != "sds_balls_dataset.pickle" else None,
    tasks.detection.AuC("top1-AuC", "top1_metrics"),
    tasks.detection.AuC("top2-AuC", "top2_metrics"),
    tasks.detection.AuC("top4-AuC", "top4_metrics"),
    tasks.detection.AuC("top8-AuC", "top8_metrics"),
    tasks.detection.AuC("initial_top1-AuC", "initial_top1_metrics"),
    tasks.ballstate.TopkNormalizedGain([1, 2, 4, 8]) if dataset_name != "sds_balls_dataset.pickle" else None,
]


balancer = tasks.ballstate.StateOnlyBalancer if nstates > 1 else None
ballsize_weights = "20230421_112340.495561" if nstates else None
freeze_ballsize = True if nstates else False
alpha = None if nstates else 0.9

globals().update(locals()) # required to use locals() in lambdas
chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image", "batch_input_image2"]),
    lambda chunk: chunk.update({'batch_input_diff': tf.subtract(chunk["batch_input_image"], chunk["batch_input_image2"])}) if with_diff else None,
    models.other.GammaAugmentation("batch_input_image"),
    lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"] if not with_diff else tf.concat((chunk["batch_input_image"], chunk["batch_input_diff"]), axis=3)}),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    models.tensorflow.SixChannelsTensorflowBackbone("vgg16.VGG16", include_top=False),
    models.other.LeNetHead(name="regression", output_features=2),
    models.other.LeNetHead(name="classification", output_features=len(state_mapping[1])) if nstates else None,
    tasks.ballsize.NamedOutputs("regression_logits"),
    tasks.ballsize.IsBallClassificationLoss(),
    tasks.ballsize.RegressionLoss(),
    lambda chunk: chunk.update({"predicted_state": chunk["classification_logits"]}) if nstates else None,
    tasks.ballstate.StateClassificationLoss() if nstates else \
        tasks.ballstate.CombineLosses(["classification_loss", "regression_loss"], weights=[alpha, 1-alpha]),
    lambda chunk: chunk.update({"predicted_is_ball": tf.nn.sigmoid(chunk["predicted_is_ball"])}),
    lambda chunk: chunk.update({"predicted_state": tf.nn.sigmoid(chunk["predicted_state"])}) if nstates else None,
]

learning_rate = 2e-5
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
