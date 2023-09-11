import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator import find
import experimentator.tf2_experiment
import experimentator.wandb_experiment
import deepsport_utilities.ds.instants_dataset
import deepsport_utilities.dataset
from dataset_utilities.ds.raw_sequences_dataset import BallState
import deepsport_utilities.ds.instants_dataset.dataset_splitters
import tasks.ballstate
import tasks.classification
import models.other
import tasks.ballsize
import models.tensorflow

nstates = 2
wd = 1
wp = 1
ws = 1

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.ballstate.AugmentedExperiment,
    {
        False: tasks.ballsize.BallSizeEstimation,
        True: tasks.ballstate.BallStateAndBallSizeExperiment
    }[bool(nstates)]
]

estimate_presence = True
with_diff = True
public_dataset = False

globals().update(locals()) # required to use 'BallState' in list comprehention
batch_size = 16


# Dataset parameters
side_length = 64
output_shape = (side_length, side_length)

# DeepSport Dataset
dataset_name1 = "ballsize_dataset_256_with_detections_from_model_trained_on_full_dataset_stitched.pickle"
dataset_name2 = "sds_balls_dataset.pickle"

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
        BallState.NONE:       [1, 0, 0],
        BallState.FLYING:     [0, 1, 0],
        BallState.CONSTRAINT: [0, 0, 1],
        BallState.DRIBBLING:  [0, 0, 1],
    },
    3: {
        BallState.NONE:       [1, 0, 0, 0],
        BallState.FLYING:     [0, 1, 0, 0],
        BallState.CONSTRAINT: [0, 0, 1, 0],
        BallState.DRIBBLING:  [0, 0, 0, 1],
    }
}[nstates]
FLYING_index = 1 if nstates > 1 else 0

# transformation should be a function of a scale factor if I want to evaluate on
# strasbourg and gravelines sequences (in order to evaluate on the scale range trained on)
size_min = 10   # 14
size_max = 30   # 37
scale_min = 0.75
scale_max = 1.25
max_shift = 10
transforms = [
    tasks.ballstate.BallViewRandomCropperTransformCompat(
        output_shape=output_shape,
        scale_min=scale_min,
        scale_max=scale_max,
        size_min=size_min,
        size_max=size_max,
        margin=side_length//2-max_shift,
        padding=side_length,
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddNextImageFactory() if with_diff else None,
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallPositionFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallStateFactory(state_mapping),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSizeFactory(origins=['annotation', 'interpolation', 'ballseg']),
        tasks.ballstate.AddBallPresenceFactory(),
    ),
]

if nstates == 0 or ws == 0:
    dataset = experimentator.CachedPickledDataset(find(dataset_name1))
else:
    ds1 = experimentator.CachedPickledDataset(find(dataset_name1))
    ds2 = experimentator.CachedPickledDataset(find(dataset_name2))
    dataset = deepsport_utilities.dataset.MergedDataset(ds1, ds2)
dataset = mlwf.TransformedDataset(dataset, transforms)
globals().update(locals()) # required for using locals in lambda

testing_arena_labels = ('KS-FR-STRASBOURG', 'KS-FR-GRAVELINES', 'KS-FR-BOURGEB', 'KS-FR-EVREUX')
validation_arena_labels = None#('KS-UK-NEWCASTLE', 'KS-US-IPSWICH', 'KS-FI-KAUHAJOKI', 'KS-FR-LEMANS', 'KS-FR-ESBVA', 'KS-FR-NANTES') if nstates == 0 else None
dataset_splitter = deepsport_utilities.ds.instants_dataset.dataset_splitters.TestingValidationArenaLabelsDatasetSplitter(testing_arena_labels, validation_arena_labels)
dataset_splitter_type = dataset_splitter.__class__.__name__
assert public_dataset is False
subsets = dataset_splitter(dataset)

balance_datasets = True
if balance_datasets:
    subsets = [deepsport_utilities.dataset.BalancedSubset(s, ['InstantKey', 'SequenceInstantKey'], lambda k: k.__class__.__name__)
           for s in subsets]

## add ballistic dataset
#dataset = mlwf.CachedDataset(mlwf.TransformedDataset(mlwf.PickledDataset(find("ballistic_ball_views.pickle")), transforms(.5)))
#subsets.append(Subset("ballistic", SubsetType.EVAL, dataset))

decay_start = 20
decay_step = 10
decay_factor = .5
warmup = False
save_weights = True
callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experimentator.SaveWeights() if save_weights else None,
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.LearningRateDecay(start=range(decay_start,101,decay_step), duration=2, factor=decay_factor) if not warmup else None,
    tasks.ballsize.ComputeDiameterError(),
    experimentator.wandb_experiment.LogStateWandB(),
    experimentator.LearningRateWarmUp() if warmup else None,
    tasks.classification.ComputeClassifactionMetrics(logits_key="predicted_state", target_key="batch_ball_state", name="state_classification"),
    tasks.classification.ExtractClassificationMetrics(class_name="BallState.FLYING", class_index=FLYING_index, name="state_classification"),
    tasks.ballstate.StateFLYINGMetrics(),
    tasks.ballstate.ComputeDetectionMetrics(origin='ballseg', key=lambda view_key: view_key.instant_key),
    tasks.detection.AuC("top1-AuC", "top1_metrics"),
    tasks.detection.AuC("top2-AuC", "top2_metrics"),
    tasks.detection.AuC("top4-AuC", "top4_metrics"),
    tasks.detection.AuC("top8-AuC", "top8_metrics"),
    tasks.detection.AuC("initial_top1-AuC", "initial_top1_metrics"),
]


balancer = tasks.ballstate.StateOnlyBalancer if nstates > 1 else None
#ballsize_weights = "20230829_095035.167455" # trained on sizes [9;30]

#starting_weights = "20230905_104213.152062"
starting_weights_trainable = {"vgg16": True, "diameter_head": True, "presence_head": True}

#alpha = 0.5
#beta = 1

globals().update(locals()) # required to use locals() in lambdas
chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image", "batch_input_image2", "batch_target", "batch_ball_position"]),
    lambda chunk: chunk.update({'batch_input_diff': tf.subtract(chunk["batch_input_image"], chunk["batch_input_image2"])}) if with_diff else None,
    models.other.GammaAugmentation("batch_input_image"),
    lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"] if not with_diff else tf.concat((chunk["batch_input_image"], chunk["batch_input_diff"]), axis=3)}),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    models.tensorflow.SixChannelsTensorflowBackbone("vgg16.VGG16", include_top=False),
    models.other.LeNetHead(name="state", output_features=len(state_mapping[1])),
    models.other.LeNetHead(name="diameter", output_features=1),
    models.other.LeNetHead(name="presence", output_features=1),
    lambda chunk: chunk.update({"predicted_diameter": chunk['diameter_logits'][...,0]}),
    lambda chunk: chunk.update({"predicted_presence": chunk['presence_logits'][...,0]}),
    lambda chunk: chunk.update({"predicted_state": chunk["state_logits"]}),
    models.other.BinaryCrossEntropyLoss(y_true="batch_ball_presence", y_pred="predicted_presence", name="presence"),
    models.other.HuberLoss(y_true='batch_ball_size', y_pred='predicted_diameter', name='diameter'),
    tasks.ballstate.StateClassificationLoss(),
    experimentator.tf2_chunk_processors.CombineLosses(["diameter_loss", "presence_loss", "state_loss"], weights=[wd, wp, ws]),
    lambda chunk: chunk.update({"predicted_presence": tf.nn.sigmoid(chunk["predicted_presence"])}),
    lambda chunk: chunk.update({"predicted_state": tf.nn.sigmoid(chunk["predicted_state"])}),
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
