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

estimate_presence = False
public_dataset = False
ballistic_dataset = True
estimate_mask = True
estimate_offset = True
assert (ballistic_dataset is False) or (public_dataset is False), "annotated ballistic sequences from raw sequences dataset cannot be used when training on public dataset"


# DeepSport Dataset
dataset_name = {
    (True, False): "ballsize_dataset_256_with_detections_from_model_trained_on_full_dataset_new.pickle",
    (True, True): "ballsize_dataset_256_with_detections_from_model_trained_on_small_dataset_top4.pickle",
    (True, None): "ballsize_dataset_256_with_detections_from_model_trained_on_small_dataset.pickle",
    (False, True): "ballsize_dataset_256_no_detections.pickle",
    (False, False): "ballsize_dataset_256_no_detections.pickle",
}[(estimate_presence, public_dataset)]

scale = 1
size_min = 14*scale
size_max = 37*scale
max_shift = 10

data_extractor_transform = deepsport_utilities.transforms.DataExtractorTransform(
    deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
    deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSizeFactory(origins=['annotation', 'interpolation', 'ballseg']),
    deepsport_utilities.ds.instants_dataset.views_transforms.AddBallFactory(),
    deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
    deepsport_utilities.ds.instants_dataset.views_transforms.AddBallPositionFactory() if estimate_offset else None,
    deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSegmentationTargetViewFactory() if estimate_mask else None,
    deepsport_utilities.ds.instants_dataset.views_transforms.AddBallPresenceFactory() if estimate_presence else None,
)

random_size_cropper_transform = deepsport_utilities.ds.instants_dataset.BallViewRandomCropperTransform(
    output_shape=output_shape,
    size_min=size_min,
    size_max=size_max,
    margin=side_length//2-max_shift,
    padding=side_length,
    on_ball=True,
)

fixed_scale_cropper_transform = deepsport_utilities.ds.instants_dataset.BallViewRandomCropperTransform(
    output_shape=output_shape,
    scale_min=1*scale,
    scale_max=1*scale,
    margin=int(side_length*scale)//2-max_shift,
    padding=int(side_length*scale),
    on_ball=True,
)

dataset = experimentator.CachedPickledDataset(find(dataset_name))
dataset = mlwf.FilteredDataset(dataset, lambda k: ballistic_dataset or bool(isinstance(k[0], deepsport_utilities.ds.instants_dataset.InstantKey)))
dataset = mlwf.FilteredDataset(dataset, lambda k,v: estimate_presence or v.ball.origin in ['annotation', 'interpolation'] and bool(v.ball.visible))
evaluation_dataset_name = "ballistic_ball_views.pickle"
evaluation_dataset = experimentator.CachedPickledDataset(find(evaluation_dataset_name, verbose=True))

globals().update(locals()) # required to use locals() in lambdas

repetitions = 1

dataset_splitter = deepsport_utilities.ds.instants_dataset.DeepSportDatasetSplitter(
    additional_keys_usage='testing2' if public_dataset else 'training',
    validation_pc=15,
    repetitions={'testing': repetitions})

random_size_subsets = dataset_splitter(experimentator.dataset.DataAugmentationDataset(dataset, [random_size_cropper_transform, data_extractor_transform]))
fixed_scale_subsets = dataset_splitter(experimentator.dataset.DataAugmentationDataset(dataset, [fixed_scale_cropper_transform, data_extractor_transform]))

random_size_subsets[2].name = "legacy_testing"
subsets = [
    random_size_subsets[0],
    fixed_scale_subsets[1],
    fixed_scale_subsets[2],
    random_size_subsets[2],
    experimentator.Subset("3d_testset",        experimentator.Stage.EVAL, experimentator.dataset.DataAugmentationDataset(evaluation_dataset, [fixed_scale_cropper_transform, data_extractor_transform]), repetitions=repetitions),
    experimentator.Subset("legacy_3d_testset", experimentator.Stage.EVAL, experimentator.dataset.DataAugmentationDataset(evaluation_dataset, [random_size_cropper_transform, data_extractor_transform]), repetitions=repetitions),
]
if public_dataset:
    random_size_subsets[3].name = "legacy_testing2"
    subsets.extend([
        fixed_scale_subsets[3],
        random_size_subsets[3],
    ])


callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.LearningRateDecay(start=range(50,101,10), duration=2, factor=.5),
    tasks.ballsize.ComputeDiameterError(),
    tasks.detection.ComputeDetectionMetrics(origin='ballseg') if estimate_presence else None,
    tasks.detection.AuC("top1-AuC", "top1_metrics") if estimate_presence else None,
    tasks.detection.AuC("top2-AuC", "top2_metrics") if estimate_presence else None,
    tasks.detection.AuC("top4-AuC", "top4_metrics") if estimate_presence else None,
    tasks.detection.AuC("top8-AuC", "top8_metrics") if estimate_presence else None,
    tasks.detection.AuC("initial_top1-AuC", "initial_top1_metrics") if estimate_presence else None,
    experimentator.wandb_experiment.LogStateWandB("validation_MAPE", False),
]

alpha = 0.5 if estimate_presence else 0
alpha_m = 1 if estimate_mask else 0
alpha_o = 1 if estimate_offset else 0
alpha_d = 1
globals().update(locals()) # required to use locals() in lambdas
chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image"]),
    lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"]}),
    models.other.GammaAugmentation("batch_input"),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    models.tensorflow.TensorflowBackbone("vgg16.VGG16", include_top=False),
    tasks.ballsize.BuildMaskFromLogits() if estimate_mask else None,
    models.other.LeNetHead(output_features=4),
    tasks.ballsize.NamedOutputs(estimate_presence=estimate_presence, estimate_offset=estimate_offset),
    models.other.BinaryCrossEntropyLoss(y_true="batch_ball_presence", y_pred="predicted_presence", name="classification") if estimate_presence else None,
    models.other.HuberLoss(y_true='batch_ball_size', y_pred='predicted_diameter', name='regression'),
    tasks.ballsize.MaskSupervision() if estimate_mask else None,
    tasks.ballsize.OffsetSupervision() if estimate_offset else None,
    #lambda chunk: chunk.update({"loss": (alpha*chunk["classification_loss"] + (1-alpha)*chunk["regression_loss"]) if estimate_presence else chunk["regression_loss"]}),
    tasks.ballsize.CombineLosses({'regression_loss': alpha_d, 'offset_loss': alpha_o, 'mask_loss': alpha_m}),
    lambda chunk: chunk.update({"predicted_presence": tf.nn.sigmoid(chunk["predicted_presence"])}) if estimate_presence else None,
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
