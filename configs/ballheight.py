import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator import find
import experimentator.tf2_experiment
import deepsport_utilities.ds.instants_dataset
import models.other
import models.tensorflow
import experimentator.wandb_experiment
import tasks.ballheight
import tasks.detection

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.ballheight.BallHeightEstimation
]

batch_size = 16

# Dataset parameters
scale = 1
side_length = 512
output_shape = (int(side_length*scale), int(side_length*scale))

estimate_diameter = False
estimate_presence = False
with_diff = False
draw_vertical_cues = False
public_dataset = True
ballistic_dataset = False
correct_distortion = False
assert (ballistic_dataset is False) or (public_dataset is False), "annotated ballistic sequences from raw sequences dataset cannot be used when training on public dataset"

# DeepSport Dataset
dataset_name = {
    (True, False): "ballsize_dataset_1000_with_detections_from_model_trained_on_full_dataset.pickle",
    (True, True): "ballsize_dataset_1000_with_detections_from_model_trained_on_small_dataset.pickle",
    (False, True): "ballsize_dataset_1000_no_detections.pickle",
    (False, False): "ballsize_dataset_1000_no_detections.pickle",
}[(estimate_presence, public_dataset)]

max_shift = 0

data_extractor_transform = deepsport_utilities.transforms.DataExtractorTransform(
    deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
    deepsport_utilities.ds.instants_dataset.views_transforms.AddNextImageFactory() if with_diff else None,
    deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSizeFactory(origins=['annotation', 'interpolation', 'ballseg']) if estimate_diameter else None,
    deepsport_utilities.ds.instants_dataset.views_transforms.AddBallHeightFactory(origins=['annotation', 'interpolation', 'ballseg']),
    deepsport_utilities.ds.instants_dataset.views_transforms.AddBallFactory(),
    deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
    deepsport_utilities.ds.instants_dataset.views_transforms.AddBallPresenceFactory() if estimate_presence else None,
)

fixed_scale_cropper_transform = deepsport_utilities.ds.instants_dataset.BallViewRandomCropperTransform(
    output_shape=output_shape,
    scale_min=1*scale,
    scale_max=1*scale,
    margin=int(side_length*scale)//2-max_shift,
    padding=int(side_length*scale),
    on_ball=True,
    rectify=correct_distortion,
)

random_scale_cropper_transform = deepsport_utilities.ds.instants_dataset.BallViewRandomCropperTransform(
    output_shape=output_shape,
    scale_min=0.75*scale,
    scale_max=1.25*scale,
    margin=int(side_length*scale)//2-max_shift,
    padding=int(side_length*scale),
    on_ball=True,
    rectify=correct_distortion,
)

random_size_cropper_transform = deepsport_utilities.ds.instants_dataset.BallViewRandomCropperTransform(
    output_shape=output_shape,
    size_min=14*scale,
    size_max=40*scale,
    margin=side_length//2-max_shift,
    padding=side_length,
    on_ball=True,
    rectify=correct_distortion,
)

dataset = experimentator.CachedPickledDataset(find(dataset_name))
dataset = mlwf.FilteredDataset(dataset, lambda k: ballistic_dataset or bool(isinstance(k[0], deepsport_utilities.ds.instants_dataset.InstantKey)))
dataset = mlwf.FilteredDataset(dataset, lambda k,v: estimate_presence or v.ball.origin in ['annotation', 'interpolation'])
evaluation_dataset_name = "ballistic_ball_views_512.pickle" # "views_from_instant_sequences_dataset_smoothed_ball_annotation.pickle"
evaluation_dataset = experimentator.CachedPickledDataset(find(evaluation_dataset_name, verbose=True))

globals().update(locals()) # required to use locals() in lambdas

repetitions = 1
additional_keys_usage = 'testing2' if public_dataset else 'training'
dataset_splitter = deepsport_utilities.ds.instants_dataset.DeepSportDatasetSplitter(
    additional_keys_usage=additional_keys_usage,
    validation_pc=15,
    repetitions={'testing': repetitions})

fixed_scale_subsets = dataset_splitter(experimentator.dataset.DataAugmentationDataset(dataset, [fixed_scale_cropper_transform, data_extractor_transform]))
random_scale_subsets = dataset_splitter(experimentator.dataset.DataAugmentationDataset(dataset, [random_scale_cropper_transform, data_extractor_transform]))
random_size_subsets = dataset_splitter(experimentator.dataset.DataAugmentationDataset(dataset, [random_size_cropper_transform, data_extractor_transform]))

balancer_str = None
balancer = {
    'two': lambda k,v: 0 if v['ball'].center.z < -200 else 1,
    'None': None,
    None: None
}[balancer_str]

classes = {
    'two': [0, 1],
    'None': None,
    None: None
}[balancer_str]


if balancer is not None:
    random_scale_subsets[0] = experimentator.BalancedSubset.convert(random_scale_subsets[0], balancer, classes)

random_size_subsets[2].name = "legacy_testing"
subsets = [
    random_scale_subsets[0],
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
    experimentator.LearningRateDecay(start=range(20,101,10), duration=2, factor=.5),
    tasks.ballheight.ComputeHeightError(),
    tasks.detection.ComputeDetectionMetrics(origin='ballseg') if estimate_presence else None,
    tasks.detection.AuC("top1-AuC", "top1_metrics") if estimate_presence else None,
    tasks.detection.AuC("top2-AuC", "top2_metrics") if estimate_presence else None,
    tasks.detection.AuC("top4-AuC", "top4_metrics") if estimate_presence else None,
    tasks.detection.AuC("top8-AuC", "top8_metrics") if estimate_presence else None,
    tasks.detection.AuC("initial_top1-AuC", "initial_top1_metrics") if estimate_presence else None,
    experimentator.wandb_experiment.LogStateWandB("validation_MAPE", False),
]

backbone = "vgg16.VGG16"

alpha = 0.5 if estimate_presence else 0
beta = 0.5 if estimate_diameter else 0
globals().update(locals()) # required to use locals() in lambdas
chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image", "batch_input_image2"]),
    lambda chunk: chunk.update({'batch_input_diff': tf.subtract(chunk["batch_input_image"], chunk["batch_input_image2"])}) if with_diff else None,
    models.other.GammaAugmentation("batch_input_image"),
    lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"] if not with_diff else tf.concat((chunk["batch_input_image"], chunk["batch_input_diff"]), axis=3)}),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    models.tensorflow.TensorflowBackbone(backbone, include_top=False),
    models.other.LeNetHead(output_features=3),
    tasks.ballheight.HeightEstimationNamedOutputs(),
    models.other.BinaryCrossEntropyLoss(y_true="batch_ball_presence", y_pred="predicted_presence", name="classification") if estimate_presence else None,
    models.other.HuberLoss(y_true='batch_ball_height', y_pred='predicted_height', name='height_regression'),
    models.other.HuberLoss(y_true='batch_ball_size', y_pred='predicted_diameter', name='size_regression') if estimate_diameter else None,
    lambda chunk: chunk.update({"regression_loss": (1-beta)*chunk["height_regression_loss"] + beta*chunk["size_regression_loss"] if estimate_diameter else chunk["height_regression_loss"]}),
    lambda chunk: chunk.update({"loss": (alpha*chunk["classification_loss"] + (1-alpha)*chunk["regression_loss"]) if estimate_presence else chunk["regression_loss"]}),
    lambda chunk: chunk.update({"predicted_presence": tf.nn.sigmoid(chunk["predicted_presence"])}) if estimate_presence else None,
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
