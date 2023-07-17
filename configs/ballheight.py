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
side_length = 224
output_shape = (side_length, side_length)

estimate_diameter = False
estimate_presence = False
full_dataset = False
with_diff = False
draw_vertical_cues = False

# DeepSport Dataset
dataset_name = f"ballsize_dataset_640_{'with' if estimate_presence else 'no'}_detections.pickle"
scale = .5
size_min = 14*scale
size_max = 40*scale
max_shift = 0
transforms = [
    deepsport_utilities.ds.instants_dataset.BallViewRandomCropperTransform(
        output_shape=output_shape,
        size_min=size_min,
        size_max=size_max,
        margin=side_length//2-max_shift,
        padding=int(side_length/scale),
        on_ball=True,
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddNextImageFactory() if with_diff else None,
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallSizeFactory(origins=['annotation', 'interpolation', 'ballseg']) if estimate_diameter else None,
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallHeightFactory(origins=['annotation', 'interpolation', 'ballseg']),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddIsBallTargetFactory() if estimate_presence else None,
    )
]

dataset_splitter_str = 'deepsport'
validation_pc = 15
testing_arena_labels = []
dataset = experimentator.CachedPickledDataset(find(dataset_name))
dataset = mlwf.FilteredDataset(dataset, lambda k: full_dataset or bool(isinstance(k[0], deepsport_utilities.ds.instants_dataset.InstantKey) and "KS-FR-" in k.arena_label))
dataset = mlwf.FilteredDataset(dataset, lambda k,v: estimate_presence or v.ball.origin in ['annotation', 'interpolation'])
globals().update(locals()) # required to use locals() in lambdas
dataset = mlwf.TransformedDataset(dataset, transforms)

fold = 0

dataset_splitter = {
    "deepsport": deepsport_utilities.ds.instants_dataset.DeepSportDatasetSplitter(additional_keys_usage='training' if full_dataset else 'skip', validation_pc=validation_pc),
    "arenas_specific": deepsport_utilities.ds.instants_dataset.dataset_splitters.TestingArenaLabelsDatasetSplitter(testing_arena_labels, validation_pc=validation_pc),
}[dataset_splitter_str]
subsets = dataset_splitter(dataset, fold)
testing_arena_labels = dataset_splitter.testing_arena_labels

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

alpha = 0.5 if estimate_presence else 0
beta = 0.5 if estimate_diameter else 0
globals().update(locals()) # required to use locals() in lambdas
chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image", "batch_input_image2"]),
    lambda chunk: chunk.update({'batch_input_diff': tf.subtract(chunk["batch_input_image"], chunk["batch_input_image2"])}) if with_diff else None,
    models.other.GammaAugmentation("batch_input_image"),
    lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"] if not with_diff else tf.concat((chunk["batch_input_image"], chunk["batch_input_diff"]), axis=3)}),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    models.tensorflow.SixChannelsTensorflowBackbone("vgg16.VGG16", include_top=False),
    models.other.LeNetHead(output_features=3),
    tasks.ballheight.HeightEstimationNamedOutputs(),
    models.other.BinaryCrossEntropyLoss(y_true="batch_is_ball", y_pred="predicted_is_ball", name="classification") if estimate_presence else None,
    models.other.HuberLoss(y_true='batch_ball_height', y_pred='predicted_height', name='height_regression'),
    models.other.HuberLoss(y_true='batch_ball_size', y_pred='predicted_diameter', name='size_regression') if estimate_diameter else None,
    lambda chunk: chunk.update({"regression_loss": (1-beta)*chunk["height_regression_loss"] + beta*chunk["size_regression_loss"] if estimate_diameter else chunk["height_regression_loss"]}),
    lambda chunk: chunk.update({"loss": (alpha*chunk["classification_loss"] + (1-alpha)*chunk["regression_loss"]) if estimate_presence else chunk["regression_loss"]}),
    lambda chunk: chunk.update({"predicted_is_ball": tf.nn.sigmoid(chunk["predicted_is_ball"])}) if estimate_presence else None,
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
