import tensorflow as tf
import mlworkflow as mlwf
import experimentator
from experimentator import find
import experimentator.tf2_experiment
import experimentator.wandb_experiment
import deepsport_utilities.utils
import deepsport_utilities.ds.instants_dataset
from dataset_utilities.ds.raw_sequences_dataset import BallState
import deepsport_utilities.ds.instants_dataset.dataset_splitters
import tasks.ballstate
import tasks.ballsize
import tasks.ballistic
import tasks.classification
import models.other
import models.tensorflow

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.ballistic.BallStateAndBallSizeExperiment,
]

batch_size = 16
side_length = 114
output_shape = (side_length, side_length)

unconfident_margin = .2
data_extraction = deepsport_utilities.transforms.DataExtractorTransform(
    deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
    #deepsport_utilities.ds.instants_dataset.views_transforms.AddNextImageFactory(),
    deepsport_utilities.ds.instants_dataset.views_transforms.AddBallPositionFactory(),
    deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
    tasks.ballstate.AddBallSizeFactory(),
    tasks.ballstate.AddBallStateFactory(),
    tasks.ballstate.AddIsBallTargetFactory(unconfident_margin=unconfident_margin),
)
testing_arena_labels = ["KS-FR-ROANNE", "KS-FR-LILLE", "KS-FR-GRAVELINES", "KS-FR-STRASBOURG"]
max_shift = 10 # pixels in the ouptut image


# InstantsDataset
#instants_dataset_name = "instants_dataset.pickle"
#instants_dataset = mlwf.PickledDataset(find(instants_dataset_name))
#experiment.tasks.ball.BallViewRandomCropperTransform(output_shape=output_shape, def_min=def_min if scaled else 0, def_max=def_max if scaled else 0, on_ball=on_ball, padding=0, margin=100//scale),




# Ball Size estimation dataset
size_min = 14
size_max = 45
ballsize_dataset_name = "ballsize_dataset.pickle"
ballsize_dataset = mlwf.PickledDataset(find(ballsize_dataset_name))
ballsize_dataset = mlwf.TransformedDataset(ballsize_dataset, [
    deepsport_utilities.ds.instants_dataset.BallCropperTransform(
        output_shape=output_shape,
        size_min=size_min,
        size_max=size_max,
        margin=side_length//2-max_shift
    ), data_extraction
])
ballsize_subsets = deepsport_utilities.ds.instants_dataset.TestingArenaLabelsDatasetSplitter(testing_arena_labels)(ballsize_dataset)


# Ball State estimation dataset
scale_min = 0.75
scale_max = 1.25
ballstate_dataset_name = "ballstate_dataset.pickle"
ballstate_dataset = mlwf.PickledDataset(find(ballstate_dataset_name))
ballstate_dataset = mlwf.TransformedDataset(ballstate_dataset, [
    deepsport_utilities.ds.instants_dataset.BallCropperTransform(
        output_shape=output_shape,
        scale_min=scale_min,
        scale_max=scale_max,
        margin=side_length//2-max_shift
    ), data_extraction
])
ballstate_subsets = deepsport_utilities.ds.instants_dataset.TestingArenaLabelsDatasetSplitter(testing_arena_labels)(ballstate_dataset)

# Combine subsets
subsets = [
    experimentator.CombinedSubset('training', ballsize_subsets[0], ballstate_subsets[0]),
    experimentator.CombinedSubset('validation', ballsize_subsets[1], ballstate_subsets[1]),
    experimentator.CombinedSubset('testing', ballsize_subsets[2], ballstate_subsets[2]),
]

globals().update(locals()) # required to use tf in lambdas or simply 'BallState'
classes = [BallState(i) for i in range(4)]
callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.LearningRateDecay(start=range(50,101,10), duration=2, factor=.5),
    #experimentator.LearningRateWarmUp(),
    tasks.classification.ComputeClassifactionMetrics(),
    tasks.classification.ComputeConfusionMatrix(classes=classes),
    tasks.ballsize.ComputeDiameterError(),
    tasks.ballsize.ComputeDetectionMetrics(),
    tasks.ballstate.ExtractClassificationMetrics(class_name=str(BallState(1)), class_index=1),
    experimentator.wandb_experiment.LogStateWandB(),
]

projector = "conv2d"
projector_network = {
    "None": None,
    "1layer": tasks.ballstate.ChannelsReductionLayer(),
    "conv2d": lambda chunk: chunk.update({"batch_input": tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='SAME')(chunk["batch_input"])})
}[projector]

backbone = "VGG"
pretrained = True
backbone_model = {
    "VGG": models.tensorflow.TensorflowBackbone("vgg16.VGG16", include_top=False, weights='imagenet' if pretrained else None),
    "RN50": models.tensorflow.TensorflowBackbone("resnet50.ResNet50", include_top=False, weights='imagenet' if pretrained else None),
}[backbone]


alpha = [1.0, 1.0, 1.0]
chunk_processors = [
    experimentator.tf2_chunk_processors.CastFloat(tensor_names=["batch_input_image", "batch_input_image2"]),
    lambda chunk: chunk.update({'batch_input_diff': tf.subtract(chunk["batch_input_image"], chunk["batch_input_image2"])}),
    models.other.GammaAugmentation("batch_input_image"),
    lambda chunk: chunk.update({"batch_input": tf.concat((chunk["batch_input_image"], chunk["batch_input_diff"]), axis=3)}),
    experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    projector_network,
    backbone_model,
    models.other.LeNetHead(output_features=2+len(classes)), # Diameter Regression + Has Ball Classification + Ball State Classification
    tasks.ballstate.NamedOutputs(),
    tasks.ballsize.ClassificationLoss(),
    tasks.ballsize.RegressionLoss(),
    tasks.ballstate.CombineLosses(["classification_loss", "regression_loss", "state_loss"], alpha),
    lambda chunk: chunk.update({"predicted_is_ball": tf.nn.sigmoid(chunk["predicted_is_ball"])}), # squash in [0,1]
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
