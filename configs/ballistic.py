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
import tasks.detection
import models.other
import models.tensorflow

experiment_type = [
    experimentator.AsyncExperiment,
    experimentator.CallbackedExperiment,
    experimentator.tf2_experiment.TensorflowExperiment,
    tasks.ballistic.BallStateAndBallSizeExperiment,
]

batch_size = 16
#testing_arena_labels = ["KS-FR-ROANNE", "KS-FR-LILLE", "KS-FR-GRAVELINES", "KS-FR-STRASBOURG"]
testing_arena_labels = ['KS-FR-LEMANS', 'KS-FR-MONACO', 'KS-FR-STRASBOURG', 'KS-FR-CAEN', 'KS-FR-LIMOGES', 'KS-FR-BLOIS', 'KS-FR-FOS', 'KS-FR-GRAVELINES', 'KS-FR-STCHAMOND', 'KS-FR-POITIERS']
dataset_name = '/home/ucl/elen/gvanzand/DeepSport/datasets/balls_dataset.pickle'

side_length = 114
output_shape = (side_length, side_length)
size_min = 9
size_max = 28
scale_min = 0.75
scale_max = 1.25
max_shift = 10 # pixels in the ouptut image

unconfident_margin = .1

dataset = mlwf.PickledDataset(find(dataset_name))
# transformation should be a function of a scale factor if I want to evaluate on
# strasbourg and gravelines sequences (in order to evaluate on the scale range trained on)
dataset = mlwf.TransformedDataset(dataset, [
    tasks.ballistic.BallViewRandomCropperTransformCompat(
        output_shape=output_shape,
        scale_min=scale_min,
        scale_max=scale_max,
        size_min=size_min,
        size_max=size_max,
        margin=side_length//2-max_shift
    ),
    deepsport_utilities.transforms.DataExtractorTransform(
        deepsport_utilities.ds.instants_dataset.views_transforms.AddImageFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddNextImageFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddCalibFactory(),
        deepsport_utilities.ds.instants_dataset.views_transforms.AddBallStateFactory(),
        tasks.ballistic.AddBallSizeFactory(),
        tasks.ballistic.AddBallPresenceFactory(unconfident_margin=unconfident_margin),
    )
])

subsets = deepsport_utilities.ds.instants_dataset.TestingArenaLabelsDatasetSplitter(testing_arena_labels, repetitions=1)(dataset)


#raise
globals().update(locals()) # required to use tf in lambdas or simply 'BallState' in list
classes = [BallState.NONE, BallState.FLYING, BallState.CONSTRAINT, BallState.DRIBBLING]
callbacks = [
    experimentator.AverageMetrics([".*loss"]),
    experimentator.SaveWeights(),
    experimentator.SaveLearningRate(),
    experimentator.GatherCycleMetrics(),
    experimentator.LogStateDataCollector(),
    experimentator.LearningRateDecay(start=range(50,101,10), duration=2, factor=.5),
    #experimentator.LearningRateWarmUp(),
    tasks.ballistic.ComputeClassifactionMetrics(),
    tasks.ballistic.ComputeConfusionMatrix(classes=classes),
    tasks.ballsize.ComputeDiameterError(),
    tasks.ballistic.ComputeDetectionMetrics(origins=['ballseg']),
    tasks.detection.AuC("top1-AuC", "top1_metrics"),
    tasks.detection.AuC("initial_top1-AuC", "initial_top1_metrics"),
    tasks.classification.ExtractClassificationMetrics(class_name=str(BallState(1)), class_index=1),
    tasks.classification.ExtractClassificationMetrics(class_name=str(BallState(2)), class_index=2),
    experimentator.wandb_experiment.LogStateWandB(),
]

projector = "conv2d"
projector_layers = {
    "nodiff": [
        lambda chunk: chunk.update({"batch_input": chunk["batch_input_image"]}),
        experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
    ],
    "1layer": [
        lambda chunk: chunk.update({"batch_input": tf.concat((chunk["batch_input_image"], chunk["batch_input_diff"]), axis=3)}),
        experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
        tasks.ballstate.ChannelsReductionLayer()
    ],
    "conv2d": [
        lambda chunk: chunk.update({"batch_input": tf.concat((chunk["batch_input_image"], chunk["batch_input_diff"]), axis=3)}),
        experimentator.tf2_chunk_processors.Normalize(tensor_names=["batch_input"]),
        lambda chunk: chunk.update({"batch_input": tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='SAME')(chunk["batch_input"])})
    ]
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
    *projector_layers,
    backbone_model,
    models.other.LeNetHead(output_features=2+len(classes)), # Diameter Regression + Has Ball Classification + Ball State Classification
    tasks.ballistic.NamedOutputs(),
    tasks.ballistic.ClassificationLoss(),
    tasks.ballsize.RegressionLoss(),
    tasks.ballstate.StateClassificationLoss(classes),
    tasks.ballstate.CombineLosses(["classification_loss", "regression_loss", "state_loss"], alpha),
    lambda chunk: chunk.update({"predicted_presence": tf.nn.sigmoid(chunk["predicted_presence"])}), # squash in [0,1]
]

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
