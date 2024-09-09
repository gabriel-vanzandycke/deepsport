
from experimentator.tf2_experiment import TensorflowExperiment





class DetectScoreboardsExperiment(TensorflowExperiment):
    batch_inputs_names =  ["batch_input_image", "epoch", "batch_target"]
    batch_metrics_names = [""]
    batch_outputs_names = ["batch_heatmap"]


