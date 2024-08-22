
from experimentator.tf_experiment import TensorflowExperiment





class DetectScoreboardsExperiment(TensorflowExperiment):
    batch_inputs_names =  ["batch_ball_presence", "batch_ball_size", "batch_input_image", "epoch", "batch_ball_position"]
    batch_metrics_names = ["predicted_presence", "predicted_height", "predicted_diameter", "regression_loss", "classification_loss", "mask_loss", "offset_loss"]
    batch_outputs_names = ["predicted_diameter", "predicted_presence", "predicted_height", "predicted_mask"]
