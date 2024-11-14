import torch

from MLOpsLib.builders import EVALUATORS
from MLOpsLib.evaluation.base import BaseEvaluator


@EVALUATORS.register()
class Accuracy(BaseEvaluator):
    def __init__(self, *args, config: dict, workspace: str = None, **kwargs):
        BaseEvaluator.__init__(self, config=config, workspace=workspace, **kwargs)
        self.name = "Accuracy"

    def calc_metric(self, y_pred, y_true, **kwargs):
        # Assuming y_pred contains raw scores (logits) and needs argmax for classification
        predicted_labels = torch.argmax(y_pred, dim=1)
        correct_predictions = (predicted_labels == y_true).float()
        accuracy = correct_predictions.sum() / len(y_true)
        return accuracy
