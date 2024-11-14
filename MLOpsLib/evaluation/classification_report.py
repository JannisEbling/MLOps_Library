import torch
from sklearn.metrics import classification_report

from MLOpsLib.builders import EVALUATORS
from MLOpsLib.evaluation.base import BaseEvaluator


@EVALUATORS.register()
class ClassificationReport(BaseEvaluator):
    def __init__(self, *args, config: dict, workspace: str = None, **kwargs):
        BaseEvaluator.__init__(self, config=config, workspace=workspace, **kwargs)
        self.name = "Classification Report"

    def calc_metric(self, y_pred, y_true, **kwargs):

        predicted_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
        true_labels = y_true.cpu().numpy()

        report = classification_report(true_labels, predicted_labels, output_dict=True)
        return report
