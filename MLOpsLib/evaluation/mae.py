import torch

from MLOpsLib.builders import EVALUATORS
from MLOpsLib.evaluation.base import BaseEvaluator


@EVALUATORS.register()
class MAE(BaseEvaluator):
    def __init__(self, *args, config: dict, workspace: str = None, **kwargs):
        BaseEvaluator.__init__(self, config=config, workspace=workspace, **kwargs)
        self.name = "MAE"

    def calc_metric(self, y_pred, y_true, **kwargs):
        mae = torch.mean(torch.abs(y_pred - y_true))
        return mae
