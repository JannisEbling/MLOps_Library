import torch

from MLOpsLib.builders import EVALUATORS
from MLOpsLib.evaluation.base import BaseEvaluator


@EVALUATORS.register()
class R2(BaseEvaluator):
    def __init__(self, *args, config: dict, workspace: str = None, **kwargs):
        BaseEvaluator.__init__(self, config=config, workspace=workspace, **kwargs)
        self.name = "R²"

    def calc_metric(self, y_pred, y_true, **kwargs):
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
