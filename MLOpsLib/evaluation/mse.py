import numpy as np
import torch

from MLOpsLib.builders import EVALUATORS
from MLOpsLib.evaluation.base import BaseEvaluator


@EVALUATORS.register()
class MSE(BaseEvaluator):
    def __init__(self, *args, config: dict, workspace: str = None, **kwargs):
        BaseEvaluator.__init__(self, config=config, workspace=workspace, **kwargs)
        self.name = "MSE"

    def calc_metric(self, y_pred, y_true, **kwargs):
        mse = torch.mean((y_pred - y_true) ** 2)
        return mse
