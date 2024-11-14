import abc
import pickle
from pathlib import Path

import yaml


class BaseEvaluator(abc.ABC):

    def __init__(self, config, workspace: str = None):
        self.config = config
        self.workspace = workspace
        self.dir_path = Path("workspaces") / self.workspace / "evaluation"
        self.metric_path = self.dir_path / "metric.pkl"
        self.config_path = self.dir_path / "config.yaml"
        self.dir_path.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def calc_metric(self, y_pred, y_true):
        """Calculates the metric"""
        pass

    def save_metric(self, y_pred, y_true):
        metric = self.calc_metric(y_pred, y_true)
        with open(self.metric_path, "wb") as f:
            pickle.dump(metric, f)

        with open(self.config_path, "w") as file:
            yaml.dump(self.config, file, default_flow_style=False)
        return metric
