from pathlib import Path

import torch
import torch.nn as nn

from MLOpsLib.builders import MODELS
from MLOpsLib.models.nn_model import NNModel


class ANNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


@MODELS.register()
class ANNRULPredictor(NNModel):  # Assuming PyTorchModel base class for PyTorch models
    def __init__(
        self, *args, workspace: str = None, config: dict, device="cpu", **kwargs
    ):
        NNModel.__init__(self, workspace=workspace, device=device)
        input_size = config.get("input_size", 10)
        hidden_size = config.get("hidden_size", 64)
        output_size = config.get("output_size", 1)
        self.model = ANNModel(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size
        )
        self.path = Path(f"workspaces/{self.workspace}/models")
        self.file_path = Path(f"workspaces/{self.workspace}/model/ANN.pth")

    def save(self):
        torch.save(self.model.state_dict(), self.file_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.file_path))
