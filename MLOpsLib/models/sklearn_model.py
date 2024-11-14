# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import abc
import pickle
from pathlib import Path

import torch

from MLOpsLib import DataBundle

from .base import BaseModel


class SklearnModel(BaseModel, abc.ABC):

    def __init__(self, workspace: str = None, device="cpu"):
        super(SklearnModel, self).__init__(workspace)
        self.file_path = Path(f"{workspace}/model.pkl")
        self.device = device

    def fit(self, dataset: DataBundle) -> None:
        dataset.pd_to_pttensor()
        feature = dataset.data.feature
        feature = feature.view(len(feature), -1)
        self.model.fit(feature, dataset.data.label.to(self.device))

        # Dump models
        with open(self.file_path, "wb") as f:
            pickle.dump(self.model, f)

    def predict(self, dataset, data_type: str = "test") -> torch.Tensor:
        dataset = dataset.to("cpu")
        dataset = dataset.view(len(dataset), -1)
        scores = self.model.predict(dataset.numpy())

        scores = torch.from_numpy(scores).to(self.device).view(-1)
        dataset = dataset.to(self.device)
        return scores

    def dump_checkpoint(self, path: str):
        with open(path, "wb") as fout:
            pickle.dump(self.model, fout)

    def load_checkpoint(self, path: str):
        with open(path, "rb") as fin:
            self.model = pickle.load(fin)
