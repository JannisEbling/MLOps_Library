# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from pathlib import Path

from sklearn.linear_model import Ridge

from MLOpsLib.builders import MODELS
from MLOpsLib.models.sklearn_model import SklearnModel


@MODELS.register()
class RidgeRULPredictor(SklearnModel):
    def __init__(
        self, *args, workspace: str = None, config: dict, device="cpu", **kwargs
    ):
        SklearnModel.__init__(self, workspace=workspace, device=device)
        self.model = Ridge(**kwargs)
        self.path = Path(f"{self.workspace}/models")
        self.file_path = Path(f"{self.workspace}/model/Ridge.pkl")
