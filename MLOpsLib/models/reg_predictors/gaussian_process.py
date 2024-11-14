from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor

from MLOpsLib.builders import MODELS
from MLOpsLib.models.sklearn_model import SklearnModel


@MODELS.register()
class GaussianProcessRULPredictor(SklearnModel):
    def __init__(
        self, *args, workspace: str = None, config: dict, device="cpu", **kwargs
    ):
        SklearnModel.__init__(self, workspace=workspace, device=device)
        self.model = GaussianProcessRegressor(**kwargs)
        self.path = Path(f"workspaces/{self.workspace}/models")
        self.file_path = Path(f"workspaces/{self.workspace}/model/GaussianProcess.pkl")
