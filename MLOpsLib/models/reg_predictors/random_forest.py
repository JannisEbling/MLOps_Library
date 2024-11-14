from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from MLOpsLib.builders import MODELS
from MLOpsLib.models.sklearn_model import SklearnModel


@MODELS.register()
class RandomForestRULPredictor(SklearnModel):
    def __init__(
        self, *args, workspace: str = None, config: dict, device="cpu", **kwargs
    ):
        SklearnModel.__init__(self, workspace=workspace, device=device)
        self.model = RandomForestRegressor(**kwargs)
        self.path = Path(f"workspaces/{self.workspace}/models")
        self.file_path = Path(f"workspaces/{self.workspace}/model/RandomForest.pkl")
