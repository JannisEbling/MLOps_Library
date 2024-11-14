from pathlib import Path

from sklearn.svm import SVR

from MLOpsLib.builders import MODELS
from MLOpsLib.models.sklearn_model import SklearnModel


@MODELS.register()
class SVMRULPredictor(SklearnModel):
    def __init__(
        self, *args, workspace: str = None, config: dict, device="cpu", **kwargs
    ):
        SklearnModel.__init__(self, workspace=workspace, device=device)
        self.model = SVR(**kwargs)
        self.path = Path(f"workspaces/{self.workspace}/models")
        self.file_path = Path(f"workspaces/{self.workspace}/model/SVM.pkl")
