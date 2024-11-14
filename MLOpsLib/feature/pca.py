import pandas as pd
from sklearn.decomposition import PCA

from MLOpsLib.builders import DATA_TRANSFORMATIONS
from MLOpsLib.feature.base import BaseFeatureTransformer


@DATA_TRANSFORMATIONS.register()
class PCASK(BaseFeatureTransformer):
    def __init__(
        self,
        workspace: str,
        features: list,
        n_components: int = 2,
        config: dict = None,
        checkpoint: bool = False,
        inference: bool = False,
    ):
        """
        Initializes the PCA transformer.
        """
        super().__init__(workspace, features, checkpoint, inference)
        self.config = config
        self.n_components = n_components

        if self.inference:
            self.pca = self.load_model()
        else:
            self.pca = PCA(n_components=self.n_components)

    def _apply_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.inference:
            X_transformed = pd.DataFrame(self.pca.fit_transform(X), index=X.index)
            self.save_model(self.pca)  # Save the PCA after fitting
        else:
            X_transformed = pd.DataFrame(self.pca.transform(X), index=X.index)
        return X_transformed
