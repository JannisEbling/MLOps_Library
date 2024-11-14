import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from MLOpsLib.builders import DATA_TRANSFORMATIONS
from MLOpsLib.feature.base import BaseFeatureTransformer


@DATA_TRANSFORMATIONS.register()
class PolynomialFeaturesSK(BaseFeatureTransformer):
    def __init__(
        self,
        workspace: str,
        features: list,
        degree: int = 2,
        config: dict = None,
        checkpoint: bool = False,
        inference: bool = False,
    ):
        """
        Initializes the PolynomialFeatures transformer.
        """
        super().__init__(workspace, features, checkpoint, inference)
        self.config = config
        self.degree = degree

        if self.inference:
            self.poly = self.load_model()
        else:
            self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)

    def _apply_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.inference:
            X_transformed = pd.DataFrame(self.poly.fit_transform(X), index=X.index)
            self.save_model(self.poly)  # Save the polynomial features after fitting
        else:
            X_transformed = pd.DataFrame(self.poly.transform(X), index=X.index)
        return X_transformed
