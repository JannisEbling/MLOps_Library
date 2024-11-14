import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from MLOpsLib.builders import DATA_TRANSFORMATIONS
from MLOpsLib.feature.base import BaseFeatureTransformer


@DATA_TRANSFORMATIONS.register()
class MinMaxScalerSK(BaseFeatureTransformer):
    def __init__(
        self,
        workspace: str,
        features: list,
        config: dict = None,
        checkpoint: bool = False,
        inference: bool = False,
    ):
        """
        Initializes the MinMaxScaler transformer.
        """
        super().__init__(workspace, features, checkpoint, inference)
        self.config = config

        if self.inference:
            self.scaler = self.load_model()
        else:
            self.scaler = MinMaxScaler()

    def _apply_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.inference:
            X_transformed = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index,
            )
            self.save_model(self.scaler)  # Save the scaler after fitting
        else:
            X_transformed = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index,
            )
        return X_transformed
