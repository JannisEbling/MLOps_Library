import pandas as pd
from sklearn.preprocessing import StandardScaler

from MLOpsLib.builders import DATA_TRANSFORMATIONS
from MLOpsLib.feature.base import BaseFeatureTransformer


@DATA_TRANSFORMATIONS.register()
class StandardScalerSK(BaseFeatureTransformer):
    def __init__(
        self,
        workspace: str,
        features: list,
        config: dict = None,
        checkpoint: bool = False,
        inference: bool = False,
    ):
        """
        Initializes the StandardScaler transformer.
        Inherits generic methods for model saving, feature handling from BaseFeatureTransformer.
        """
        super().__init__(workspace, features, config, checkpoint, inference)
        self.config = config
        self.name = "standardscaler"

        # Check if we are in inference mode, load an existing scaler
        if self.inference:
            self.scaler = self.load_model()
        else:
            self.scaler = StandardScaler()

    def _apply_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the StandardScaler transformation to the selected features.
        """
        if not self.inference:
            # Fit the scaler during training
            X_transformed = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index,
            )
            self.save_model(self.scaler)  # Save the scaler after fitting
        else:
            # During inference, transform using the loaded scaler
            X_transformed = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index,
            )
        return X_transformed
