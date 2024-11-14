import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from MLOpsLib.builders import DATA_TRANSFORMATIONS
from MLOpsLib.feature.base import BaseFeatureTransformer


@DATA_TRANSFORMATIONS.register()
class OneHotEncoderSK(BaseFeatureTransformer):
    def __init__(
        self,
        workspace: str,
        features: list,
        config: dict = None,
        checkpoint: bool = False,
        inference: bool = False,
    ):
        """
        Initializes the OneHotEncoder transformer.
        """
        super().__init__(workspace, features, checkpoint, inference)
        self.config = config

        if self.inference:
            self.encoder = self.load_model()
        else:
            self.encoder = OneHotEncoder(sparse=False)

    def _apply_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.inference:
            X_transformed = pd.DataFrame(
                self.encoder.fit_transform(X),
                columns=self.encoder.get_feature_names_out(self.features),
                index=X.index,
            )
            self.save_model(self.encoder)  # Save the encoder after fitting
        else:
            X_transformed = pd.DataFrame(
                self.encoder.transform(X),
                columns=self.encoder.get_feature_names_out(self.features),
                index=X.index,
            )
        return X_transformed
