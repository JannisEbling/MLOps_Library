import abc
import hashlib
import pickle
from pathlib import Path
from typing import List, Union

import pandas as pd


class BaseFeatureTransformer(abc.ABC):
    def __init__(
        self,
        workspace: str,
        features: list,
        config: dict,
        checkpoint: bool = False,
        inference: bool = False,
    ):
        """
        Initializes the base feature transformer.
        """
        self.features = features
        self.inference = inference
        self.workspace = workspace
        self.checkpoint = checkpoint

        # Set up model directory and filename for saving/loading models
        workspace_path = Path(config["workspace"])
        stage = "data_preparation"
        self.models_dir = workspace_path / stage
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_filename = (
            self.models_dir
            / f"model_{self.hash_string(self.recursive_dump_string(features))}.pkl"
        )

    def save_model(self, model) -> None:
        """
        Saves the fitted model (scaler/transformer) to disk.
        """
        with open(self.model_filename, "wb") as f:
            pickle.dump(model, f)

    def load_model(self) -> Union[None, object]:
        """
        Loads a model (scaler/transformer) from disk if it exists.
        """
        if self.model_filename.exists():
            with open(self.model_filename, "rb") as f:
                return pickle.load(f)
        return None

    def process_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Selects features to transform and returns the modified DataFrame.
        """
        X_selected_features = X[self.features]  # Features to transform
        X_other_features = X.drop(self.features, axis=1)  # Features to leave untouched
        return X_selected_features, X_other_features

    def transform_data(self, X: pd.DataFrame, y: pd.Series):
        """
        General transform method that applies the feature transformation.
        Delegates the specific transformation logic to the subclass.
        """
        # Select and process features
        X_selected_features, X_other_features = self.process_features(X)

        # Perform the actual transformation (implemented in subclass)
        X_transformed = self._apply_transformation(X_selected_features)

        # Recombine transformed and untransformed features
        X_combined = pd.concat([X_transformed, X_other_features], axis=1)

        return X_combined, y

    @abc.abstractmethod
    def _apply_transformation(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply the transformation to selected features.
        Must be implemented by subclasses.
        """
        pass

    def recursive_dump_string(self, data):
        """
        Recursively dumps lists or dictionaries into a string format.
        Used for generating a hashable string from configuration data.
        """
        if isinstance(data, list):
            return "_".join([self.recursive_dump_string(x) for x in data])
        if isinstance(data, dict):
            return "_".join(
                [self.recursive_dump_string(data[key]) for key in sorted(data.keys())]
            )
        return str(data)

    def hash_string(self, string: str) -> str:
        """
        Hashes a string using SHA-256 and returns a truncated hash (first 32 characters).
        """
        sha256_hash = hashlib.sha256()
        sha256_hash.update(string.encode("utf-8"))
        hash_value = sha256_hash.hexdigest()
        return hash_value[:32]
