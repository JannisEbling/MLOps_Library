# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import random

from sklearn.model_selection import train_test_split

from MLOpsLib.builders import TRAIN_TEST_SPLITTERS


@TRAIN_TEST_SPLITTERS.register()
class RandomTrainTestSplitter:

    def __init__(
        self,
        workspace: str,
        config: dict,
        label: str,
        test_size: float = 0.25,
        random_state=None,
    ):
        """
        Initializes the DataSplitter with feature matrix X and target variable y.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int, optional): Seed for random number generator.
        """
        self.workspace = workspace
        self.label = label
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, data):
        """Splits the data into training and testing sets."""
        self.X = data.drop(self.label, axis=1)
        self.y = data[self.label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_train_data(self):
        """Returns the training features and labels."""
        return self.X_train, self.y_train

    def get_test_data(self):
        """Returns the testing features and labels."""
        return self.X_test, self.y_test
