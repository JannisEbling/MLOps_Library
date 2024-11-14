# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from MLOpsLib.utils.registry import Registry

DATA_INGESTION = Registry("Data")
MODELS = Registry("Models")
DATA_TRANSFORMATIONS = Registry("Data Transformations")
TRAIN_TEST_SPLITTERS = Registry("Train Test Splitters")
EVALUATORS = Registry("Evaluators")
