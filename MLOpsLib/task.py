# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from MLOpsLib.builders import (
    DATA_INGESTION,
    DATA_TRANSFORMATIONS,
    MODELS,
    TRAIN_TEST_SPLITTERS,
)
from MLOpsLib.data import DataBundle


class Task:

    def __init__(self, config: dict):
        self.config = config
        data = self.config["data"]
        train_test_splitter = self.config["train_test_split"]
        basic_feature_transformation = self.config["basic_feature_transformation"]
        specific_feature_transformation = self.config["specific_feature_transformation"]
        model = self.config["model"]
        if isinstance(self.config, dict):
            dataingestor = DATA_INGESTION.build(
                self.config, "data_ingestion", self.config["workspace"]
            )[0]
        if isinstance(self.config, dict):
            train_test_splitter = TRAIN_TEST_SPLITTERS.build(
                self.config, "train_test_split", self.config["workspace"]
            )[0]
        if isinstance(self.config, dict):
            data_preperation = DATA_TRANSFORMATIONS.build(
                self.config, "basic_feature_transformation", self.config["workspace"]
            )
        if isinstance(self.config, dict):
            data_preperation = DATA_TRANSFORMATIONS.build(
                self.config, "specific_feature_transformation", self.config["workspace"]
            )

        self.dataingestor = dataingestor
        self.train_test_splitter = train_test_splitter
        self.basic_feature_transformation = basic_feature_transformation
        self.specific_feature_transformation = specific_feature_transformation
        self.model = model
        self.stage = self.get_current_stage(self.config)
        self.cache_file = self.get_cache_file(config)

    def build(self) -> DataBundle:

        if self.stage <= 1:
            raw_data = self.dataingestor.get_data()
            save_checkpoint()
            create_data_schema()
        elif self.stage == 2:
            raw_data = load_checkpoint()

        if self.stage <= 2:
            X_train, X_test, y_train, y_test = self.train_test_splitter(raw_data)
            save_checkpoint()
        elif self.stage == 3:
            X_train, X_test, y_train, y_test = load_checkpoint()

        if self.stage <= 3:
            for transformation in self.basic_feature_transformation:
                X_train, X_test = transformation.transform_data(X_train, X_test)
            X_train_base, X_test_base = X_train, X_test
            save_checkpoint()
        elif self.stage == 4:
            X_train_base, X_test_base, y_train, y_test = load_checkpoint()
        if self.stage <= 4:
            for transformation in self.specific_feature_transformation:
                X_train_base, X_test_base = transformation.transform_data(
                    X_train_base, X_test_base
                )
            X_train_spec, X_test_spec = X_train_base, X_test_base
            save_checkpoint()
        elif self.stage == 5:
            X_train_spec, X_test_spec, y_train, y_test = load_checkpoint()
        self.stage = 5

        dataset = DataBundle(
            X_train_spec,
            y_train,
            X_test_spec,
            y_test,
            basic_feature_transformation=self.basic_feature_transformation,
            specific_feature_transformation=self.specific_feature_transformation,
        )

        return dataset

    def load_checkpoint(self, configs):
        cache_file = self.get_cache_file(configs)
        pass

    def save_checkpoint(self, configs):
        cache_file = self.get_cache_file(configs)
        pass

    def get_current_stage(self, configs: dict):
        if self.check_cache(configs):
            stage = 5
        else:
            if self.check_cache(configs):
                stage = 4
            else:
                if self.check_cache(configs):
                    stage = 3
                else:
                    if self.check_cache(configs):
                        stage = 2
                    else:
                        stage = 1
        return stage

    def check_cache(self, configs: dict):
        cache_file = self.get_cache_file(configs)
        return cache_file.exists()

    def get_cache_file(self, configs: dict):
        strings = []
        config_fields = CONFIG_FIELDS
        for field in config_fields[: self.stage]:
            strings.append(recursive_dump_string(configs[field]))

        filename = hash_string("+".join(strings))
        cache_dir = Path("artifacts/" + config_fields[self.stage - 1])
        if not cache_dir.exists():
            cache_dir.mkdir()
        cache_file = Path(cache_dir / f"cache_{filename}.pkl")
        return cache_file


def recursive_dump_string(data):
    if isinstance(data, list):
        return "_".join([recursive_dump_string(x) for x in data])
    if isinstance(data, dict):
        return "_".join(
            [recursive_dump_string(data[key]) for key in sorted(data.keys())]
        )
    return str(data)


def hash_string(string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(string.encode("utf-8"))
    hash_value = sha256_hash.hexdigest()
    truncated_hash = hash_value[:32]
    return truncated_hash


def timestamp(marker: bool = False):
    template = "%Y-%m-%d %H:%M:%S" if marker else "%Y%m%d%H%M%S"
    return datetime.now().strftime(template)


CONFIG_FIELDS = [
    "data",
    "train_test_split",
    "basic_feature_transformation",
    "specific_feature_transformation",
    "model",
]
