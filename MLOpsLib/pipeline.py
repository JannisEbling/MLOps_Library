# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

import hashlib
import os
import pickle
import random
import shutil
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import torch

from MLOpsLib.builders import (
    DATA_INGESTION,
    DATA_TRANSFORMATIONS,
    EVALUATORS,
    MODELS,
    TRAIN_TEST_SPLITTERS,
)
from MLOpsLib.cache import CacheHandler
from MLOpsLib.data import DataBundle
from MLOpsLib.models.base import BaseModel
from MLOpsLib.task import Task
from MLOpsLib.utils import import_config


class Pipeline:
    def __init__(
        self, config_path: Path | str, workspace: Path | str, seed: int = 0, split=True
    ):

        self.config_path = config_path
        self.config = load_config(config_path, workspace)
        self.workspace = self.config["workspace"]
        self.seed = seed
        self.create_directories(self.workspace)
        set_seed(self.seed)
        self.cachehandler = CacheHandler(self.config_path, self.config, 1)
        self.cachehandler.set_current_stage()
        self.split = split

        self.dataingestor = DATA_INGESTION.build(
            self.config, "data_ingestion", self.config["workspace"]
        )[0]
        self.train_test_splitter = TRAIN_TEST_SPLITTERS.build(
            self.config, "train_test_split", self.config["workspace"]
        )[0]
        self.data_preparation = DATA_TRANSFORMATIONS.build(
            self.config, "data_preparation", self.config["workspace"]
        )

    @staticmethod
    def create_directories(workspace):
        sub_dirs = ["data_ingestion", "data_preparation", "model", "train_test_split"]
        os.makedirs(workspace, exist_ok=True)
        for sub_dir in sub_dirs:
            new_dir = Path(workspace) / sub_dir
            os.makedirs(new_dir, exist_ok=True)

    def train(self, model=None, dataset=None):
        if model is None:
            model = self._prepare_model()
        if dataset is None:
            dataset = self.build_dataset()
        model.fit(dataset)
        return model

    def test(self, model=None, dataset=None):
        if model is None:
            model = self._prepare_model()
        if dataset is None:
            dataset = self.build_dataset(split=False)
        evaluators = self._prepare_test()
        metrics = {}
        dataset.pd_to_pttensor()
        X, y_true = dataset.extract()
        y_pred = self.predict(X, model)

        for step in evaluators:
            metric = step.save_metric(y_true, y_pred)
            metrics[step.name] = metric

        return metrics

    def build_train_test(
        self,
        seed: int = 0,
        epochs: int | None = None,
        device: torch.device | str = "cpu",
        ckpt_to_resume: str | None = None,
        dataset: DataBundle | None = None,
        skip_if_executed=False,
        track_with_mlflow=False,
        mlflow_exp_name=None,
    ):
        dataset_train, dataset_test = self.load_datasets()
        prepared_train_dataset = self.transform_data(dataset_train, inference=False)
        prepared_test_dataset = self.transform_data(dataset_test, inference=True)

        # Stage 5: Model Training
        model = self.train(dataset=prepared_train_dataset)

        metrics = self.test(model=model, dataset=prepared_test_dataset)

        if track_with_mlflow:
            self.track_with_mlflow(dataset, model, metrics, mlflow_exp_name)

        return (dataset_train, dataset_test), model, metrics

    def track_with_mlflow(
        self, dataset=None, model=None, metrics=None, mlflow_exp_name=None
    ):
        """
        Tracks the dataset, model (sklearn or PyTorch), and metrics in an MLflow experiment. Logs self.config as an artifact.

        Parameters:
            dataset (optional): The dataset to log (could be a file path or a dataset object).
            model (optional): The machine learning model to log (supports sklearn and PyTorch).
            metrics (optional): Dictionary of metrics to log.
            mlflow_exp_name (optional): Name of the MLflow experiment to log under. If not provided, a default name is used.
        """
        # Start an MLflow run in the specified experiment
        if mlflow_exp_name:
            mlflow.set_experiment(mlflow_exp_name)
        else:
            mlflow.set_experiment("Default_Experiment")

        with mlflow.start_run() as run:

            # Log the model if provided
            if model is not None:
                if "sklearn" in str(type(model)):
                    mlflow.sklearn.log_model(model, "model")
                elif "torch" in str(type(model)):
                    mlflow.pytorch.log_model(model, "model")
                else:
                    raise ValueError(
                        "Model type not supported. Only sklearn and PyTorch models are supported."
                    )

            # Log metrics if provided
            if metrics is not None:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

            mlflow.log_artifact(self.config)

            mlflow.log_param("run_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        print(
            f"Run {run.info.run_id} logged to experiment: {mlflow_exp_name or 'Default_Experiment'}"
        )

    def load_datasets(self):

        # Stage 1: Data Ingestion
        if self.cachehandler.stage > 1:
            raw_data = self.cachehandler.load_data_checkpoint(stage=1)
            raw_data = list(raw_data.values())[0]
        else:
            raw_data = self.dataingestor.get_data()
            self.cachehandler.save_data_checkpoint(raw_data, stage=1)
        if self.split:
            # Stage 2: Train Test Split
            if self.cachehandler.stage > 2:
                dataset_train = self.cachehandler.load_from_databundle_checkpoint(
                    stage=2, prefix="train"
                )
                dataset_test = self.cachehandler.load_from_databundle_checkpoint(
                    stage=2, prefix="test"
                )

            else:
                X_train, X_test, y_train, y_test = self.train_test_splitter.split_data(
                    raw_data
                )
                dataset_train = DataBundle(X_train, y_train)
                dataset_test = DataBundle(X_test, y_test)
                self.cachehandler.save_as_databundle_checkpoint(
                    dataset_train, stage=2, prefix="train"
                )
                self.cachehandler.save_as_databundle_checkpoint(
                    dataset_test, stage=2, prefix="test"
                )
            return dataset_train, dataset_test

    def transform_data(self, dataset, inference=False):
        X, y = dataset.extract()
        if inference is True:
            if not self.dataingestor.confirm_schema(dataset.data.feature):
                assert "Error"
            for step in self.data_preparation:
                X, y = step.transform_data(X, y)
                dataset = DataBundle(X, y)
        else:
            if self.cachehandler.stage > 3:
                dataset = self.cachehandler.load_from_databundle_checkpoint(stage=3)
            else:
                step_ids = []
                step_id = ""
                for step in self.data_preparation:
                    step_id = step_id + step.name
                    step_ids.append(step_id)
                if not self.dataingestor.is_schema_present():
                    self.dataingestor.create_schema(dataset.data.feature)
                for step, step_id in zip(
                    reversed(self.data_preparation), reversed(step_ids)
                ):
                    if self.cachehandler.check_cache(stage=3, add_on=step_id):
                        data = self.cachehandler.load_from_databundle_checkpoint(
                            stage=3, add_on=step_id
                        )
                        X, y = data.extract()
                        break

                else:
                    step_id = ""
                    for step in self.data_preparation:
                        step_id = step_id + step.name
                        X, y = step.transform_data(X, y)
                        dataset = DataBundle(X, y)
                        if step.checkpoint is True:
                            self.cachehandler.save_as_databundle_checkpoint(
                                dataset, stage=3, add_on=step_id
                            )
            self.cachehandler.save_as_databundle_checkpoint(dataset, stage=3)
        return dataset

    def predict(self, data, model=None, device: torch.device | None = "cpu"):
        if model == None:
            model = self._prepare_model(device)
        y_pred = model.predict(data)
        return y_pred

    def _prepare_model(
        self, ckpt_to_resume: str | None = None, device: torch.device | None = "cpu"
    ) -> BaseModel:
        model = MODELS.build(self.config, "model", self.config["workspace"])[0]
        if model.workspace is None:
            model.workspace = self.config["workspace"]
        # if ckpt_to_resume is not None:
        #     model.load_checkpoint(ckpt_to_resume)

        if torch.__version__ >= "2" and isinstance(model, torch.nn.Module):
            model = torch.compile(model)

        model = model.to(device)
        return model

    def _prepare_test(self):
        evaluators = EVALUATORS.build(
            config=self.config,
            stage_name="evaluation",
            workspace=self.config["workspace"],
        )
        return evaluators

    def _prepare_dataset_building(self):
        self.dataingestor = DATA_INGESTION.build(
            self.config, "data_ingestion", self.config["workspace"]
        )[0]
        self.train_test_splitter = TRAIN_TEST_SPLITTERS.build(
            self.config, "train_test_split", self.config["workspace"]
        )[0]
        self.data_preparation = DATA_TRANSFORMATIONS.build(
            self.config, "data_preparation", self.config["workspace"]
        )
        self.data_processing = DATA_TRANSFORMATIONS.build(
            self.config, "data_processing", self.config["workspace"]
        )

    # def evaluate(
    #     self,
    #     seed: int = 0,
    #     device: torch.device | str = "cpu",
    #     metric: list | str = "RMSE",
    #     model: BaseModel | None = None,
    #     dataset: DataBundle | None = None,
    #     ckpt_to_resume: str | None = None,
    #     skip_if_executed: bool = True,
    # ):

    #     set_seed(seed)

    #     if skip_if_executed and (
    #         self.config["workspace"] is not None
    #         and any(
    #             Path(self.config["workspace"]).glob(f"predictions_seed_{seed}_*.pkl")
    #         )
    #     ):
    #         print(
    #             f'Skip evaluation for {self.config["workspace"]} '
    #             "as the prediction exists."
    #         )
    #         return

    #     if dataset is None:
    #         dataset, raw_data = build_dataset(self.config, device)
    #         self.raw_data = raw_data
    #     if model is None:
    #         model = self._prepare_model(ckpt_to_resume, device)

    #     prediction = model.predict(dataset)

    #     if isinstance(metric, str):
    #         metric = [metric]

    #     scores = {m: dataset.evaluate(prediction, m) for m in metric}
    #     print(scores)
    #     ts = timestamp()

    #     if self.config["workspace"] is not None:
    #         obj = {
    #             "prediction": prediction,
    #             "scores": scores,
    #             "data": dataset.to("cpu"),
    #             "seed": seed,
    #         }
    #         filename = f"predictions_seed_{seed}_{ts}.pkl"
    #         with open(Path(self.config["workspace"]) / filename, "wb") as f:
    #             pickle.dump(obj, f)


def load_config(config_path: str, workspace: str | None) -> dict:

    configs = import_config(Path(config_path))

    configs["workspace"] = Path("workspaces") / workspace
    return configs


def check_cache(stage: int, configs: dict):
    strings = []
    config_fields = CONFIG_FIELDS
    for field in config_fields[:stage]:
        strings.append(recursive_dump_string(configs[field]))

    filename = hash_string("+".join(strings))
    cache_dir = Path("artifacts/" + config_fields[stage - 1])
    if not cache_dir.exists():
        cache_dir.mkdir()
    cache_file = Path(cache_dir / f"cache_{filename}.pkl")

    return cache_file.exists()


def build_dataset(configs: dict, device: str, config_fields: list | None = None):
    config_fields = config_fields or CONFIG_FIELDS[1:]
    filename = recursive_dump_string(configs["data"])
    filename = hash_string("+".join(filename))
    cache_dir = Path("artifacts/datasets")
    if not cache_dir.exists():
        cache_dir.mkdir()
    cache_file = Path(cache_dir / f"cache_{filename}.pkl")

    if cache_file.exists():
        print(f"Load datasets from cache {str(cache_file)}.")
        with open(cache_file, "rb") as f:
            dataset = pickle.load(f)

    else:

        task = Task(
            label_annotator=configs["label"],
            feature_extractor=configs["feature"],
            train_test_splitter=configs["train_test_split"],
        )

        dataset = task.build()
        train_cells, test_cells = task.get_raw_data()
        data = {
            "dataset": dataset,
            "raw_data": {
                "train_cells": train_cells,
                "test_cells": test_cells,
            },
        }
        raw_data = data["raw_data"]
        # store cache
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    return dataset.to(device)


def set_seed(seed: int):
    print(f"Seed is set to {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def get_workspace_name(configs: dict):
    strings = []
    for field in configs:
        strings.append(recursive_dump_string(configs[field]))

    workspace_name = hash_string("+".join(strings))
    return workspace_name


def get_cache_file(configs: dict):
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
