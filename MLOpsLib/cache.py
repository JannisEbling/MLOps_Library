import hashlib
import io
import os
import pickle
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from MLOpsLib import DataBundle


class CacheHandler:

    def __init__(self, config_path: Path, config: dict, stage: int):
        self.config = config
        self.config_Path = config_path
        self.stage = stage

    def create_workspace(self, config_path, get_workspace_path=False):
        cache_file = self.get_workspace_name(self.config)
        if self.config["workspace"] is not None:
            workspace_name = f"workspace_{cache_file}"
            workspace_path = Path(self.config["workspace"], workspace_name)
            os.makedirs(workspace_path, exist_ok=True)
            shutil.copyfile(config_path, workspace_path / f"config.yaml")
            if get_workspace_path is True:
                return workspace_path

    def recursive_dump_string(self, data):
        if isinstance(data, list):
            return "_".join([self.recursive_dump_string(x) for x in data])
        if isinstance(data, dict):
            return "_".join(
                [self.recursive_dump_string(data[key]) for key in sorted(data.keys())]
            )
        return str(data)

    def hash_string(self, string):
        sha256_hash = hashlib.sha256()
        sha256_hash.update(string.encode("utf-8"))
        hash_value = sha256_hash.hexdigest()
        truncated_hash = hash_value[:32]
        return truncated_hash

    def get_workspace_name(self, config: dict):
        strings = []
        for field in config:
            strings.append(self.recursive_dump_string(config[field]))

        workspace_name = self.hash_string("+".join(strings))
        return workspace_name

    def set_current_stage(self):

        if self.check_cache(3):
            stage = 4
        else:
            if self.check_cache(2, prefix="train"):
                stage = 3
            else:
                if self.check_cache(1):
                    stage = 2
                else:
                    stage = 1
        self.stage = stage

    def get_cache_file(
        self,
        stage: int = None,
        prefix: str = "cache",
        add_on: str = "",
    ):
        if stage is None:
            stage = self.stage
        strings = []
        config_fields = list(self.config.keys())
        for field in config_fields[:stage]:
            strings.append(self.recursive_dump_string(self.config[field]))

        filename = self.hash_string("+".join(strings) + add_on)
        cache_dir = Path(f"{self.config['workspace']}/" + config_fields[stage - 1])
        if not cache_dir.exists():
            cache_dir.mkdir()
        cache_file = Path(cache_dir / f"{prefix}_{filename}.pkl")
        return cache_file

    def check_cache(self, stage: int = None, prefix: str = "cache", add_on: str = ""):
        if stage is None:
            stage = self.stage

        cache_file = self.get_cache_file(stage, prefix, add_on)

        return cache_file.exists()

    def load_data_checkpoint(self, stage: int = None):

        if stage is None:
            stage = self.stage
        cache_file = self.get_cache_file(stage)

        with open(cache_file, "rb") as f:
            # Load the zipped content from the pickle file
            zipped_data = pickle.load(f)

        # Create a bytes buffer from the loaded zipped content
        zip_buffer = io.BytesIO(zipped_data)

        # Open the zip file from the bytes buffer
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            loaded_data = {}
            for file_name in zip_file.namelist():
                # Read each file and load it into a DataFrame or Series
                with zip_file.open(file_name) as file:
                    if file_name.startswith("file_"):
                        # Load as DataFrame
                        loaded_data[file_name] = pd.read_csv(file)
                    # elif file_name.startswith("series_"):
                    #     # Load as Series
                    #     loaded_data[file_name] = pd.Series(pd.read_csv(file).squeeze())
                    # Remove the else block for unpickling since we don't need it
                    # If any other file type needs to be loaded, you can handle it here.

        return loaded_data

    def save_data_checkpoint(self, *checkpoint_files, stage: int = None):
        if stage is None:
            stage = self.stage
        cache_file = self.get_cache_file(stage)
        zip_buffer = io.BytesIO()

        # Open the zipfile in memory and add the files
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for i, file in enumerate(checkpoint_files):
                if isinstance(file, pd.DataFrame) or isinstance(file, pd.Series):
                    # Convert DataFrame to CSV in memory and add it to the zip file
                    csv_buffer = io.StringIO()
                    file.to_csv(csv_buffer, index=False)
                    zip_file.writestr(f"file_{i}.csv", csv_buffer.getvalue())
                # Write each file to the zip archive (file can be a path or file-like object)
                elif isinstance(file, str):  # If file is a path
                    zip_file.write(file, f"file_{i}")
                else:  # If it's a file-like object (e.g., from open or io.BytesIO)
                    zip_file.writestr(f"file_{i}", file.read())

        # Move the buffer's cursor to the beginning before saving
        zip_buffer.seek(0)

        # Save the zipped data as a pickle file
        with open(cache_file, "wb") as f:
            # pickle the zipped content
            pickle.dump(zip_buffer.getvalue(), f)

    def save_as_databundle_checkpoint(
        self, databundle, stage: int = None, prefix: str = "cache", add_on: str = ""
    ):
        if stage is None:
            stage = self.stage
        cache_file = self.get_cache_file(stage, prefix, add_on)
        databundle.dump(cache_file)

    def load_from_databundle_checkpoint(
        self, stage: int = None, prefix: str = "cache", add_on: str = ""
    ):
        if stage is None:
            stage = self.stage
        cache_file = self.get_cache_file(stage, prefix, add_on)
        databundle = DataBundle.load(cache_file)
        return databundle
