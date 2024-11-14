# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.
import copy
from pathlib import Path


class Registry:
    """Build a custom class instance using a dict.

    Examples:

    Register a class:

    ```python
    MODELS = Registry('Models')

    @MODELS.register()
    class MyModel:
        def __init__(self, data):
            self.data = data
    ```

    Build an instance by dict:

    ```python
    # You must include MyModel first, this can be done by include MyModel in
    # the `__init__.py` file of the project, and then users can just include
    # the project to register all models.
    import src

    config = dict(name='MyModel', data='foo')

    mymodel = MODELS.build(config)
    ```
    """

    def __init__(self, name: str):
        self.name = name
        self.class_mapping = {}

    def register(self, name=None):
        def _register(cls):
            module_name = name or cls.__name__
            if module_name in self.class_mapping:
                raise ValueError(f"class {module_name} is already registered!")
            self.class_mapping[module_name] = cls
            return cls

        return _register

    def build(
        self,
        config: dict,
        stage_name: str,
        workspace: Path,
        error_handle: str = "raise",
        **kwargs,
    ):
        if config is None:
            return
        if isinstance(config, dict):
            stage_config = config[stage_name]
            if isinstance(stage_config, dict):
                stage_config = [stage_config]
        steps = []

        for step in stage_config:
            step = copy.deepcopy(step)
            step["workspace"] = workspace
            step["config"] = config
            name = step.get("name", None)
            if name is None:
                return

            if name in self.class_mapping:
                step = self.class_mapping[name](
                    **{
                        k: v for k, v in step.items() if k != "name" and k not in kwargs
                    },
                    **kwargs,
                )
                steps.append(step)
            else:
                if error_handle.lower() != "ignore":
                    raise KeyError(f"{name} is not registered to {self.name}!")
        return steps
