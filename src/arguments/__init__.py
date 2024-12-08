#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Union


class GroupParams:
    pass


class ParamGroup:

    __yaml_name__ = ""

    def __init__(self, parser: ArgumentParser = None, name: str = "", fill_none=False):
        if parser is None:
            return
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("__"):
                continue

            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):

    def __init__(self, parser=None, sentinel=False):
        self.__yaml_name__ = "Model".upper()
        self.model_id = "llava-hf/llava-1.5-7b-hf"
        self.device = "cuda"
        self.patch_size = 14
        self.vision_feature_select_strategy = "full"  # "default" or "full"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        return g


class PipelineParams(ParamGroup):

    def __init__(self, parser=None):
        self.__yaml_name__ = "Pipeline".upper()
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class DatasetParams(ParamGroup):

    def __init__(self, parser=None):
        self.__yaml_name__ = "Dataset".upper()
        self._dataset_path = "data/test"
        self.seed = 42
        self.batch_size = 8
        self.num_workers = 20
        super().__init__(parser, "Dataset Parameters")


class OptimizationParams(ParamGroup):

    def __init__(self, parser=None):
        self.__yaml_name__ = "Optimization".upper()
        self.epochs = 100
        self.lr = 3e-5
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")


import yaml


class YamlArgsLoader:
    params_classes: List[ParamGroup] = [
        ModelParams,
        PipelineParams,
        DatasetParams,
        OptimizationParams,
    ]

    def __init__(self, yaml_file: Union[str, Path]):
        """
        Initialize with a YAML configuration file.
        :param yaml_file: Path to the YAML configuration file.
        """
        self.yaml_file = Path(yaml_file)

    def _load_yaml(self):
        """
        Load the YAML configuration file.
        :return: Dictionary with configuration settings.
        """
        with open(self.yaml_file, "r") as file:
            return yaml.safe_load(file)

    def overwrite_args(self, args: Namespace) -> Namespace:
        """
        Overwrite parsed arguments using the YAML file values.
        :param args: argparse.Namespace object with already parsed arguments.
        :return: Updated argparse.Namespace object.
        """
        self.config = self._load_yaml()

        params_classes_map = {
            params_class().__yaml_name__: params_class
            for params_class in self.params_classes
        }
        for key, value in self.config.items():
            if key in params_classes_map:
                params_class = params_classes_map[key]()
                for k, v in value.items():
                    if hasattr(params_class, k):
                        setattr(args, k, v)
            elif hasattr(args, key):
                setattr(args, key, value)
        return args

    def save_args(self, args: Namespace, exclude: List[str] = []):
        """
        Save the current arguments to the YAML file.
        :param args: argparse.Namespace object with already parsed arguments.
        """

        origin = {**vars(args)}
        for key in exclude:
            if key in origin:
                del origin[key]
        args = Namespace(**origin)
        config = {}

        for params_class in self.params_classes:
            group: GroupParams = params_class().extract(args)

            default_values = vars(params_class())
            params = vars(group)
            for key, value in params.items():
                if key in default_values and default_values[key] == value:
                    del default_values[key]
                del origin[key]
            params = {k: v for k, v in params.items() if k in default_values}
            if len(params.keys()) == 0:
                # print("No changes in", params_class().__yaml_name__)
                continue
            config[params_class().__yaml_name__] = params

        for key, value in origin.items():
            if "." in key:
                keys = key.split(".")
                if keys[0] not in config:
                    config[keys[0]] = {}
                config[keys[0]][keys[1]] = value
            else:
                config[key] = value

        with open(self.yaml_file, "w") as file:
            yaml.dump(config, file)
