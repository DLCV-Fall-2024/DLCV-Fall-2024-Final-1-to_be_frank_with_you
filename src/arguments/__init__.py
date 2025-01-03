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
from typing import List, Union, Optional, ClassVar, Any, Dict, Protocol
from dataclasses import is_dataclass


class Dataclass(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]] 


class GroupParams:
    pass


class ParamGroup:

    __yaml_name__ = ""

    def __init__(self, parser: ArgumentParser = None, name: Optional[str] = None, fill_none=False):
        if parser is None:
            return
        if name is None:
            group = parser
        else:
            group = parser.add_argument_group(name)

        for key, value in vars(self).items():
            if isinstance(value, dict) or is_dataclass(value):
                continue
            
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
                    if not any(arg.dest == key for arg in group._actions):
                        group.add_argument(
                            "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                        )
                        group.add_argument(
                            "--no_" + key, dest=key, action="store_false"
                        )
                else:
                    if not any(arg.dest == key for arg in group._actions):
                        group.add_argument(
                            "--" + key, ("-" + key[0:1]), default=value
                        )
            else:
                if t == bool:
                    if not any(arg.dest == key for arg in group._actions):
                        group.add_argument("--" + key, default=value, action="store_true")
                        group.add_argument(
                            "--no_" + key, dest=key, action="store_false"
                        )
                else:
                    if not any(arg.dest == key for arg in group._actions):
                        group.add_argument("--" + key, default=value)

    # def extract(self, args):
    #     group = GroupParams()
    #     for arg in vars(args).items():
    #         if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
    #             setattr(group, arg[0], arg[1])
    #     return group

    def extract(self, args: Namespace):
        group = {}
        self_vars = vars(self)
        for key, value in vars(args).items():
            if value is None:
                continue
            if key.startswith("no_") and isinstance(value, bool):
                key = key[3:]
            if key in self_vars and value != self_vars[key]:
                group[key] = value
        return group
    

class DataclassInstanceAndParamGroup(Dataclass, ParamGroup):
    pass


class ModelParams(ParamGroup):

    def __init__(self, parser=None, sentinel=False):
        self.__yaml_name__ = "Model".upper()
        self.model_id = "llava-hf/llava-1.5-7b-hf"
        self.device = "cuda"
        self.patch_size = 14
        self.vision_feature_select_strategy = "default"  # "default" or "full"
        # self.vision_feature_select_strategy = "full"  # "default" or "full"
        self.gradient_checkpointing = True
        self.lora_config = {
            "r": 4,
            "lora_alpha": 32,
            "target_modules": [
                "q_proj",
                "v_proj",
                "multi_modal_projector.linear_1",
                "multi_modal_projector.linear_2",
            ],
            "exclude_modules": "vision_tower.*",
            "lora_dropout": 0.1,
            "use_dora": True,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
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
        self.num_workers = 10
        self.prefetch_factor = 2
        super().__init__(parser, "Dataset Parameters")


class OptimizationParams(ParamGroup):

    def __init__(self, parser=None):
        self.__yaml_name__ = "Optimization".upper()
        self.epochs = 100
        self.lr = 3e-5
        self.optimizer_type = "default"
        self.accumulation_steps = 4
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
                if isinstance(value, dict):
                    for k, v in value.items():
                        setattr(args, k, v)
                else:
                    setattr(args, key, value)
            elif "." in key:
                keys = key.split(".")
                current = args
                for k in keys[:-1]:
                    try:
                        current = getattr(current, k)
                    except AttributeError:
                        _current = current.get(k, {})
                        setattr(current, k, _current)
                        current = _current
                try:
                    setattr(current, keys[-1], value)
                except AttributeError:
                    current[keys[-1]] = value
        return args

    def save_args(
        self,
        args: Namespace,
        exclude: List[str] = [],
        additional: dict = {},
        only_log_diff=False,
    ):
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
                if (
                    only_log_diff
                    and key in default_values
                    and default_values[key] == value
                ):
                    del default_values[key]
                del origin[key]
            params = {k: v for k, v in params.items() if k in default_values}
            if len(params.keys()) == 0:
                continue
            config[params_class().__yaml_name__] = params

        config.update(origin)
        config.update(additional)
        with open(self.yaml_file, "w") as file:
            yaml.dump(config, file, Dumper=CustomDumper, default_flow_style=False)


class CustomDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(CustomDumper, self).increase_indent(flow, False)


# Custom representer for strings to use `|` for multiline strings
def str_presenter(dumper, data):
    if "\n" in data:  # Use literal block style `|` for multiline strings
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


# Avoid quotes on keys
def dict_presenter(dumper, data):
    return dumper.represent_dict(data.items())


# Register custom representers
yaml.add_representer(str, str_presenter, Dumper=CustomDumper)
yaml.add_representer(dict, dict_presenter, Dumper=CustomDumper)
