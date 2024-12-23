from typing import List, Tuple, Dict, Any, TypeVar, Type, cast

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
from transformers.feature_extraction_utils import BatchFeature
from copy import deepcopy
from dataclasses import is_dataclass
from argparse import Namespace

from src.arguments import DataclassInstanceAndParamGroup, Dataclass


def default(val, default_val):
    return val if val is not None else default_val


# Call `to` for tensors in List or Dict recursively
def container_to(
    container,
    target_dtypes: List[torch.dtype] = [],
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    if isinstance(container, torch.Tensor):
        if container.dtype in target_dtypes:
            return container.to(device=device, dtype=dtype)
        else:
            return container.to(device=device)
    elif isinstance(container, (list, tuple)):
        return type(container)(
            container_to(item, target_dtypes, device=device, dtype=dtype)
            for item in container
        )
    elif isinstance(container, dict):
        return {
            key: container_to(value, target_dtypes, device=device, dtype=dtype)
            for key, value in container.items()
        }
    return container


# Concatenate tensors in different List or Dict recursively
def iterate_container(container: Dict | List | Tuple | torch.Tensor | Any):
    if isinstance(container, (list, tuple)):
        for i, item in enumerate(container):
            for tensor, paths in iterate_container(item):
                yield (tensor, [i] + paths)
    elif isinstance(container, dict):
        for key, value in container.items():
            for tensor, paths in iterate_container(value):
                yield (tensor, [key] + paths)
    else:
        if not isinstance(container, torch.Tensor):
            container = torch.tensor([container])
        yield (container, [])


def container_cat(containers: List[Dict | List | Tuple | torch.Tensor], dim: int = 0):
    if isinstance(containers[0], torch.Tensor):
        return torch.cat(containers, dim=dim)

    cated = deepcopy(containers[0])

    iters = [iterate_container(container) for container in containers]
    while True:
        iter_values = [next(iter, (None, [])) for iter in iters]
        if all(tensor is None for tensor, _ in iter_values):
            break
        tensors = [tensor for tensor, _ in iter_values]
        paths = iter_values[0][1]

        tensor = torch.cat(tensors, dim=dim)
        _cated = cated
        for path in paths[:-1]:
            _cated = _cated[path]
        _cated[paths[-1]] = tensor

    return cated


def batch_feature_to_dict(
    container: Dict | List | Tuple | torch.Tensor | BatchFeature | Any,
):
    if isinstance(container, BatchFeature) or isinstance(container, dict):
        return {key: batch_feature_to_dict(value) for key, value in container.items()}
    elif isinstance(container, (list, tuple)):
        return type(container)(batch_feature_to_dict(item) for item in container)
    else:
        return container


def convert_to_dict(config: DictConfig | Dataclass) -> Dict:
    if isinstance(config, Dict):
        return config
    elif isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    elif is_dataclass(config):
        return OmegaConf.to_container(OmegaConf.create(config), resolve=True)

    raise ValueError(f"Unsupported config type: {type(config)}")


def is_container(config: Any) -> bool:
    return isinstance(config, DictConfig) or is_dataclass(config)


def merge_config(
    config: DictConfig | Dataclass,
    additional: Dict | DictConfig | Dataclass,
) -> Dataclass:
    if isinstance(config, DictConfig):
        config: Dataclass = OmegaConf.to_object(config)

    additional: Dict = convert_to_dict(additional)

    config_vars = vars(config)
    for key, value in additional.items():
        if key not in config_vars:
            continue

        config_value = config_vars[key]
        if is_container(config_value):
            config.__dict__[key] = merge_config(config_value, value)
        else:
            config.__dict__[key] = value

    return config


C = TypeVar("C", bound=DataclassInstanceAndParamGroup)


def load_dataclass(Config: Type[C], config_path: str, strict: bool = True) -> C:
    default_config = Config()
    if strict:
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(default_config, config)
        config: C = OmegaConf.to_object(config)
    else:
        config = OmegaConf.load(config_path)
        config: C = merge_config(default_config, config)

    return config


def extract_args(config: C, args: Namespace) -> C:
    config = OmegaConf.merge(config, config.extract(args))
    return OmegaConf.to_object(config)


def dataclass_to_dict(config: Dataclass) -> Dict:
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True)