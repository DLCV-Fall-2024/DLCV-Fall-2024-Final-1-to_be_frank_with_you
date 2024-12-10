from typing import Optional, Tuple, Type, TypeVar, cast, Any, Dict

import os
import time
from omegaconf import OmegaConf


ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def create_config(root: str, exp_name: str, default_config: Any) -> Tuple[bool, str]:
    configs_dir = os.path.join(root, "configs")
    config_path = os.path.join(configs_dir, f"{exp_name}.yaml")
    os.makedirs(configs_dir, exist_ok=True)

    config_exists = os.path.exists(config_path)
    if not config_exists:
        with open(config_path, "w") as config_file:
            OmegaConf.save(default_config, config_file)

    return not config_exists, config_path


def create_assets(root: str, exp_name: str) -> Tuple[str, str, str]:
    output_dir = os.path.join(root, "outputs", exp_name)
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    log_dir = os.path.join(output_dir, "log")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return output_dir, checkpoint_dir, log_dir


C = TypeVar("C")


def load_config(
    name: str,
    Config: Type[C],
    config_path: Optional[str] = None,
) -> Tuple[Optional[C], str, str, str]:
    # Initialize default configuration
    default_config = OmegaConf.structured(Config)

    new_config, default_config_path = create_config(ROOT_DIR, name, default_config)
    if new_config:
        return None, None, None, None
    if config_path is None:
        config_path = default_config_path

    # Create experiment assets (folders and default configuration)
    timestamp = time.strftime("%m%d_%H%M%S")
    output_dir, checkpoint_dir, log_dir = create_assets(ROOT_DIR, f"{name}_{timestamp}")

    # Load user-modified configuration
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(default_config, config)
    config = cast(Config, config)

    return config, output_dir, checkpoint_dir, log_dir


def dump_additional_config(config: Dict[str, Any], output_dir: str):
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)
