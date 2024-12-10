from typing import Optional, Tuple, Type, TypeVar, cast, Any

import os
from omegaconf import OmegaConf


ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def create_exp_assets(
    root: str, exp_name: str, default_config: Any
) -> Tuple[bool, str, str]:
    configs_dir = os.path.join(root, "configs")
    config_path = os.path.join(configs_dir, f"{exp_name}.yaml")
    os.makedirs(configs_dir, exist_ok=True)

    config_exists = os.path.exists(config_path)
    if not config_exists:
        with open(config_path, "w") as config_file:
            OmegaConf.save(default_config, config_file)

    model_dir = os.path.join(root, "models", exp_name)
    log_dir = os.path.join(root, "logs", exp_name)
    output_dir = os.path.join(root, "outputs", exp_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return not config_exists, config_path, model_dir, log_dir, output_dir


C = TypeVar("C")


def load_config(
    name: str,
    Config: Type[C],
    config_path: Optional[str] = None,
) -> Tuple[Optional[C], str, str, str]:
    # Initialize default configuration
    default_config = OmegaConf.structured(Config)

    # Create experiment assets (folders and default configuration)
    new_config, default_config_path, model_dir, log_dir, output_dir = (
        create_exp_assets(ROOT_DIR, name, default_config)
    )

    if new_config:
        return None, model_dir, log_dir, output_dir

    if config_path is None:
        config_path = default_config_path

    # Load user-modified configuration
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(default_config, config)
    config = cast(Config, config)

    return config, model_dir, log_dir, output_dir
