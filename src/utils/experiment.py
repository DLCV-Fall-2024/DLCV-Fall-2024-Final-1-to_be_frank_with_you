import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Type, TypeVar

import yaml
from omegaconf import OmegaConf
from src.utils import extract_args, load_dataclass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def create_assets(root: Path, exp_name: str, timestamp: str, prefix: Optional[str] = None) -> Tuple[Path, Path, Path]:
    output_dir = root / "outputs"
    if prefix is not None:
        output_dir = output_dir / prefix
    output_dir = output_dir / exp_name / timestamp
    checkpoint_dir = output_dir / "checkpoint"
    log_dir = output_dir / "log"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, checkpoint_dir, log_dir


C = TypeVar("C")


def load_config(
    Config: Type[C],
    name: Optional[str] = None,
    prefix: Optional[str] = None,
    return_config_path: bool = False,
    cli_args: Optional[Namespace] = None,
    sync_fn: Callable[[], None] = None,
    auto_create: bool = False,
    external_defaults: Tuple[List[str], str] = [],
) -> (
    Tuple[Optional[C], str, Path, Path, Path, Path]
    | Tuple[Optional[C], str, Path, Path, Path]
):
    # Initialize default configuration
    default_config = Config()
    use_default_config = name is None

    if not use_default_config:
        configs_dir = ROOT_DIR / "configs"
        if prefix is not None:
            configs_dir = configs_dir / prefix
        config_path = configs_dir / f"{name}.yaml"

        print(f"Config path: {config_path}")

        configs_dir.mkdir(parents=True, exist_ok=True)

        if not config_path.exists():
            if sync_fn is not None:
                sync_fn()
            if auto_create:
                # Add external defaults
                for keys, external_config_name in external_defaults:
                    external_config = OmegaConf.load(
                        configs_dir / f"{external_config_name}.yaml"
                    )
                    target = default_config
                    for key in keys[:-1]:
                        if key not in target.__dict__:
                            target.__dict__[key] = {}
                        target = target.__dict__[key]
                    target.__dict__[keys[-1]] = external_config

                with config_path.open("w") as f:
                    OmegaConf.save(default_config, f)

            return None, None, None

        # Load user-modified configuration
        config = load_dataclass(Config, config_path)
    else:
        config = OmegaConf.to_object(default_config)
        name = "DEFAULT"

    if cli_args is not None:
        config = extract_args(config, cli_args)

    # Create experiment assets (folders and default configuration)
    timestamp = time.strftime("%m%d_%H%M%S")
    output_dir, checkpoint_dir, log_dir = create_assets(ROOT_DIR, name, timestamp, prefix)

    if return_config_path:
        return config, timestamp, output_dir, checkpoint_dir, log_dir, config_path
    else:
        return config, timestamp, output_dir, checkpoint_dir, log_dir


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


def dump_config(config: Any, output_path: Path):
    if not isinstance(config, dict):
        config = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    with output_path.open("w") as f:
        yaml.dump(config, f, Dumper=CustomDumper, default_flow_style=False)
