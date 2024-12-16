import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, cast

import yaml
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def create_assets(root: Path, exp_name: str) -> Tuple[Path, Path, Path]:
    output_dir = root / "outputs" / exp_name
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
    cli_args: Optional[Dict] = None,
    sync_fn: Callable[[], None] = None,
    auto_create: bool = False,
    external_defaults: Tuple[List[str], str] = [],
) -> Tuple[Optional[C], str, Path, Path, Path]:
    # Initialize default configuration
    default_config = OmegaConf.structured(Config)
    use_default_config = name is None

    if not use_default_config:
        configs_dir = ROOT_DIR / "configs"
        config_path = configs_dir / f"{name}.yaml"

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
                        if key not in target:
                            target[key] = {}
                        target = target[key]
                    target[keys[-1]] = external_config

                with config_path.open("w") as f:
                    OmegaConf.save(default_config, f)

            return None, None, None, None, None

        # Load user-modified configuration
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(default_config, config)
        config = OmegaConf.to_object(config)
    
    else:
        config = OmegaConf.to_object(default_config)
        name = "DEFAULT"

    if cli_args is not None:
        config = OmegaConf.merge(config, cli_args)
    config = cast(Config, config)

    # Create experiment assets (folders and default configuration)
    timestamp = time.strftime("%m%d_%H%M%S")
    output_dir, checkpoint_dir, log_dir = create_assets(ROOT_DIR, f"{name}_{timestamp}")

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


def dump_additional_config(config: Dict[str, Any], output_dir: str):
    with (Path(output_dir) / "config.yaml").open("w") as f:
        yaml.dump(config, f, Dumper=CustomDumper, default_flow_style=False)
