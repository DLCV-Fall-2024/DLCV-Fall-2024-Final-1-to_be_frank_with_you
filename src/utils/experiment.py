from typing import Optional, Tuple, List, Type, TypeVar, cast, Any, Dict

from pathlib import Path
import time
import yaml
from omegaconf import OmegaConf


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_config_path(root: Path, exp_name: str) -> Path:
    configs_dir = root / "configs"
    config_path = configs_dir / f"{exp_name}.yaml"
    return config_path


def create_config(root: Path, exp_name: str, default_config: Any) -> Tuple[bool, Path]:
    configs_dir = root / "configs"
    config_path = configs_dir / f"{exp_name}.yaml"
    configs_dir.mkdir(parents=True, exist_ok=True)

    config_exists = config_path.exists()
    if not config_exists:
        with config_path.open("w") as config_file:
            OmegaConf.save(default_config, config_file)

    return not config_exists, config_path


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
    name: str,
    Config: Type[C],
    auto_create: bool = False,
    external_defaults: Tuple[List[str], str] = [],
) -> Tuple[Optional[C], Path, Path, Path, Path]:
    # Initialize default configuration
    default_config = OmegaConf.structured(Config)

    configs_dir = ROOT_DIR / "configs"
    config_path = configs_dir / f"{name}.yaml"

    configs_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
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

    # Create experiment assets (folders and default configuration)
    timestamp = time.strftime("%m%d_%H%M%S")
    output_dir, checkpoint_dir, log_dir = create_assets(ROOT_DIR, f"{name}_{timestamp}")

    # Load user-modified configuration
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(default_config, config)
    config = cast(Config, config)

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
