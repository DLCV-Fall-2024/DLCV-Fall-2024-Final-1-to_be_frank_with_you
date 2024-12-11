from typing import Optional, Tuple, Type, TypeVar, cast, Any, Dict

from pathlib import Path
import time
import yaml
from omegaconf import OmegaConf


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def create_config(root: Path, exp_name: str, default_config: Any) -> Tuple[bool, str]:
    configs_dir = root / "configs"
    config_path = configs_dir / f"{exp_name}.yaml"
    configs_dir.mkdir(parents=True, exist_ok=True)

    config_exists = config_path.exists()
    if not config_exists:
        with config_path.open("w") as config_file:
            OmegaConf.save(default_config, config_file)

    return not config_exists, str(config_path)


def create_assets(root: Path, exp_name: str) -> Tuple[str, str, str]:
    output_dir = root / "outputs" / exp_name
    checkpoint_dir = output_dir / "checkpoint"
    log_dir = output_dir / "log"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir), str(checkpoint_dir), str(log_dir)


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
        return None, None, None, None, None
    if config_path is None:
        config_path = default_config_path

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
