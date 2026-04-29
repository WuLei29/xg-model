from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "paths") -> dict:
    config_path = ROOT / "configs" / f"{name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_path(key_path: str) -> Path:
    """Resolve a dotted config key (e.g. 'data.raw_dir') to an absolute Path."""
    cfg = load_config("paths")
    keys = key_path.split(".")
    node = cfg
    for k in keys:
        node = node[k]
    return ROOT / node
