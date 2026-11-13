from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"


def load_config():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Pipeline configuration not found: {CONFIG_PATH}")

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Pipeline configuration must be a YAML mapping: {CONFIG_PATH}")

    return config


def resolve_path(config, key):
    value = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing configuration value: {key}")
        value = value[part]

    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration path must be a non-empty string: {key}")

    return PROJECT_ROOT / value
