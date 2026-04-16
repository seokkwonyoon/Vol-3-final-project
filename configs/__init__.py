"""
Configuration package for the dynamic IC pipeline.

By default, the package imports settings from configs.default.
It also supports a command-line key of the form:
    --config configs/test.py
    --config test.py

The selected config is loaded at import time so that all modules
which import from `configs` receive the correct settings.
"""
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional

from configs.default import *  # noqa: F401, F403

CONFIG_NAME = os.getenv("CONFIG_NAME", "default")
CONFIG_PATH = None


def _parse_config_arg(argv: List[str]) -> Optional[str]:
    for index, arg in enumerate(argv):
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
        if arg == "--config" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _resolve_config_path(config_value: str) -> Path:
    candidate = Path(config_value)
    if candidate.suffix == ".py" or "/" in config_value or "\\" in config_value:
        if candidate.is_file():
            return candidate

        cwd_candidate = Path.cwd() / config_value
        if cwd_candidate.is_file():
            return cwd_candidate

        repo_candidate = Path(__file__).resolve().parent / config_value
        if repo_candidate.is_file():
            return repo_candidate

        if not candidate.suffix:
            candidate_py = candidate.with_suffix(".py")
            repo_candidate_py = Path(__file__).resolve().parent / candidate_py
            if repo_candidate_py.is_file():
                return repo_candidate_py

        raise ImportError(f"Config file not found: {config_value}")
    raise FileNotFoundError


def _load_config_module(config_value: str):
    global CONFIG_NAME, CONFIG_PATH

    module = None
    config_name = config_value

    try:
        config_path = _resolve_config_path(config_value)
        config_name = config_path.stem
        spec = importlib.util.spec_from_file_location(f"configs._override_{config_name}", str(config_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config file: {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        CONFIG_PATH = str(config_path)
    except (FileNotFoundError, ImportError):
        module = importlib.import_module(f"configs.{config_value}")
        CONFIG_PATH = None

    CONFIG_NAME = config_name
    for attr, value in vars(module).items():
        if attr.isupper():
            globals()[attr] = value


def _load_config_from_argv() -> None:
    config_value = _parse_config_arg(sys.argv)
    if config_value:
        _load_config_module(config_value)
        return

    if CONFIG_NAME and CONFIG_NAME != "default":
        _load_config_module(CONFIG_NAME)


_load_config_from_argv()
