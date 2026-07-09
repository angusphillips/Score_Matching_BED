import importlib
import logging
import os
import subprocess
import sys

from omegaconf import OmegaConf

# Register custom OmegaConf resolver for eval expressions
# This allows using ${eval:'...'} in config files for arithmetic operations
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def get_git_info() -> dict:
    def _git(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    dirty_output = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty_output),
    }


def load_config(path: str):
    """
    Load a YAML config using OmegaConf.

    Returns a Python dict that can be passed directly to wandb.init(config=config_dict).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load config with OmegaConf
    cfg = OmegaConf.load(path)

    # Convert to plain nested dict (resolves interpolations)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    return cfg_dict


def instantiate(cfg: dict, **kwargs):
    if isinstance(cfg, dict) and "_target_" in cfg:
        # Extract the callable
        target = cfg["_target_"]
        module_name, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        # Remaining fields are kwargs
        kwargs_from_config = {
            k: instantiate(v) for k, v in cfg.items() if k != "_target_"
        }
        merged_kwargs = {**kwargs_from_config, **kwargs}
        return cls(**merged_kwargs)
    else:
        return cfg


def configure_logging(
    root_level: int = logging.INFO,
    format_string: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    suppress_loggers: dict[str, int] | None = None,
):
    """
    Configure logging for the application.

    Args:
        root_level: The logging level for the root logger (default: INFO)
        format_string: The format string for log messages
        suppress_loggers: A dictionary mapping logger names to their desired logging levels.
                         Use this to suppress verbose third-party loggers.
                         Example: {"orbax.checkpoint": logging.WARNING}
    """
    # Configure root logger (force replaces existing handlers common in notebooks)
    logging.basicConfig(
        level=root_level,
        format=format_string,
        stream=sys.stdout,
        force=True,
    )

    # Set specific loggers to different levels
    if suppress_loggers is None:
        suppress_loggers = {}

    for logger_name, level in suppress_loggers.items():
        logging.getLogger(logger_name).setLevel(level)
