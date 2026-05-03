"""YAML config loading with OmegaConf and CLI overrides."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config with optional base config merging and CLI overrides.

    Config files may specify a `base` key pointing to a parent config.
    Overrides use OmegaConf dotlist syntax: ``key=value``.

    Args:
        config_path: Path to the YAML config file.
        overrides: List of ``key=value`` strings to override config values.

    Returns:
        Merged DictConfig.
    """
    config_path = Path(config_path)
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)

    # Merge base config if specified
    if "base" in cfg:
        base_path = config_path.parent / cfg.base
        base_cfg = OmegaConf.load(base_path)
        assert isinstance(base_cfg, DictConfig)
        cfg = OmegaConf.merge(base_cfg, cfg)

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg


def parse_cli_overrides() -> tuple[str | None, list[str]]:
    """Parse ``--config path`` and remaining ``key=value`` overrides from sys.argv.

    Returns:
        Tuple of (config_path, list_of_overrides).
    """
    config_path: str | None = None
    overrides: list[str] = []

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif "=" in args[i]:
            overrides.append(args[i])
            i += 1
        else:
            i += 1

    return config_path, overrides
