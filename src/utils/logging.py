"""Experiment logging — Python logger, W&B, and TensorBoard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def setup_logger(
    name: str = "zerowaste",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger with console and optional file output.

    Args:
        name: Logger name.
        log_file: Optional path to write logs to disk.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "zerowaste") -> logging.Logger:
    """Retrieve an existing logger by name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def init_wandb(cfg: DictConfig, project: str = "zerowaste-semi-supervised") -> Any:
    """Initialise a Weights & Biases run from config.

    Args:
        cfg: Experiment config.
        project: W&B project name.

    Returns:
        The wandb run object, or None if wandb is disabled.
    """
    if not cfg.get("wandb", {}).get("enabled", False):
        return None

    try:
        import wandb
    except ImportError:
        logging.getLogger("zerowaste").warning("wandb not installed, skipping init.")
        return None

    return wandb.init(
        project=project,
        name=cfg.get("wandb", {}).get("run_name"),
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.get("wandb", {}).get("tags", []),
    )


def init_tensorboard(log_dir: str | Path) -> Any:
    """Initialise a TensorBoard SummaryWriter.

    Args:
        log_dir: Directory for TensorBoard event files.

    Returns:
        SummaryWriter instance.
    """
    from torch.utils.tensorboard import SummaryWriter

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))
