#!/usr/bin/env python3
"""Entry point: semi-supervised teacher-student training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.semi_supervised import SemiSupervisedTrainer
from src.utils.config import load_config, parse_cli_overrides
from src.utils.logging import setup_logger, init_wandb
from src.utils.seed import seed_everything


def main() -> None:
    config_path, overrides = parse_cli_overrides()
    if config_path is None:
        config_path = "configs/semi_supervised.yaml"

    cfg = load_config(config_path, overrides)
    seed_everything(cfg.seed)

    logger = setup_logger(
        log_file=Path(cfg.output_dir) / cfg.get("experiment_name", "semi_supervised") / "train.log"
    )
    logger.info("Config:\n%s", cfg)

    init_wandb(cfg)

    trainer = SemiSupervisedTrainer(cfg)
    results = trainer.train()

    logger.info("Training complete. Results: %s", results)


if __name__ == "__main__":
    main()
