"""Exponential Moving Average model wrapper for teacher-student training."""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

logger = logging.getLogger("zerowaste")


class EMATeacher:
    """Maintains an exponential moving average of a student model's parameters.

    The teacher model is a shadow copy that smoothly tracks the student. It is used
    to generate stable pseudo-labels in the semi-supervised loop.

    Args:
        student_model: The student model whose parameters to track.
        decay: EMA decay rate. Higher values produce a more conservative teacher.
            Common values: 0.999 (default), 0.9999 (very conservative).
    """

    def __init__(self, student_model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(student_model)
        self.shadow.eval()

        # Freeze shadow parameters
        for param in self.shadow.parameters():
            param.requires_grad_(False)

        self.num_updates = 0
        logger.info("Initialised EMA teacher with decay=%.6f", decay)

    @torch.no_grad()
    def update(self, student_model: nn.Module) -> None:
        """Update the teacher's parameters with the student's current state.

        Uses a warmup schedule: effective decay ramps from 0 to ``self.decay``
        over the first few updates so early student noise doesn't dominate.

        Args:
            student_model: Current student model.
        """
        self.num_updates += 1
        # Warmup: effective_decay = min(decay, (1 + num_updates) / (10 + num_updates))
        effective_decay = min(
            self.decay,
            (1 + self.num_updates) / (10 + self.num_updates),
        )

        student_params = dict(student_model.named_parameters())
        for name, shadow_param in self.shadow.named_parameters():
            if name in student_params:
                student_param = student_params[name]
                shadow_param.data.mul_(effective_decay).add_(
                    student_param.data, alpha=1.0 - effective_decay
                )

        # Also update buffers (e.g. batch norm running stats)
        student_buffers = dict(student_model.named_buffers())
        for name, shadow_buf in self.shadow.named_buffers():
            if name in student_buffers:
                shadow_buf.data.copy_(student_buffers[name].data)

    def get_model(self) -> nn.Module:
        """Return the teacher (shadow) model for inference.

        Returns:
            The EMA shadow model in eval mode.
        """
        self.shadow.eval()
        return self.shadow

    def state_dict(self) -> dict:
        """Return serialisable state for checkpointing.

        Returns:
            Dict with shadow state dict, decay, and update count.
        """
        return {
            "shadow_state_dict": self.shadow.state_dict(),
            "decay": self.decay,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a checkpoint.

        Args:
            state: Dict as returned by :meth:`state_dict`.
        """
        self.shadow.load_state_dict(state["shadow_state_dict"])
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
        self.shadow.eval()
