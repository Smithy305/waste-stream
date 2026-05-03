"""Loss functions including noise-robust variants for detection classification heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricCrossEntropy(nn.Module):
    """Symmetric Cross Entropy loss for learning with noisy labels.

    Combines standard CE with a reverse CE term that is more robust to label noise.
    Reference: Wang et al., "Symmetric Cross Entropy for Robust Learning with Noisy
    Labels," ICCV 2019.

    Args:
        alpha: Weight for the standard CE term.
        beta: Weight for the reverse CE term.
        num_classes: Number of classes (for clipping in reverse CE).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, num_classes: int = 4) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute symmetric cross entropy.

        Args:
            logits: Raw model outputs of shape ``(N, C)``.
            targets: Ground-truth class indices of shape ``(N,)``.

        Returns:
            Scalar loss.
        """
        # Standard CE: -sum(y * log(p))
        ce_loss = F.cross_entropy(logits, targets)

        # Reverse CE: -sum(p * log(y))
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=1e-7, max=1.0)

        # One-hot encode targets
        one_hot = F.one_hot(targets, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)

        rce_loss = -torch.mean(torch.sum(probs * torch.log(one_hot), dim=1))

        return self.alpha * ce_loss + self.beta * rce_loss


class GeneralisedCrossEntropy(nn.Module):
    """Generalised Cross Entropy loss — a noise-robust alternative to standard CE.

    Uses a Box-Cox transformation of probabilities that down-weights samples
    the model is very uncertain about (which are likely mislabelled).

    Reference: Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training Deep
    Neural Networks with Noisy Labels," NeurIPS 2018.

    Args:
        q: Truncation parameter in (0, 1]. Lower q = more robust, less informative.
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute generalised cross entropy.

        Args:
            logits: Raw model outputs of shape ``(N, C)``.
            targets: Ground-truth class indices of shape ``(N,)``.

        Returns:
            Scalar loss.
        """
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=1e-7, max=1.0)

        # Gather the probability of the target class
        target_probs = probs[torch.arange(probs.size(0), device=probs.device), targets]

        # Generalised CE: (1 - p^q) / q
        loss = (1.0 - target_probs**self.q) / self.q

        return loss.mean()


def build_loss_fn(cfg) -> nn.Module:
    """Factory function to build a loss from config.

    Args:
        cfg: Full experiment config. Reads ``training.loss_fn`` and
            loss-specific sub-configs.

    Returns:
        Loss module.
    """
    loss_name = cfg.training.get("loss_fn", "cross_entropy")
    num_classes = cfg.dataset.num_classes

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "symmetric_ce":
        params = cfg.training.get("symmetric_ce", {})
        return SymmetricCrossEntropy(
            alpha=params.get("alpha", 1.0),
            beta=params.get("beta", 1.0),
            num_classes=num_classes,
        )
    elif loss_name == "generalised_ce":
        params = cfg.training.get("generalised_ce", {})
        return GeneralisedCrossEntropy(q=params.get("q", 0.7))
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
