# src/utils.py
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Iterable

import numpy as np
import torch


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (can be a bit slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Device helper
# -----------------------------
def get_device() -> torch.device:
    """
    Return a CUDA device if available, else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Metrics
# -----------------------------
def accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute top-1 accuracy for a batch.

    Args:
        logits: Model outputs of shape (B, num_classes).
        targets: Ground truth labels of shape (B,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


class AverageMeter:
    """
    Track and update the average of a metric (e.g., loss, accuracy).
    """
    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        fmt_str = "{name} {val" + self.fmt + "} (avg: {avg" + self.fmt + "})"
        return fmt_str.format(name=self.name, val=self.val, avg=self.avg)


# -----------------------------
# Checkpointing
# -----------------------------
def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_path: str | Path,
) -> None:
    """
    Save training checkpoint.

    Typical `state` dict:
        {
            "epoch": int,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": float,
            "classes": class_names,
        }
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint into model (and optimizer if given).

    Returns:
        The full checkpoint dict, so you can read epoch, best_val_acc, etc.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint


# -----------------------------
# Misc helpers
# -----------------------------
def create_experiment_dir(
    root: str | Path = "runs",
    exp_name: str | None = None,
) -> Path:
    """
    Create (or reuse) an experiment directory to store logs, checkpoints, etc.

    If exp_name is None, uses a simple incremental naming scheme: run_001, run_002, ...
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if exp_name is not None:
        exp_dir = root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    # auto-increment run_xxx
    existing = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("run_")]
    existing_ids = []
    for d in existing:
        try:
            existing_ids.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            continue

    next_id = max(existing_ids, default=0) + 1
    exp_dir = root / f"run_{next_id:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
