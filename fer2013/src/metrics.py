# src/metrics.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)


# ----------------------------------------
# Basic batch accuracy
# ----------------------------------------
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy for a batch.
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


# ----------------------------------------
# AverageMeter (for tracking loss/accuracy)
# ----------------------------------------
class AverageMeter:
    """
    Tracks and updates the average of any metric.
    """
    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        fmt_str = "{name}: {avg" + self.fmt + "}"
        return fmt_str.format(name=self.name, avg=self.avg)


# ----------------------------------------
# Full evaluation function
# ----------------------------------------
@torch.no_grad()
def evaluate_classification(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, any]:
    """
    Run full evaluation over a dataloader and generate:
    - accuracy
    - f1-score (macro)
    - confusion matrix
    - classification report

    Returns:
        dict with metrics
    """
    all_preds = []
    all_targets = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds)
    cls_report = classification_report(all_targets, all_preds, target_names=class_names)

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "y_true": all_targets,
        "y_pred": all_preds,
    }
