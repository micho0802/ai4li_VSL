import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns

import torch
from torch.utils.data import Dataset


def unnormalize(img_tensor: torch.Tensor,
                mean: List[float],
                std: List[float]) -> np.ndarray:
    """
    Convert a normalized PyTorch image tensor (C,H,W) back to numpy (H,W,C)
    for visualization.
    """
    img = img_tensor.numpy().transpose((1, 2, 0))  # (C,H,W) -> (H,W,C)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def show_random_images(
    dataset: Dataset,
    class_names: List[str],
    mean: List[float],
    std: List[float],
    n_images: int = 32,
    rows: int = 4,
    cols: int = 8,
    seed: int = 42,
):
    """
    Visualize a grid of random images from a dataset.

    Args:
        dataset: PyTorch dataset (e.g., train_data)
        class_names: list of class labels (strings)
        mean/std: normalization statistics
        n_images: number of images to display
        rows, cols: layout of the subplot grid
    """

    assert n_images <= rows * cols, "Grid too small for number of images."

    np.random.seed(seed)
    idxs = np.random.choice(len(dataset), size=n_images, replace=False)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.flatten()

    for ax, idx in zip(axes, idxs):
        img, label = dataset[idx]
        ax.imshow(unnormalize(img, mean, std))
        ax.set_title(class_names[label], fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, normalize=False):
    """
    Plot a confusion matrix using seaborn heatmap.

    Args:
        cm: confusion matrix (numpy array)
        class_names: list of class labels
        normalize: whether to show percentages instead of raw counts
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar=True
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    plt.tight_layout()
    plt.show()
