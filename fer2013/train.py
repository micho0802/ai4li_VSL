#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml
import torch
from torch import nn, optim
from tqdm.auto import tqdm

from src.data import create_dataloaders
from src.models import create_resnet50
from src.utils import (
    set_seed,
    get_device,
    save_checkpoint,
    create_experiment_dir,
)
from src.metrics import accuracy, AverageMeter, evaluate_classification
from src.visualize import plot_training_curves
from src.visualize import plot_confusion_matrix


def train(cfg: dict) -> None:
    # -----------------------------
    # Setup
    # -----------------------------
    set_seed(cfg["data"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # Experiment directory
    exp_dir = create_experiment_dir(
        root=cfg["output"]["save_dir"],
        exp_name=cfg["output"]["run_name"],
    )
    print(f"Saving checkpoints to: {exp_dir}")

    # -----------------------------
    # Data
    # -----------------------------
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        train_root=cfg["data"]["train_root"],
        test_root=cfg["data"]["test_root"],
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        img_size=cfg["data"]["img_size"],
        seed=cfg["data"]["seed"],
        num_workers=cfg["data"]["num_workers"],
    )

    num_classes = len(class_names)
    print(f"Num classes: {num_classes} -> {class_names}")

    # -----------------------------
    # Model
    # -----------------------------
    model = create_resnet50(
        num_classes=num_classes,
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
    ).to(device)

    # Print trainable params once
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params}/{total_params} "
          f"({trainable_params/total_params:.2%})")

    # -----------------------------
    # Loss, Optimizer, Scheduler
    # -----------------------------
    # Loss
    loss_func = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["loss"]["label_smoothing"])
    )

    # Optimizer (force floats)
    lr = float(cfg["optimizer"]["lr"])
    weight_decay = float(cfg["optimizer"]["weight_decay"])

    print(f"Using optimizer AdamW with lr={lr}, weight_decay={weight_decay}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler (force correct types)
    scheduler_mode = cfg["scheduler"]["mode"]
    scheduler_factor = float(cfg["scheduler"]["factor"])
    scheduler_patience = int(cfg["scheduler"]["patience"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    # -----------------------------
    # Training config
    # -----------------------------
    epochs = cfg["training"]["epochs"]
    tolerance = cfg["training"]["early_stopping_patience"]

    best_val_loss = math.inf
    early_stop_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(epochs):
        # ----- Train -----
        model.train()
        train_loss_meter = AverageMeter("train_loss")

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            unit="batch"
        ) as pbar:
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)
                loss = loss_func(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_meter.update(loss.item(), n=images.size(0))

                pbar.set_postfix(loss=train_loss_meter.avg)
                pbar.update(1)

        train_losses.append(train_loss_meter.avg)

        # ----- Validation -----
        model.eval()
        val_loss_meter = AverageMeter("val_loss")
        val_acc_meter = AverageMeter("val_acc")

        with torch.no_grad():
            with tqdm(
                total=len(val_loader),
                desc=f"Epoch {epoch+1}/{epochs} [Val]",
                unit="batch"
            ) as pbar_val:
                for images_v, targets_v in val_loader:
                    images_v, targets_v = images_v.to(device), targets_v.to(device)

                    logits = model(images_v)
                    loss_v = loss_func(logits, targets_v)

                    batch_acc = accuracy(logits, targets_v)

                    val_loss_meter.update(loss_v.item(), n=images_v.size(0))
                    val_acc_meter.update(batch_acc, n=images_v.size(0))

                    pbar_val.set_postfix(
                        loss=val_loss_meter.avg,
                        acc=val_acc_meter.avg
                    )
                    pbar_val.update(1)

        val_losses.append(val_loss_meter.avg)
        val_accuracies.append(val_acc_meter.avg)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- Train Loss: {train_loss_meter.avg:.4f} "
            f"- Val Loss: {val_loss_meter.avg:.4f} "
            f"- Val Acc: {val_acc_meter.avg:.4f}"
        )

        # ----- LR scheduler step -----
        scheduler.step(val_loss_meter.avg)

        # ----- Early stopping & checkpoint -----
        if val_loss_meter.avg < best_val_loss:
            best_val_loss = val_loss_meter.avg
            early_stop_counter = 0

            print("✅ Validation loss improved — saving checkpoint and resetting patience.")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_acc": val_acc_meter.avg,
                    "classes": class_names,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_accuracies": val_accuracies,
                },
                checkpoint_path=exp_dir / "best_model.pth",
            )
        else:
            early_stop_counter += 1
            print(f"⚠️ No improvement in val loss for {early_stop_counter} epoch(s).")

        if early_stop_counter >= tolerance:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

    # -----------------------------
    # Final evaluation on test set
    # -----------------------------
    print("\nEvaluating best model on test set...")
    # Reload best model (optional, but safer)
    checkpoint = torch.load(exp_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    results = evaluate_classification(model, test_loader, device, class_names)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test F1 (macro): {results['f1_macro']:.4f}")
    print("\nClassification report:\n")
    print(results["classification_report"])
    plot_training_curves(train_losses, val_losses, val_accuracies)
    cm = results["confusion_matrix"]
    plot_confusion_matrix(cm, class_names, normalize=False)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet50_fer2013.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
