"""
Evaluation metrics and visualization for KWS model.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import IDX_TO_LABEL, LABELS


def evaluate_model(model, dataloader, device, verbose=True):
    """
    Run full evaluation on a dataloader.
    Returns accuracy, per-class metrics, and predictions for confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating", disable=not verbose):
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(LABELS))
    )

    if verbose:
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print(f"Macro F1 Score:   {f1_macro:.4f}")
        print("\nPer-class metrics:")
        print(classification_report(
            all_labels, all_preds,
            target_names=LABELS,
            labels=range(len(LABELS)),
            digits=4,
            zero_division=0,
        ))

    results = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "f1_per_class": f1,
        "support": support,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }

    return results


def plot_confusion_matrix(labels, predictions, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(labels, predictions, labels=range(len(LABELS)))
    cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_training_curves(history, save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
    plt.show()
