"""
Training pipeline for the KWS DS-CNN model.

Includes:
  - Full training loop with validation
  - Learning rate scheduling
  - Early stopping
  - Checkpoint saving
  - Training history tracking
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    AudioConfig, DataConfig, ModelConfig, TrainConfig,
    LABELS, LABEL_TO_IDX,
)
from model.ds_cnn import DSCNN, count_parameters
from data.dataset import create_dataloaders
from utils.metrics import evaluate_model, plot_confusion_matrix, plot_training_curves


class Trainer:
    """
    Handles the full training lifecycle:
    initialization, training loop, validation, checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        train_cfg: TrainConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = train_cfg
        self.device = device

        # Loss function with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # AdamW optimizer (decoupled weight decay)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        # Cosine annealing scheduler for smooth LR decay
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.epochs,
            eta_min=1e-6,
        )

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stop_patience = 15

        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    def train_epoch(self):
        """Run one training epoch. Returns average loss and accuracy."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)

            # Backward
            loss.backward()

            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            running_loss += loss.item() * features.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.4f}",
            })

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self):
        """Run validation. Returns average loss and accuracy."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in self.val_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(features)
            loss = self.criterion(logits, labels)

            running_loss += loss.item() * features.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }

        # Save latest
        path = os.path.join(self.cfg.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = os.path.join(self.cfg.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (val_acc={self.best_val_acc:.4f})")

    def train(self):
        """Full training loop with validation, scheduling, and early stopping."""
        print("=" * 60)
        print("TRAINING START")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.cfg.epochs}")
        print(f"Batch size: {self.cfg.batch_size}")
        print(f"Learning rate: {self.cfg.learning_rate}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print()

        start_time = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            print(f"\nEpoch {epoch}/{self.cfg.epochs}")
            print("-" * 40)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update LR
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            print(
                f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            )
            print(
                f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}"
            )
            print(f"  LR: {current_lr:.6f}")

            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.patience_counter >= self.early_stop_patience:
                print(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"Best val_acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}"
                )
                break

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")

        return self.history

    def evaluate_best_model(self):
        """Load the best checkpoint and evaluate on test set."""
        best_path = os.path.join(self.cfg.checkpoint_dir, "best_model.pt")
        if not os.path.exists(best_path):
            print("No best model found. Run training first.")
            return None

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        print("\n" + "=" * 60)
        print("TEST SET EVALUATION (Best Model)")
        print("=" * 60)

        results = evaluate_model(self.model, self.test_loader, self.device)

        # Save plots
        plot_dir = os.path.join(self.cfg.checkpoint_dir, "..", "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plot_confusion_matrix(
            results["labels"],
            results["predictions"],
            save_path=os.path.join(plot_dir, "confusion_matrix.png"),
        )

        if self.history["train_loss"]:
            plot_training_curves(
                self.history,
                save_path=os.path.join(plot_dir, "training_curves.png"),
            )

        return results


def main():
    """Main entry point for training."""
    # Configuration
    audio_cfg = AudioConfig()
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Seed for reproducibility
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create data loaders
    print("\nPreparing data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_cfg, audio_cfg, train_cfg
    )

    # Build model
    print("\nBuilding model...")
    model = DSCNN(
        n_classes=model_cfg.n_classes,
        n_mfcc=audio_cfg.n_mfcc,
        first_filters=model_cfg.first_conv_filters,
        first_kernel=model_cfg.first_conv_kernel,
        first_stride=model_cfg.first_conv_stride,
        ds_filters=model_cfg.ds_conv_filters,
        ds_kernels=model_cfg.ds_conv_kernels,
        dropout=model_cfg.dropout,
    )
    count_parameters(model)

    # Train
    trainer = Trainer(model, train_loader, val_loader, test_loader, train_cfg, device)
    history = trainer.train()

    # Evaluate on test set
    results = trainer.evaluate_best_model()

    print("\nDone! Model saved to:", train_cfg.checkpoint_dir)
    return results


if __name__ == "__main__":
    main()
