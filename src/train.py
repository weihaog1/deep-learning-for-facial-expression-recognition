"""
Training module for neural network models.

This module provides:
1. Training loop with validation
2. Early stopping to prevent overfitting
3. Learning rate scheduling
4. Model checkpointing
5. Training history tracking

Usage:
    python train.py --model custom   # Train custom CNN
    python train.py --model transfer # Train transfer learning model
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from config import (
    NUM_EPOCHS, LEARNING_RATE, PATIENCE,
    CHECKPOINT_DIR, RANDOM_SEED, FINE_TUNE_LR, NUM_CLASSES
)
from models import get_model, count_parameters
from dataset import create_data_loaders, load_and_clean_labels


def calculate_class_weights(device: torch.device) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced data.

    From FERDCNN paper and other group's findings:
    - Class weighting is CRITICAL for preventing mode collapse
    - Without it, model just predicts majority class (neutral) for everything

    Formula: weight[i] = total_samples / (num_classes * samples_in_class[i])

    Returns:
        Tensor of class weights to use in CrossEntropyLoss
    """
    # Load the dataset to get class counts
    df = load_and_clean_labels()
    class_counts = df['label'].value_counts().sort_index().values

    print(f"\nClass distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count} samples")

    # Calculate weights: inverse frequency
    # More samples = lower weight, fewer samples = higher weight
    total_samples = len(df)
    weights = total_samples / (NUM_CLASSES * class_counts)

    # Normalize weights so they sum to NUM_CLASSES
    weights = weights / weights.sum() * NUM_CLASSES

    print(f"\nClass weights:")
    for i, w in enumerate(weights):
        print(f"  Class {i}: {w:.4f}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Get the best available device (GPU if available)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training if it doesn't improve
    for a specified number of epochs (patience).

    Also saves the best model based on validation loss.
    """

    def __init__(self, patience: int = PATIENCE, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if this is the best model so far
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement - reset counter
            self.best_loss = val_loss
            self.counter = 0
            return True  # This is the best model
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in pbar:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(
    model_type: str = 'custom',
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    save_name: Optional[str] = None,
    use_class_weights: bool = False  # Disabled: using smart oversampling instead (better results)
) -> Dict:
    """
    Complete training pipeline.

    Args:
        model_type: 'custom' or 'transfer'
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        save_name: Name for saved model (defaults to model_type)

    Returns:
        Dictionary with training history and final results
    """
    set_seed()

    print("=" * 60)
    print(f"TRAINING: {model_type.upper()} MODEL")
    print("=" * 60)

    # Setup
    device = get_device()
    save_name = save_name or model_type

    # Create data loaders (with appropriate transforms for model type)
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders(model_type=model_type)

    # Create model
    print("\nCreating model...")
    model = get_model(model_type)
    model = model.to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Loss function with optional class weighting
    # Class weighting is CRITICAL for imbalanced datasets like ours
    # Without it, the model learns to always predict "neutral" (majority class)
    if use_class_weights:
        print("\nCalculating class weights for balanced training...")
        class_weights = calculate_class_weights(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class-weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss (no class weighting)")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    # Checkpoints directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = CHECKPOINT_DIR / f"{save_name}_best.pth"

    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Check early stopping
        is_best = early_stopping(val_loss)
        if is_best:
            print(f"  ✓ New best model! Saving to {best_model_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_model_path)

        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Load best model for final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save training history
    history_path = CHECKPOINT_DIR / f"{save_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    return {
        'history': history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_epoch': checkpoint['epoch'],
        'model_path': str(best_model_path)
    }


def train_with_fine_tuning(num_epochs: int = NUM_EPOCHS) -> Dict:
    """
    Two-phase training for transfer learning:
    1. Train with frozen backbone (fast, learn classifier)
    2. Fine-tune entire network with lower learning rate

    Returns:
        Combined training results
    """
    print("=" * 60)
    print("TRANSFER LEARNING WITH FINE-TUNING")
    print("=" * 60)

    set_seed()
    device = get_device()

    # Create data loaders (224x224 RGB for transfer learning)
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders(model_type='transfer')

    # Phase 1: Train classifier with frozen backbone
    print("\n" + "-" * 60)
    print("PHASE 1: Training classifier (backbone frozen)")
    print("-" * 60)

    model = get_model('transfer', freeze_backbone=True)
    model = model.to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    early_stopping = EarlyStopping(patience=3)  # Shorter patience for phase 1

    history_phase1 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs // 2):  # Half epochs for phase 1
        print(f"\nPhase 1 - Epoch {epoch + 1}/{num_epochs // 2}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history_phase1['train_loss'].append(train_loss)
        history_phase1['train_acc'].append(train_acc)
        history_phase1['val_loss'].append(val_loss)
        history_phase1['val_acc'].append(val_acc)

        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if early_stopping.should_stop:
            print("Early stopping phase 1")
            break

    # Phase 2: Fine-tune entire network
    print("\n" + "-" * 60)
    print("PHASE 2: Fine-tuning entire network")
    print("-" * 60)

    model.unfreeze_backbone()
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Use lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    early_stopping = EarlyStopping(patience=PATIENCE)

    history_phase2 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_path = CHECKPOINT_DIR / "transfer_finetuned_best.pth"

    for epoch in range(num_epochs // 2):  # Half epochs for phase 2
        print(f"\nPhase 2 - Epoch {epoch + 1}/{num_epochs // 2}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history_phase2['train_loss'].append(train_loss)
        history_phase2['train_acc'].append(train_acc)
        history_phase2['val_loss'].append(val_loss)
        history_phase2['val_acc'].append(val_acc)

        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        scheduler.step(val_loss)
        is_best = early_stopping(val_loss)

        if is_best:
            print(f"  ✓ Saving best model")
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_model_path)

        if early_stopping.should_stop:
            print("Early stopping phase 2")
            break

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save combined training history for both phases
    combined_history = {
        'phase1': history_phase1,
        'phase2': history_phase2,
        'train_loss': history_phase1['train_loss'] + history_phase2['train_loss'],
        'train_acc': history_phase1['train_acc'] + history_phase2['train_acc'],
        'val_loss': history_phase1['val_loss'] + history_phase2['val_loss'],
        'val_acc': history_phase1['val_acc'] + history_phase2['val_acc'],
        'phase1_epochs': len(history_phase1['train_loss']),
        'phase2_epochs': len(history_phase2['train_loss'])
    }

    history_path = CHECKPOINT_DIR / "transfer_history.json"
    with open(history_path, 'w') as f:
        json.dump(combined_history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    return {
        'history_phase1': history_phase1,
        'history_phase2': history_phase2,
        'combined_history': combined_history,
        'test_loss': test_loss,
        'test_acc': test_acc
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train facial expression models")
    parser.add_argument(
        '--model', type=str, default='custom',
        choices=['custom', 'transfer', 'transfer_finetune'],
        help='Model type to train'
    )
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)

    args = parser.parse_args()

    if args.model == 'transfer_finetune':
        results = train_with_fine_tuning(args.epochs)
    else:
        results = train_model(args.model, args.epochs, args.lr)

    print("\nTraining complete!")
