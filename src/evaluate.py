"""
Evaluation module for model performance analysis.

This module provides:
1. Classification metrics (precision, recall, F1-score)
2. Confusion matrix visualization
3. Per-class accuracy analysis
4. Training history plots
5. Model comparison utilities

Use this module to:
- Evaluate trained models on test data
- Generate visualizations for reports
- Compare different model architectures
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
from tqdm import tqdm

from config import EMOTION_CLASSES, IDX_TO_EMOTION, CHECKPOINT_DIR, OUTPUT_DIR
from models import get_model


def get_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all predictions from a model on a dataset.

    Args:
        model: Trained model
        data_loader: DataLoader for evaluation data
        device: Device to run on

    Returns:
        Tuple of (predictions, ground_truth, probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """
    Calculate classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with accuracy, precision, recall, F1 per class and overall
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(EMOTION_CLASSES))
    )

    # Macro-averaged metrics (equal weight to each class)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )

    # Weighted average (accounts for class imbalance)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    metrics = {
        'accuracy': accuracy,
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'weighted': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        },
        'per_class': {
            emotion: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
            for i, emotion in enumerate(EMOTION_CLASSES)
        }
    }

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """
    Plot a confusion matrix heatmap.

    The confusion matrix shows:
    - Rows: True labels
    - Columns: Predicted labels
    - Diagonal: Correct predictions
    - Off-diagonal: Misclassifications

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Optional path to save figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by row (true label) to get percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES,
        ax=axes[0]
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'{title} (Counts)')

    # Normalized (percentages)
    sns.heatmap(
        cm_normalized, annot=True, fmt='.1%', cmap='Blues',
        xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES,
        ax=axes[1]
    )
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(f'{title} (Normalized)')

    plt.tight_layout()

    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation metrics over epochs.

    Shows:
    - Loss curve (training vs validation)
    - Accuracy curve (training vs validation)

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title: Plot title
        save_path: Optional path to save figure
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_per_class_metrics(
    metrics: Dict,
    title: str = "Per-Class Metrics",
    save_path: Optional[str] = None
) -> None:
    """
    Plot bar chart of precision, recall, F1 for each class.

    Args:
        metrics: Dictionary from calculate_metrics()
        title: Plot title
        save_path: Optional path to save figure
    """
    per_class = metrics['per_class']

    # Extract data
    emotions = list(per_class.keys())
    precision = [per_class[e]['precision'] for e in emotions]
    recall = [per_class[e]['recall'] for e in emotions]
    f1 = [per_class[e]['f1'] for e in emotions]

    # Create grouped bar chart
    x = np.arange(len(emotions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='darkorange')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='forestgreen')

    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-class metrics to {save_path}")
    else:
        plt.show()

    plt.close()


def evaluate_model(
    model_type: str = 'custom',
    checkpoint_path: Optional[str] = None,
    save_plots: bool = True
) -> Dict:
    """
    Complete evaluation pipeline for a trained model.

    Args:
        model_type: 'custom' or 'transfer'
        checkpoint_path: Path to model checkpoint (uses default if None)
        save_plots: Whether to save visualization plots

    Returns:
        Dictionary with all evaluation metrics
    """
    from dataset import create_data_loaders

    print("=" * 60)
    print(f"EVALUATING: {model_type.upper()} MODEL")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / f"{model_type}_best.pth"
    print(f"Loading checkpoint: {checkpoint_path}")

    # Create model and load weights
    model = get_model(model_type)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test data
    _, _, test_loader = create_data_loaders()

    # Get predictions
    print("\nGenerating predictions...")
    y_pred, y_true, y_probs = get_predictions(model, test_loader, device)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nMacro-averaged:")
    print(f"  Precision: {metrics['macro']['precision']:.4f}")
    print(f"  Recall:    {metrics['macro']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['macro']['f1']:.4f}")

    print(f"\nPer-class breakdown:")
    for emotion, class_metrics in metrics['per_class'].items():
        print(f"  {emotion:12s}: P={class_metrics['precision']:.3f}, "
              f"R={class_metrics['recall']:.3f}, F1={class_metrics['f1']:.3f}, "
              f"n={class_metrics['support']}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTION_CLASSES))

    # Generate plots
    if save_plots:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            title=f"{model_type.title()} Model Confusion Matrix",
            save_path=str(OUTPUT_DIR / f"{model_type}_confusion_matrix.png")
        )

        # Per-class metrics
        plot_per_class_metrics(
            metrics,
            title=f"{model_type.title()} Model Per-Class Performance",
            save_path=str(OUTPUT_DIR / f"{model_type}_per_class_metrics.png")
        )

        # Training history (if available)
        history_path = CHECKPOINT_DIR / f"{model_type}_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            plot_training_history(
                history,
                title=f"{model_type.title()} Model Training History",
                save_path=str(OUTPUT_DIR / f"{model_type}_training_history.png")
            )

    # Save metrics
    metrics_path = OUTPUT_DIR / f"{model_type}_metrics.json"
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(metrics), f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    return metrics


def compare_models(
    model_results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Create comparison visualization of multiple models.

    Args:
        model_results: Dictionary mapping model names to their metrics
        save_path: Optional path to save figure
    """
    model_names = list(model_results.keys())

    # Extract metrics for comparison
    accuracies = [model_results[name]['accuracy'] for name in model_names]
    macro_f1s = [model_results[name]['macro']['f1'] for name in model_names]

    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, macro_f1s, width, label='Macro F1', color='darkorange')

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument('--model', type=str, default='custom',
                        choices=['custom', 'transfer', 'all'])
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()

    if args.model == 'all':
        # Evaluate all models and compare
        results = {}
        for model_type in ['custom', 'transfer']:
            checkpoint = CHECKPOINT_DIR / f"{model_type}_best.pth"
            if checkpoint.exists():
                results[model_type] = evaluate_model(model_type)
            else:
                print(f"Checkpoint not found for {model_type}, skipping...")

        if len(results) > 1:
            compare_models(results, str(OUTPUT_DIR / "model_comparison.png"))
    else:
        evaluate_model(args.model, args.checkpoint)
