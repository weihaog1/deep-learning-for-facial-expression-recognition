"""
Main script to run all experiments.

This script provides a simple interface to:
1. Run the baseline model (HOG + SVM)
2. Train the custom CNN
3. Train the transfer learning model
4. Evaluate all models and compare results
5. Generate Grad-CAM visualizations

Usage:
    python run_experiments.py --all          # Run everything
    python run_experiments.py --baseline     # Only baseline
    python run_experiments.py --cnn          # Only custom CNN
    python run_experiments.py --transfer     # Only transfer learning
    python run_experiments.py --evaluate     # Evaluate trained models
    python run_experiments.py --gradcam      # Generate Grad-CAM visualizations
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import CHECKPOINT_DIR, OUTPUT_DIR


def run_baseline():
    """Run the baseline HOG + SVM model."""
    print("\n" + "=" * 70)
    print("RUNNING BASELINE MODEL (HOG + SVM)")
    print("=" * 70)

    from baseline_model import train_baseline_model
    model, results = train_baseline_model()

    print(f"\nBaseline Results:")
    print(f"  Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"  Testing Accuracy:  {results['test_accuracy']:.4f}")

    return results


def run_custom_cnn(epochs: int = 25):
    """Train the custom CNN model."""
    print("\n" + "=" * 70)
    print("TRAINING CUSTOM CNN")
    print("=" * 70)

    from train import train_model
    results = train_model('custom', num_epochs=epochs)

    print(f"\nCustom CNN Results:")
    print(f"  Test Accuracy: {results['test_acc']:.4f}")
    print(f"  Best Epoch:    {results['best_epoch'] + 1}")

    return results


def run_transfer_learning(epochs: int = 25):
    """Train the transfer learning model with fine-tuning."""
    print("\n" + "=" * 70)
    print("TRAINING TRANSFER LEARNING MODEL")
    print("=" * 70)

    from train import train_with_fine_tuning
    results = train_with_fine_tuning(num_epochs=epochs)

    print(f"\nTransfer Learning Results:")
    print(f"  Test Accuracy: {results['test_acc']:.4f}")

    return results


def run_evaluation():
    """Evaluate all trained models and compare."""
    print("\n" + "=" * 70)
    print("EVALUATING ALL MODELS")
    print("=" * 70)

    from evaluate import evaluate_model, compare_models

    results = {}

    # Check which models are available
    for model_type in ['custom', 'transfer']:
        checkpoint = CHECKPOINT_DIR / f"{model_type}_best.pth"
        if checkpoint.exists():
            print(f"\nEvaluating {model_type} model...")
            results[model_type] = evaluate_model(model_type)
        else:
            print(f"\nCheckpoint not found for {model_type}, skipping...")

    # Also check for fine-tuned transfer model
    finetuned_checkpoint = CHECKPOINT_DIR / "transfer_finetuned_best.pth"
    if finetuned_checkpoint.exists():
        print("\nEvaluating fine-tuned transfer model...")
        # Load and evaluate the fine-tuned model
        import torch
        from models import get_model
        from dataset import create_data_loaders
        from evaluate import get_predictions, calculate_metrics

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model('transfer')
        checkpoint = torch.load(finetuned_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        _, _, test_loader = create_data_loaders()
        y_pred, y_true, _ = get_predictions(model, test_loader, device)
        results['transfer_finetuned'] = calculate_metrics(y_true, y_pred)

    # Compare if multiple models available
    if len(results) > 1:
        print("\nGenerating model comparison...")
        compare_models(results, str(OUTPUT_DIR / "model_comparison.png"))

    return results


def run_gradcam():
    """Generate Grad-CAM visualizations for trained models."""
    print("\n" + "=" * 70)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("=" * 70)

    import torch
    from gradcam import visualize_batch
    from models import get_model
    from dataset import create_data_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = create_data_loaders()

    for model_type in ['custom', 'transfer']:
        checkpoint = CHECKPOINT_DIR / f"{model_type}_best.pth"
        if checkpoint.exists():
            print(f"\nGenerating Grad-CAM for {model_type} model...")

            model = get_model(model_type)
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device)
            model.eval()

            save_path = OUTPUT_DIR / f"gradcam_{model_type}.png"
            visualize_batch(model, test_loader, model_type, num_samples=8,
                          save_path=str(save_path))
        else:
            print(f"\nCheckpoint not found for {model_type}, skipping Grad-CAM...")


def main():
    parser = argparse.ArgumentParser(
        description="Run facial expression recognition experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --all          Run complete pipeline
  python run_experiments.py --baseline     Only run baseline model
  python run_experiments.py --cnn          Only train custom CNN
  python run_experiments.py --transfer     Only train transfer learning
  python run_experiments.py --evaluate     Evaluate trained models
  python run_experiments.py --gradcam      Generate visualizations
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    parser.add_argument('--baseline', action='store_true',
                        help='Run baseline HOG + SVM')
    parser.add_argument('--cnn', action='store_true',
                        help='Train custom CNN')
    parser.add_argument('--transfer', action='store_true',
                        help='Train transfer learning model')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained models')
    parser.add_argument('--gradcam', action='store_true',
                        help='Generate Grad-CAM visualizations')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs (default: 25)')

    args = parser.parse_args()

    # If no specific option, show help
    if not any([args.all, args.baseline, args.cnn, args.transfer,
                args.evaluate, args.gradcam]):
        parser.print_help()
        return

    # Create output directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run requested experiments
    if args.all or args.baseline:
        run_baseline()

    if args.all or args.cnn:
        run_custom_cnn(args.epochs)

    if args.all or args.transfer:
        run_transfer_learning(args.epochs)

    if args.all or args.evaluate:
        run_evaluation()

    if args.all or args.gradcam:
        run_gradcam()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Model checkpoints: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
