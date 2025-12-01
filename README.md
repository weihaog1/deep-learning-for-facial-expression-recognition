# Deep Learning for Facial Expression Recognition

A project comparing traditional ML and deep learning approaches for classifying facial expressions into 7 emotion categories.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments (baseline + CNN + transfer learning)
cd src
python run_experiments.py --all
```

## Project Structure

```
.
├── data/                    # Dataset (not in git - clone separately)
│   └── facial_expressions-master/
│       ├── images/          # ~13,700 face images
│       └── data/legend.csv  # Image labels
├── src/                     # Source code
│   ├── config.py           # Settings and hyperparameters
│   ├── dataset.py          # Data loading and augmentation
│   ├── baseline_model.py   # HOG + SVM baseline
│   ├── models.py           # CNN architectures
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Metrics and plots
│   ├── gradcam.py          # Visualization
│   └── run_experiments.py  # Main entry point
├── checkpoints/            # Saved models (created during training)
├── outputs/                # Results and plots (created during evaluation)
└── docs/                   # Documentation
```

## Dataset

The Muxspace Facial Expression dataset contains ~13,700 labeled face images.

**7 Emotion Classes:**
- anger, disgust, fear, happiness, neutral, sadness, surprise

**Note:** The original data has inconsistent casing (e.g., "HAPPINESS" vs "happiness"). Our code normalizes all labels to lowercase automatically.

## Models

### 1. Baseline: HOG + SVM
Traditional approach using hand-crafted features:
- **Feature extraction:** Histogram of Oriented Gradients (HOG)
- **Classifier:** Support Vector Machine with RBF kernel
- **Purpose:** Establishes a performance baseline for comparison

```bash
python run_experiments.py --baseline
```

### 2. Custom CNN
A CNN trained from scratch:
- 4 convolutional blocks (Conv + BatchNorm + ReLU + MaxPool)
- Global average pooling (reduces overfitting)
- 2 fully connected layers with dropout

```bash
python run_experiments.py --cnn
```

### 3. Transfer Learning (ResNet18)
Pretrained on ImageNet, fine-tuned for our task:
- **Phase 1:** Train classifier with frozen backbone
- **Phase 2:** Fine-tune entire network with lower learning rate

```bash
python run_experiments.py --transfer
```

## How to Run

### Option 1: Run Everything
```bash
cd src
python run_experiments.py --all
```
This will:
1. Train the baseline model
2. Train the custom CNN
3. Train the transfer learning model
4. Evaluate and compare all models
5. Generate Grad-CAM visualizations

### Option 2: Run Individual Experiments
```bash
# Just the baseline
python run_experiments.py --baseline

# Just the custom CNN
python run_experiments.py --cnn --epochs 30

# Just transfer learning
python run_experiments.py --transfer

# Evaluate trained models
python run_experiments.py --evaluate

# Generate Grad-CAM visualizations
python run_experiments.py --gradcam
```

### Option 3: Train Models Directly
```bash
# Train custom CNN
python train.py --model custom --epochs 25

# Train transfer learning with fine-tuning
python train.py --model transfer_finetune --epochs 25
```

## Understanding the Code

### config.py
All settings in one place:
- Paths to data and outputs
- Emotion class definitions
- Image sizes
- Training hyperparameters (batch size, learning rate, etc.)

### dataset.py
Handles data loading:
- `load_and_clean_labels()` - Reads CSV and normalizes labels
- `FacialExpressionDataset` - PyTorch Dataset class
- `create_data_loaders()` - Creates train/val/test splits

**Data Augmentation (training only):**
- Random horizontal flip
- Small rotation (up to 10 degrees)
- Color jitter (brightness/contrast)

### models.py
Neural network architectures:
- `CustomCNN` - Our CNN built from scratch
- `TransferLearningModel` - ResNet18 with custom classifier

### train.py
Training loop with:
- Early stopping (stops if validation loss doesn't improve)
- Learning rate scheduling (reduces LR on plateau)
- Model checkpointing (saves best model)

### evaluate.py
Metrics and visualization:
- Accuracy, precision, recall, F1-score
- Confusion matrix plots
- Per-class performance charts
- Training history plots

### gradcam.py
Model interpretability:
- Shows which image regions the model focuses on
- Helps verify the model learns meaningful features

## Output Files

After running experiments, you'll find:

**checkpoints/**
- `custom_best.pth` - Best custom CNN model
- `transfer_finetuned_best.pth` - Best transfer learning model
- `baseline_model.pkl` - Trained baseline model
- `*_history.json` - Training history data

**outputs/**
- `*_confusion_matrix.png` - Confusion matrix plots
- `*_per_class_metrics.png` - Per-class performance bars
- `*_training_history.png` - Loss/accuracy curves
- `gradcam_*.png` - Grad-CAM visualizations
- `model_comparison.png` - Side-by-side model comparison

## Key Concepts

### Why Use Data Augmentation?
The dataset has ~13,700 images, which is small for deep learning. Augmentation creates variations of training images to:
- Prevent overfitting
- Make the model more robust to variations
- Effectively increase dataset size

### Why Transfer Learning?
Instead of learning from scratch, we use a model pretrained on ImageNet (1M+ images). Benefits:
- Faster training
- Better performance with limited data
- Pretrained layers already know how to detect edges, textures, shapes

### What is Grad-CAM?
A visualization technique that shows which parts of an image the model uses for predictions. This helps us:
- Verify the model looks at faces, not backgrounds
- Understand which facial features matter for each emotion
- Debug when the model makes mistakes

## Tips for Getting Good Results

1. **Check the data first:** Run `python dataset.py` to verify data loading works

2. **Start with the baseline:** It trains in minutes and gives you a quick benchmark

3. **Use GPU if available:** Training is much faster on GPU

4. **Monitor for overfitting:** Watch training vs validation accuracy diverging

5. **Try different hyperparameters:** Edit `config.py` to adjust batch size, learning rate, etc.

## Common Issues

**"File not found" errors:**
- Make sure the data is in `data/facial_expressions-master/`
- Check that `images/` folder contains the actual image files

**Out of memory:**
- Reduce `BATCH_SIZE` in `config.py`
- Use a smaller `IMG_SIZE` (e.g., 128 instead of 224)

**Training is slow:**
- Make sure you have GPU support: `torch.cuda.is_available()`
- Reduce number of epochs for initial testing

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
