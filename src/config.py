"""
Configuration settings for the Facial Expression Recognition project.

This file contains all the constants and hyperparameters used throughout
the project, making it easy to modify settings in one place.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory (parent of src folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "facial_expressions-master"
IMAGES_DIR = DATA_DIR / "images"
LEGEND_PATH = DATA_DIR / "data" / "legend.csv"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# =============================================================================
# EMOTION CLASSES
# =============================================================================

# The 7 emotion classes we will use (excluding 'contempt' due to low samples)
# Note: Original data has inconsistent casing - we normalize to lowercase
EMOTION_CLASSES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

# Mapping from emotion name to numeric label
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_CLASSES)}
IDX_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTION_CLASSES)}

NUM_CLASSES = len(EMOTION_CLASSES)

# =============================================================================
# IMAGE SETTINGS
# =============================================================================

# Image size for CNN models (standard for transfer learning)
IMG_SIZE = 224

# Image size for baseline model (smaller for faster processing)
BASELINE_IMG_SIZE = 48

# Normalization values (ImageNet statistics for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# General settings
RANDOM_SEED = 42
TEST_SPLIT = 0.2      # 20% for testing
VAL_SPLIT = 0.1       # 10% of training for validation

# Training settings
BATCH_SIZE = 128  # Increased from 32 for faster training on GPU
NUM_EPOCHS = 25
LEARNING_RATE = 0.001

# Early stopping
PATIENCE = 5          # Stop if no improvement for 5 epochs

# =============================================================================
# MODEL-SPECIFIC SETTINGS
# =============================================================================

# Custom CNN architecture
CNN_DROPOUT = 0.5     # Dropout rate for regularization

# Transfer learning
FREEZE_LAYERS = True  # Freeze pretrained layers initially
FINE_TUNE_LR = 0.0001 # Lower learning rate for fine-tuning
