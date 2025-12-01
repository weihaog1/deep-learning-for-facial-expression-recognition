"""
Dataset module for loading and preprocessing facial expression images.

This module provides:
1. FacialExpressionDataset - PyTorch Dataset class for loading images
2. Data loading utilities for train/val/test splits
3. Data augmentation transforms
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import (
    IMAGES_DIR, LEGEND_PATH, EMOTION_CLASSES, EMOTION_TO_IDX,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, TEST_SPLIT, VAL_SPLIT, RANDOM_SEED
)


def load_and_clean_labels() -> pd.DataFrame:
    """
    Load the legend.csv file and clean the emotion labels.

    The raw data has inconsistent casing (e.g., 'happiness' vs 'HAPPINESS').
    This function normalizes all labels to lowercase and filters out
    unsupported emotions like 'contempt'.

    Returns:
        DataFrame with columns: ['image', 'emotion'] - cleaned and filtered
    """
    # Read the CSV file
    df = pd.read_csv(LEGEND_PATH)

    # Normalize emotion labels to lowercase
    df['emotion'] = df['emotion'].str.lower().str.strip()

    # Keep only the emotions we're using (filter out 'contempt', etc.)
    df = df[df['emotion'].isin(EMOTION_CLASSES)].copy()

    # Verify that all image files exist
    df['image_path'] = df['image'].apply(lambda x: IMAGES_DIR / x)
    df = df[df['image_path'].apply(lambda x: x.exists())].copy()

    # Add numeric labels
    df['label'] = df['emotion'].map(EMOTION_TO_IDX)

    print(f"Loaded {len(df)} valid samples")
    print(f"Class distribution:\n{df['emotion'].value_counts()}")

    return df[['image', 'emotion', 'label', 'image_path']]


class FacialExpressionDataset(Dataset):
    """
    PyTorch Dataset for facial expression images.

    This dataset:
    - Loads images from disk on-demand (memory efficient)
    - Applies transforms (augmentation, normalization)
    - Returns (image_tensor, label) pairs

    Args:
        dataframe: DataFrame with 'image_path' and 'label' columns
        transform: Optional torchvision transforms to apply
    """

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get image path and label
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label = row['label']

        # Load image and convert to RGB (some images might be grayscale)
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform() -> transforms.Compose:
    """
    Get transforms for training data.

    Includes data augmentation to prevent overfitting:
    - Random horizontal flip (faces are mostly symmetric)
    - Small rotation (up to 10 degrees)
    - Color jitter (simulate lighting changes)
    - Normalization with ImageNet statistics
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_test_transform() -> transforms.Compose:
    """
    Get transforms for validation/test data.

    No augmentation - just resize and normalize for consistent evaluation.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def create_data_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    The data is split as follows:
    - 80% training (of which 10% is validation)
    - 20% testing

    Args:
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load and clean the labels
    df = load_and_clean_labels()

    # First split: separate test set (20%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=df['label']  # Maintain class balance
    )

    # Second split: separate validation from training (10% of original)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),  # Adjust ratio
        random_state=RANDOM_SEED,
        stratify=train_val_df['label']
    )

    print(f"\nData splits:")
    print(f"  Training:   {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Testing:    {len(test_df)} samples")

    # Create datasets with appropriate transforms
    train_dataset = FacialExpressionDataset(train_df, transform=get_train_transform())
    val_dataset = FacialExpressionDataset(val_df, transform=get_test_transform())
    test_dataset = FacialExpressionDataset(test_df, transform=get_test_transform())

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Shuffle training data
        num_workers=num_workers,
        pin_memory=True         # Faster GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # No shuffle for validation
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # No shuffle for testing
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Utility functions for baseline model (HOG features)
# =============================================================================

def load_images_for_baseline() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images as numpy arrays for the baseline model.

    Returns grayscale images resized to 48x48 for HOG feature extraction.

    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    from config import BASELINE_IMG_SIZE

    df = load_and_clean_labels()

    images = []
    labels = []

    for _, row in df.iterrows():
        # Load and convert to grayscale
        img = Image.open(row['image_path']).convert('L')
        # Resize to baseline size
        img = img.resize((BASELINE_IMG_SIZE, BASELINE_IMG_SIZE))
        # Convert to numpy array
        images.append(np.array(img))
        labels.append(row['label'])

    return np.array(images), np.array(labels)


if __name__ == "__main__":
    # Quick test of data loading
    print("Testing data loading...")
    train_loader, val_loader, test_loader = create_data_loaders()

    # Get a sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch shape: {images.shape}")
    print(f"Labels: {labels}")
