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
from PIL import Image, ImageOps
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split


# =============================================================================
# Custom Image Pre-processing Transforms (from FERDCNN paper)
# =============================================================================

class GammaCorrection:
    """
    Apply gamma correction to adjust image contrast.

    From FERDCNN paper: gamma=1.7 achieved best results.
    - gamma > 1: darker image (emphasizes shadows)
    - gamma < 1: brighter image
    - gamma = 1: no change

    Formula: I_out = I_in ^ gamma
    """

    def __init__(self, gamma: float = 1.7):
        self.gamma = gamma

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert to numpy for gamma correction
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Apply gamma correction
        img_corrected = np.power(img_array, self.gamma)

        # Convert back to PIL Image
        img_corrected = (img_corrected * 255).astype(np.uint8)
        return Image.fromarray(img_corrected)

    def __repr__(self):
        return f"GammaCorrection(gamma={self.gamma})"


class HistogramEqualization:
    """
    Apply histogram equalization to normalize brightness distribution.

    This improves contrast and makes the model more robust to
    different lighting conditions.

    From FERDCNN paper: Applied after gamma correction.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        # For RGB images, apply equalization to each channel
        if img.mode == 'RGB':
            # Convert to LAB color space for better results
            # Equalize only the L (lightness) channel
            from PIL import Image
            import numpy as np

            # Simple per-channel equalization
            r, g, b = img.split()
            r_eq = ImageOps.equalize(r)
            g_eq = ImageOps.equalize(g)
            b_eq = ImageOps.equalize(b)
            return Image.merge('RGB', (r_eq, g_eq, b_eq))
        else:
            return ImageOps.equalize(img)

    def __repr__(self):
        return "HistogramEqualization()"

from config import (
    IMAGES_DIR, LEGEND_PATH, EMOTION_CLASSES, EMOTION_TO_IDX,
    CNN_IMG_SIZE, CNN_CHANNELS, TRANSFER_IMG_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, GRAYSCALE_MEAN, GRAYSCALE_STD,
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


def get_train_transform(model_type: str = 'custom', use_preprocessing: bool = True) -> transforms.Compose:
    """
    Get transforms for training data based on model type.

    Strong augmentation ensures each sample is unique, even when oversampling.
    This helps the model learn robust features instead of memorizing images.

    Augmentations applied:
    - RandomHorizontalFlip: Faces are symmetric
    - RandomRotation: Slight head tilts (±15°)
    - RandomAffine: Scale (90-110%) and translate (±10%)
    - ColorJitter: Brightness/contrast variation
    - RandomPerspective: Slight camera angle changes
    - GaussianBlur: Occasional blur for robustness

    Args:
        model_type: 'custom' for CNN or 'transfer' for ResNet
        use_preprocessing: Whether to apply gamma correction and histogram equalization
    """
    if model_type == 'custom':
        # Custom CNN: 50x50 grayscale
        transform_list = [
            transforms.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
            transforms.Grayscale(num_output_channels=1),
        ]

        if use_preprocessing:
            transform_list.extend([
                GammaCorrection(gamma=1.7),
                HistogramEqualization(),
            ])

        # Strong augmentation - makes each oversampled image unique
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,              # Rotation handled above
                translate=(0.1, 0.1),   # Shift up to 10% in x/y
                scale=(0.9, 1.1),       # Zoom 90-110%
            ),
            transforms.ColorJitter(
                brightness=0.3,         # ±30% brightness
                contrast=0.3,           # ±30% contrast
            ),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5))
            ], p=0.2),  # Occasional stronger blur
            transforms.ToTensor(),
            transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD)
        ])
    else:
        # Transfer Learning: 224x224 RGB
        transform_list = [
            transforms.Resize((TRANSFER_IMG_SIZE, TRANSFER_IMG_SIZE)),
        ]

        if use_preprocessing:
            transform_list.extend([
                GammaCorrection(gamma=1.7),
                HistogramEqualization(),
            ])

        # Strong augmentation for transfer learning
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,         # Color saturation (RGB only)
                hue=0.05,               # Slight hue shift (RGB only)
            ),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    return transforms.Compose(transform_list)


def get_test_transform(model_type: str = 'custom', use_preprocessing: bool = True) -> transforms.Compose:
    """
    Get transforms for validation/test data based on model type.

    No augmentation - just resize and normalize for consistent evaluation.

    Args:
        model_type: 'custom' for CNN or 'transfer' for ResNet
        use_preprocessing: Whether to apply gamma correction and histogram equalization
    """
    if model_type == 'custom':
        # Custom CNN: 48x48 grayscale
        transform_list = [
            transforms.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
            transforms.Grayscale(num_output_channels=1),
        ]

        if use_preprocessing:
            transform_list.extend([
                GammaCorrection(gamma=1.7),
                HistogramEqualization(),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD)
        ])
    else:
        # Transfer Learning: 224x224 RGB
        transform_list = [
            transforms.Resize((TRANSFER_IMG_SIZE, TRANSFER_IMG_SIZE)),
        ]

        if use_preprocessing:
            transform_list.extend([
                GammaCorrection(gamma=1.7),
                HistogramEqualization(),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    return transforms.Compose(transform_list)


def smart_oversample(df: pd.DataFrame, target_min: int = 5300, target_max: int = 6700) -> pd.DataFrame:
    """
    Smart oversampling to balance classes WITHOUT making them perfectly uniform.

    

    Args:
        df: DataFrame with 'label' column
        target_min: Minimum target samples per class
        target_max: Maximum target samples per class

    Returns:
        Oversampled DataFrame
    """
    np.random.seed(RANDOM_SEED)

    oversampled_dfs = []
    class_counts = df['label'].value_counts()

    print("\nSmart Oversampling (based on other group's findings):")
    print(f"  Target range: {target_min:,} - {target_max:,} samples per class")
    print(f"  Original distribution:")

    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        original_count = len(class_df)

        if original_count >= target_min:
            # Keep as-is if already above minimum
            oversampled_dfs.append(class_df)
            print(f"    Class {label}: {original_count:,} -> {original_count:,} (kept)")
        else:
            # Oversample to a random target within range (adds variation)
            # Use lower target for very small classes to reduce overfitting risk
            if original_count < 100:
                target = target_min  # Smallest classes get minimum target
            else:
                target = np.random.randint(target_min, target_max + 1)

            # Calculate how many times to repeat + remainder
            n_repeats = target // original_count
            n_remainder = target % original_count

            # Repeat the entire dataframe n_repeats times
            repeated_dfs = [class_df] * n_repeats

            # Add random samples for the remainder
            if n_remainder > 0:
                remainder_df = class_df.sample(n=n_remainder, replace=True, random_state=RANDOM_SEED)
                repeated_dfs.append(remainder_df)

            oversampled_class = pd.concat(repeated_dfs, ignore_index=True)
            oversampled_dfs.append(oversampled_class)
            print(f"    Class {label}: {original_count:,} -> {len(oversampled_class):,} (oversampled {len(oversampled_class)//original_count}x)")

    result = pd.concat(oversampled_dfs, ignore_index=True)

    # Shuffle the result
    result = result.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n  Total samples: {len(df):,} -> {len(result):,}")

    return result


def create_data_loaders(
    model_type: str = 'custom',
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,  # Parallel data loading (speeds up training)
    use_oversampling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Proper approach (no data leakage):
    - Split FIRST into train/val/test (no overlap)
    - Oversample ONLY the training set
    - Val/test keep original distribution for realistic evaluation

    Args:
        model_type: 'custom' for CNN (50x50 grayscale) or 'transfer' for ResNet (224x224 RGB)
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        use_oversampling: Whether to apply smart oversampling to training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Print image format info
    if model_type == 'custom':
        print(f"\nUsing Custom CNN format: {CNN_IMG_SIZE}x{CNN_IMG_SIZE} grayscale")
    else:
        print(f"\nUsing Transfer Learning format: {TRANSFER_IMG_SIZE}x{TRANSFER_IMG_SIZE} RGB")

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

    print(f"\nData splits (before oversampling):")
    print(f"  Training:   {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Testing:    {len(test_df)} samples")

    # Apply smart oversampling to TRAINING DATA ONLY (proper method - no data leakage)
    if use_oversampling:
        train_df = smart_oversample(train_df)
        print(f"\nData splits (after oversampling training only):")
        print(f"  Training:   {len(train_df)} samples (oversampled)")
        print(f"  Validation: {len(val_df)} samples (original distribution)")
        print(f"  Testing:    {len(test_df)} samples (original distribution)")

    # Create datasets with appropriate transforms for the model type
    train_dataset = FacialExpressionDataset(train_df, transform=get_train_transform(model_type))
    val_dataset = FacialExpressionDataset(val_df, transform=get_test_transform(model_type))
    test_dataset = FacialExpressionDataset(test_df, transform=get_test_transform(model_type))

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
