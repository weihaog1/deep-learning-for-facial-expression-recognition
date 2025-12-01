"""
Baseline Model: HOG Features + SVM Classifier

This module implements a traditional machine learning approach for
facial expression recognition as a baseline comparison.

Approach:
1. Extract HOG (Histogram of Oriented Gradients) features from images
2. Train an SVM (Support Vector Machine) classifier
3. Evaluate and compare with deep learning models

HOG features capture edge/gradient structure in images, which is
useful for recognizing facial expressions based on muscle movements.
"""

import numpy as np
from typing import Tuple, Dict
import pickle
from pathlib import Path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import hog
from tqdm import tqdm

from config import (
    BASELINE_IMG_SIZE, EMOTION_CLASSES, IDX_TO_EMOTION,
    TEST_SPLIT, RANDOM_SEED, CHECKPOINT_DIR
)


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extract HOG features from a single grayscale image.

    HOG (Histogram of Oriented Gradients) captures:
    - Edge directions and magnitudes
    - Local shape information
    - Texture patterns

    Args:
        image: Grayscale image as 2D numpy array (48x48)

    Returns:
        1D feature vector from HOG descriptor
    """
    # HOG parameters tuned for facial expressions
    features = hog(
        image,
        orientations=9,        # Number of gradient direction bins
        pixels_per_cell=(8, 8),  # Cell size for local histograms
        cells_per_block=(2, 2),  # Cells per normalization block
        block_norm='L2-Hys',    # Block normalization method
        visualize=False,
        feature_vector=True     # Return as 1D vector
    )
    return features


def extract_features_from_dataset(
    images: np.ndarray,
    show_progress: bool = True
) -> np.ndarray:
    """
    Extract HOG features from all images in the dataset.

    Args:
        images: Array of grayscale images (N, H, W)
        show_progress: Whether to show progress bar

    Returns:
        Feature matrix (N, num_features)
    """
    features_list = []

    # Use tqdm for progress bar
    iterator = tqdm(images, desc="Extracting HOG features") if show_progress else images

    for image in iterator:
        features = extract_hog_features(image)
        features_list.append(features)

    return np.array(features_list)


class BaselineModel:
    """
    Baseline classifier using HOG features and SVM.

    This class provides a complete pipeline:
    1. Feature extraction (HOG)
    2. Feature scaling (StandardScaler)
    3. Classification (SVM with RBF kernel)

    Attributes:
        scaler: StandardScaler for feature normalization
        classifier: SVM classifier
    """

    def __init__(self):
        # StandardScaler normalizes features to zero mean, unit variance
        # This is important for SVM which is sensitive to feature scales
        self.scaler = StandardScaler()

        # SVM with RBF kernel - good for non-linear classification
        # C=10: moderate regularization
        # gamma='scale': automatic gamma based on feature variance
        self.classifier = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',  # Handle class imbalance
            random_state=RANDOM_SEED
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the baseline model.

        Args:
            X: Feature matrix (N, num_features)
            y: Labels (N,)
        """
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        print("Training SVM classifier...")
        self.classifier.fit(X_scaled, y)
        print("Training complete!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix (N, num_features)

        Returns:
            Predicted labels (N,)
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy on given data.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Accuracy score (0-1)
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.score(X_scaled, y)

    def save(self, filepath: Path) -> None:
        """Save model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'classifier': self.classifier}, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: Path) -> None:
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.classifier = data['classifier']
        print(f"Model loaded from {filepath}")


def train_baseline_model() -> Tuple[BaselineModel, Dict]:
    """
    Complete pipeline to train and evaluate the baseline model.

    Returns:
        Tuple of (trained model, evaluation results dict)
    """
    from dataset import load_images_for_baseline

    print("=" * 60)
    print("BASELINE MODEL: HOG + SVM")
    print("=" * 60)

    # Step 1: Load images
    print("\n[1/4] Loading images...")
    images, labels = load_images_for_baseline()
    print(f"Loaded {len(images)} images of shape {images[0].shape}")

    # Step 2: Extract HOG features
    print("\n[2/4] Extracting HOG features...")
    features = extract_features_from_dataset(images)
    print(f"Feature vector size: {features.shape[1]}")

    # Step 3: Split data
    print("\n[3/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Step 4: Train and evaluate
    print("\n[4/4] Training model...")
    model = BaselineModel()
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}")

    # Detailed classification report
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_CLASSES))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save model
    model.save(CHECKPOINT_DIR / "baseline_model.pkl")

    results = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'ground_truth': y_test
    }

    return model, results


if __name__ == "__main__":
    model, results = train_baseline_model()
