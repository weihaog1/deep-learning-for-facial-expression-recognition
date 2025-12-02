"""
Neural Network Models for Facial Expression Recognition.

This module provides:
1. CustomCNN - A CNN built from scratch for the task
2. TransferLearningModel - ResNet18 pretrained on ImageNet, fine-tuned for FER

Both models output predictions for 7 emotion classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import NUM_CLASSES, CNN_DROPOUT, CNN_IMG_SIZE, CNN_CHANNELS


class ConvBlock(nn.Module):
    """
    A reusable convolutional block.

    Structure: Conv2D -> BatchNorm -> LeakyReLU -> MaxPool

    This pattern is standard in CNNs:
    - Conv2D: Learns spatial features (edges, textures, patterns)
    - BatchNorm: Stabilizes training, allows higher learning rates
    - LeakyReLU: Non-linearity that prevents dying neurons (improvement over ReLU)
    - MaxPool: Reduces spatial dimensions, adds translation invariance

    Using LeakyReLU instead of ReLU (from CS178 other group's findings):
    - Prevents "dying ReLU" problem where neurons output 0 for all inputs
    - Allows small gradient when input is negative
    - Improved their accuracy from 91% to 95%
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # Same padding to preserve size
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # LeakyReLU: f(x) = x if x > 0, else 0.01 * x
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)  # LeakyReLU instead of ReLU
        x = self.pool(x)
        return x


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for facial expression recognition.

    Architecture (based on other CS178 group's successful model):
    - 3 Convolutional blocks (32 -> 64 -> 32 filters)
    - Global average pooling (reduces parameters, prevents overfitting)
    - 2 Fully connected layers with dropout

    Input: GRAYSCALE images of size 48x48 (much faster than 224x224 RGB!)
    Output: 7 class logits (one per emotion)

    Improvements applied:
    - LeakyReLU instead of ReLU (prevents dying neurons)
    - Batch normalization (faster convergence)
    - Dropout 0.2 (from other group's findings)
    - Smart oversampling for class balance
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.2,
                 in_channels: int = CNN_CHANNELS):
        super().__init__()

        # Convolutional layers (like other CS178 group: 32 -> 64 -> 32)
        # Each block halves the spatial dimensions
        self.conv1 = ConvBlock(in_channels, 32)  # 48 -> 24
        self.conv2 = ConvBlock(32, 64)           # 24 -> 12
        self.conv3 = ConvBlock(64, 32)           # 12 -> 6

        # Global Average Pooling
        # Reduces each channel to a single value (6x6 -> 1x1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head with LeakyReLU
        self.fc1 = nn.Linear(32, 64)
        self.fc_activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout)  # 0.2 from other group
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 32, 1, 1) -> (batch, 32)

        # Classification with LeakyReLU
        x = self.fc_activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps from the last conv layer (for Grad-CAM).

        Returns the output of conv3 before global pooling.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class TransferLearningModel(nn.Module):
    """
    Transfer learning model using ResNet18 pretrained on ImageNet.

    Strategy:
    - Use ResNet18's convolutional layers as a feature extractor
    - Replace the final fully connected layer for our 7 classes
    - Optionally freeze early layers to preserve learned features

    Why ResNet18?
    - Pretrained on 1M+ images, learned rich visual features
    - Residual connections allow deeper networks without vanishing gradients
    - Good balance of accuracy and computational cost
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True
    ):
        super().__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze backbone layers if specified
        # This preserves the pretrained features and only trains the classifier
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        # ResNet18's fc layer outputs 1000 classes (ImageNet)
        # We need 7 classes for emotions
        num_features = self.backbone.fc.in_features  # 512 for ResNet18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone layers for fine-tuning.

        Call this after initial training to fine-tune the entire network
        with a lower learning rate.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen - all layers are now trainable")

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps from the last conv layer (for Grad-CAM).

        Returns the output before the average pooling layer.
        """
        # ResNet18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x


def get_model(model_type: str = 'custom', **kwargs) -> nn.Module:
    """
    Factory function to create a model.

    Args:
        model_type: Either 'custom' or 'transfer'
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Initialized model (on CPU)
    """
    if model_type == 'custom':
        return CustomCNN(**kwargs)
    elif model_type == 'transfer':
        return TransferLearningModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing CustomCNN (48x48 grayscale)...")
    custom_model = CustomCNN()
    x_cnn = torch.randn(4, CNN_CHANNELS, CNN_IMG_SIZE, CNN_IMG_SIZE)  # Batch of 4 grayscale images
    out = custom_model(x_cnn)
    print(f"  Input shape:  {x_cnn.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters:   {count_parameters(custom_model):,}")

    print("\nTesting TransferLearningModel (224x224 RGB)...")
    transfer_model = TransferLearningModel(freeze_backbone=True)
    x_transfer = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
    out = transfer_model(x_transfer)
    print(f"  Input shape:  {x_transfer.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Trainable parameters (frozen): {count_parameters(transfer_model):,}")

    transfer_model.unfreeze_backbone()
    print(f"  Trainable parameters (unfrozen): {count_parameters(transfer_model):,}")
