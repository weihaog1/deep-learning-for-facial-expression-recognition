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

from config import NUM_CLASSES, CNN_DROPOUT


class ConvBlock(nn.Module):
    """
    A reusable convolutional block.

    Structure: Conv2D -> BatchNorm -> ReLU -> MaxPool

    This pattern is standard in CNNs:
    - Conv2D: Learns spatial features (edges, textures, patterns)
    - BatchNorm: Stabilizes training, allows higher learning rates
    - ReLU: Non-linearity for learning complex patterns
    - MaxPool: Reduces spatial dimensions, adds translation invariance
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # Same padding to preserve size
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for facial expression recognition.

    Architecture:
    - 4 Convolutional blocks (progressively increasing channels)
    - Global average pooling (reduces parameters, prevents overfitting)
    - 2 Fully connected layers with dropout

    Input: RGB images of size 224x224
    Output: 7 class logits (one per emotion)

    This architecture is designed to:
    - Be deep enough to learn complex facial features
    - Use dropout and batch norm to prevent overfitting on small dataset
    - Be computationally efficient with global average pooling
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = CNN_DROPOUT):
        super().__init__()

        # Convolutional layers
        # Each block doubles the channels and halves the spatial dimensions
        self.conv1 = ConvBlock(3, 32)      # 224 -> 112
        self.conv2 = ConvBlock(32, 64)     # 112 -> 56
        self.conv3 = ConvBlock(64, 128)    # 56 -> 28
        self.conv4 = ConvBlock(128, 256)   # 28 -> 14

        # Global Average Pooling
        # Reduces each channel to a single value (14x14 -> 1x1)
        # This is more robust than flattening and reduces overfitting
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 256, 1, 1) -> (batch, 256)

        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps from the last conv layer (for Grad-CAM).

        Returns the output of conv4 before global pooling.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
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
    print("Testing CustomCNN...")
    custom_model = CustomCNN()
    x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    out = custom_model(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters:   {count_parameters(custom_model):,}")

    print("\nTesting TransferLearningModel...")
    transfer_model = TransferLearningModel(freeze_backbone=True)
    out = transfer_model(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Trainable parameters (frozen): {count_parameters(transfer_model):,}")

    transfer_model.unfreeze_backbone()
    print(f"  Trainable parameters (unfrozen): {count_parameters(transfer_model):,}")
