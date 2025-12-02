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


class CustomCNN(nn.Module):
    """
    Custom CNN architecture - EXACT replica of other CS178 group's model.

    From their paper:
    - 3 Conv layers: 32 -> 64 -> 32 filters
    - BatchNorm after each conv, before pooling
    - MaxPool (2x2) only after first TWO conv layers
    - LeakyReLU activation (increased accuracy from 91% to 95%)
    - Dropout 0.20
    - RMSprop optimizer

    Input: 50x50 grayscale images
    Output: 7 class logits

    Architecture:
        Conv1(32) -> BN -> LeakyReLU -> Pool    [50 -> 25]
        Conv2(64) -> BN -> LeakyReLU -> Pool    [25 -> 12]
        Conv3(32) -> BN -> LeakyReLU            [12 -> 12, no pool]
        Flatten -> FC -> Dropout -> Output
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.20,
                 in_channels: int = CNN_CHANNELS):
        super().__init__()

        # Conv Layer 1: 32 filters (extracts low-level features: edges, shadows)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 50 -> 25

        # Conv Layer 2: 64 filters (extracts facial features: eyes, nose, contours)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 25 -> 12

        # Conv Layer 3: 32 filters (compression layer - NO pooling)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        # No pooling after conv3 (from paper: "pooling layers after first two conv layers")

        # LeakyReLU activation (from paper: improved accuracy from 91% to 95%)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        # Classifier head
        # After conv3: 32 channels * 12 * 12 = 4608 features
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 12 * 12, 128)  # Hidden layer
        self.dropout = nn.Dropout(dropout)  # 0.20 from paper
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1: Conv -> BN -> LeakyReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool1(x)

        # Conv block 2: Conv -> BN -> LeakyReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool2(x)

        # Conv block 3: Conv -> BN -> LeakyReLU (NO pool)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps from the last conv layer (for Grad-CAM visualization).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

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
    print("Testing CustomCNN (50x50 grayscale - matching other CS178 group)...")
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
