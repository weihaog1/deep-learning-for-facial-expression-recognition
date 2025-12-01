"""
Grad-CAM: Gradient-weighted Class Activation Mapping

This module implements Grad-CAM for visualizing which parts of an image
the CNN focuses on when making predictions.

How Grad-CAM works:
1. Forward pass: Get feature maps from last conv layer
2. Backward pass: Get gradients of the predicted class score
3. Weight feature maps by their gradient importance
4. Create a heatmap showing important regions

This helps us understand:
- Is the model looking at the face or background?
- Which facial features are important for each emotion?
- Is the model learning meaningful patterns?
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import (
    IDX_TO_EMOTION, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CHECKPOINT_DIR, OUTPUT_DIR
)
from models import get_model
from dataset import get_test_transform


class GradCAM:
    """
    Grad-CAM visualization for CNN models.

    This class hooks into the model to capture:
    - Feature maps from the target layer (forward pass)
    - Gradients flowing back to that layer (backward pass)

    Attributes:
        model: The CNN model to visualize
        target_layer: The layer to visualize (last conv layer)
    """

    def __init__(self, model: torch.nn.Module, model_type: str = 'custom'):
        self.model = model
        self.model.eval()
        self.model_type = model_type

        # Storage for hooks
        self.feature_maps = None
        self.gradients = None

        # Register hooks on the target layer
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Register forward and backward hooks on the target layer.

        For CustomCNN: hook on conv4 (last conv block)
        For TransferLearningModel: hook on layer4 (last ResNet block)
        """
        if self.model_type == 'custom':
            target_layer = self.model.conv4
        else:
            target_layer = self.model.backbone.layer4

        # Forward hook: save feature maps
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        # Backward hook: save gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            image: Input image tensor (1, 3, H, W) - already preprocessed
            target_class: Class to visualize (None = predicted class)

        Returns:
            Tuple of (heatmap, predicted_class, confidence)
            - heatmap: 2D numpy array (H, W) with values 0-1
            - predicted_class: The class index the model predicted
            - confidence: Softmax probability for predicted class
        """
        # Forward pass
        output = self.model(image)
        probs = F.softmax(output, dim=1)

        # Get target class (use predicted if not specified)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get gradients and feature maps
        gradients = self.gradients  # (1, C, H, W)
        feature_maps = self.feature_maps  # (1, C, H, W)

        # Global average pooling of gradients -> importance weights
        # Average over spatial dimensions (H, W)
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

        # Normalize to 0-1 range
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class, confidence


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized image tensor back to displayable format.

    Args:
        image: Normalized image tensor (3, H, W)

    Returns:
        RGB image as numpy array (H, W, 3) with values 0-255
    """
    # Reverse normalization
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image * std + mean

    # Convert to numpy and transpose to (H, W, C)
    image = image.permute(1, 2, 0).numpy()

    # Clip to valid range and convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay a heatmap on an image.

    Args:
        image: Original RGB image (H, W, 3)
        heatmap: Grad-CAM heatmap (H, W) with values 0-1
        alpha: Blending factor (0 = only image, 1 = only heatmap)

    Returns:
        Blended image with heatmap overlay
    """
    # Apply colormap to heatmap
    colormap = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
    colormap = (colormap * 255).astype(np.uint8)

    # Blend original image with heatmap
    blended = (1 - alpha) * image + alpha * colormap
    blended = blended.astype(np.uint8)

    return blended


def visualize_gradcam(
    model: torch.nn.Module,
    image_path: str,
    model_type: str = 'custom',
    save_path: Optional[str] = None
) -> None:
    """
    Generate and display Grad-CAM visualization for a single image.

    Args:
        model: Trained CNN model
        image_path: Path to the input image
        model_type: 'custom' or 'transfer'
        save_path: Optional path to save the visualization
    """
    device = next(model.parameters()).device

    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    transform = get_test_transform()
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    # Generate Grad-CAM
    gradcam = GradCAM(model, model_type)
    heatmap, predicted_class, confidence = gradcam.generate_heatmap(image_tensor)

    # Prepare display image
    display_image = denormalize_image(image_tensor.squeeze().cpu())
    overlay = overlay_heatmap(display_image, heatmap)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(display_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    emotion = IDX_TO_EMOTION[predicted_class]
    axes[2].set_title(f"Prediction: {emotion} ({confidence:.1%})")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_batch(
    model: torch.nn.Module,
    data_loader,
    model_type: str = 'custom',
    num_samples: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Generate Grad-CAM visualizations for multiple samples.

    Args:
        model: Trained CNN model
        data_loader: DataLoader with test images
        model_type: 'custom' or 'transfer'
        num_samples: Number of samples to visualize
        save_path: Optional path to save the visualization
    """
    device = next(model.parameters()).device
    gradcam = GradCAM(model, model_type)

    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))

    for i in range(num_samples):
        image = images[i:i+1]
        true_label = labels[i].item()

        # Generate Grad-CAM
        heatmap, pred_class, confidence = gradcam.generate_heatmap(image)

        # Prepare display
        display_image = denormalize_image(image.squeeze().cpu())
        overlay = overlay_heatmap(display_image, heatmap)

        # Plot
        axes[i, 0].imshow(display_image)
        axes[i, 0].set_title(f"True: {IDX_TO_EMOTION[true_label]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title("Grad-CAM")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(overlay)
        pred_emotion = IDX_TO_EMOTION[pred_class]
        correct = "✓" if pred_class == true_label else "✗"
        axes[i, 2].set_title(f"Pred: {pred_emotion} {correct} ({confidence:.1%})")
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Demo: Load a trained model and visualize some predictions
    import argparse

    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument('--model', type=str, default='custom',
                        choices=['custom', 'transfer'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint_path = args.checkpoint or (CHECKPOINT_DIR / f"{args.model}_best.pth")
    print(f"Loading model from {checkpoint_path}")

    model = get_model(args.model)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test data
    from dataset import create_data_loaders
    _, _, test_loader = create_data_loaders()

    # Visualize
    save_path = OUTPUT_DIR / f"gradcam_{args.model}.png"
    visualize_batch(model, test_loader, args.model, num_samples=8, save_path=str(save_path))
