# file: train_and_evaluate/visualize_grad_cam.py
# description: Grad-CAM visualization for T1 Mapping and LGE models to understand what they learned.
# author: María Victoria Anconetani
# date: 24/06/2025

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from backend.model.t1map_model import T1MappingCNN
from backend.model.lge_model import LGE_CNN
from backend.utils.grad_cam import GradCAM


def generate_synthetic_t1_image():
    """Generate a synthetic T1 mapping image (256, 256, 1) with a Gaussian blob."""
    img = np.random.randn(256, 256, 1).astype(np.float32) * 0.1
    y, x = np.ogrid[-1:1:256j, -1:1:256j]
    gaussian = np.exp(-(x**2 + y**2) / 0.3)[..., np.newaxis]
    img += gaussian
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def generate_synthetic_lge_image():
    """Generate a synthetic LGE image (172, 192, 12, 1) with volumetric pattern."""
    img = np.random.randn(172, 192, 12, 1).astype(np.float32) * 0.1
    i = np.arange(172)[:, np.newaxis, np.newaxis]
    j = np.arange(192)[np.newaxis, :, np.newaxis]
    k = np.arange(12)[np.newaxis, np.newaxis, :]
    x = (j - 96) / 96
    y = (i - 86) / 86
    z = (k - 6) / 6
    gaussian = np.exp(-(x**2 + y**2 + z**2) / 0.4)
    img[..., 0] += gaussian
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def find_last_conv_layer(model):
    """Find the name of the last convolutional layer in a model."""
    for layer in reversed(model.layers):
        layer_type = type(layer).__name__.lower()
        if "conv" in layer_type:
            return layer.name
    raise ValueError("No convolutional layer found in model")


def visualize_t1_grad_cam(model, layer_name=None, num_samples=4):
    """Generate and visualize Grad-CAM heatmaps for T1 model."""
    print("\n🎯 T1 Mapping - Grad-CAM Visualization")

    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    print(f"Using layer: {layer_name}")
    grad_cam = GradCAM(model, layer_name)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        img = generate_synthetic_t1_image()
        img_batch = np.expand_dims(img, axis=0)

        # Get prediction and heatmap
        pred_prob = model.predict(img_batch, verbose=0)[0, 0]
        heatmap = grad_cam(img_batch)

        # Normalize image for display
        img_display = (img[..., 0] - img[..., 0].min()) / (img[..., 0].max() - img[..., 0].min() + 1e-8)

        # Plot image
        axes[i, 0].imshow(img_display, cmap='gray')
        axes[i, 0].set_title(f'Input Image')
        axes[i, 0].axis('off')

        # Plot heatmap
        axes[i, 1].imshow(heatmap, cmap='hot')
        axes[i, 1].set_title(f'Grad-CAM Heatmap')
        axes[i, 1].axis('off')

        # Plot overlay
        overlay = grad_cam.overlay_heatmap(img[..., 0], heatmap, alpha=0.6)
        axes[i, 2].imshow(overlay[..., ::-1])  # BGR to RGB
        axes[i, 2].set_title(f'Overlay (P={pred_prob:.3f})')
        axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/t1_grad_cam.png", dpi=150, bbox_inches='tight')
    print("✅ Saved: visualizations/t1_grad_cam.png")
    plt.close()


def visualize_lge_grad_cam(model, layer_name=None, num_slices=3):
    """Generate and visualize Grad-CAM heatmaps for LGE model (volumetric slices)."""
    print("\n🎯 LGE - Grad-CAM Visualization")

    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    print(f"Using layer: {layer_name}")
    grad_cam = GradCAM(model, layer_name)

    # Generate one synthetic 3D sample
    img = generate_synthetic_lge_image()
    img_batch = np.expand_dims(img, axis=0)

    # Get prediction and 3D heatmap
    pred_prob = model.predict(img_batch, verbose=0)[0, 0]
    heatmap_3d = grad_cam(img_batch)  # Shape: (172, 192, 12)

    # Select middle slices along z-axis to visualize
    z_indices = np.linspace(1, 10, num_slices, dtype=int)

    fig, axes = plt.subplots(num_slices, 3, figsize=(12, 4 * num_slices))
    if num_slices == 1:
        axes = axes.reshape(1, -1)

    for row, z_idx in enumerate(z_indices):
        img_slice = img[..., z_idx, 0]
        heatmap_slice = heatmap_3d[..., z_idx]

        # Normalize for display
        img_display = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

        # Plot image slice
        axes[row, 0].imshow(img_display, cmap='gray')
        axes[row, 0].set_title(f'Slice z={z_idx}')
        axes[row, 0].axis('off')

        # Plot heatmap slice
        axes[row, 1].imshow(heatmap_slice, cmap='hot')
        axes[row, 1].set_title(f'Grad-CAM z={z_idx}')
        axes[row, 1].axis('off')

        # Plot overlay
        overlay = grad_cam.overlay_heatmap(img_slice, heatmap_slice, alpha=0.6)
        axes[row, 2].imshow(overlay[..., ::-1])  # BGR to RGB
        axes[row, 2].set_title(f'Overlay (P={pred_prob:.3f})')
        axes[row, 2].axis('off')

    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/lge_grad_cam.png", dpi=150, bbox_inches='tight')
    print("✅ Saved: visualizations/lge_grad_cam.png")
    plt.close()


if __name__ == "__main__":
    print("📊 Grad-CAM Visualization Pipeline")
    print("=" * 50)

    # Initialize models
    print("\n🔧 Building models...")
    t1_model = T1MappingCNN().model
    lge_model = LGE_CNN().model
    print("✅ Models initialized")

    # Try to load trained models if they exist
    if os.path.exists("t1_mapping_cnn_model.h5"):
        print("📂 Loading saved T1 model...")
        t1_model = tf.keras.models.load_model("t1_mapping_cnn_model.h5")
        # Rebuild by forward pass
        t1_model.predict(generate_synthetic_t1_image()[np.newaxis], verbose=0)

    if os.path.exists("lge_cnn_model.h5"):
        print("📂 Loading saved LGE model...")
        lge_model = tf.keras.models.load_model("lge_cnn_model.h5")
        # Rebuild by forward pass
        lge_model.predict(generate_synthetic_lge_image()[np.newaxis], verbose=0)

    # Generate visualizations
    visualize_t1_grad_cam(t1_model, num_samples=4)
    visualize_lge_grad_cam(lge_model, num_slices=3)

    print("\n✅ Visualization pipeline complete!")
