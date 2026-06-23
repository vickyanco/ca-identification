# file: utils/grad_cam.py
# description: Grad-CAM visualization for 2D and 3D CNNs to interpret model predictions.
# author: María Victoria Anconetani

import numpy as np
import tensorflow as tf
import cv2


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability."""

    def __init__(self, model, layer_name):
        """
        Args:
            model: Keras model to analyze.
            layer_name: Name of the layer to use for generating attention maps
                       (typically the last conv layer before global pooling).
        """
        self.model = model
        self.layer_name = layer_name
        self.layer = model.get_layer(layer_name)


    def __call__(self, img_array, class_idx=None):
        """
        Compute Grad-CAM heatmap for an image.

        Args:
            img_array: Input image (4D array: batch_size=1, spatial dims, channels).
            class_idx: Class index for which to generate Grad-CAM (default: argmax).

        Returns:
            heatmap: Numpy array of shape (spatial_dims) with activation importance.
        """
        img_array = tf.cast(img_array, tf.float32)
        layer_idx = next(i for i, l in enumerate(self.model.layers) if l.name == self.layer_name)

        with tf.GradientTape() as tape:
            x = img_array
            for i, layer in enumerate(self.model.layers):
                x = layer(x, training=False)
                if i == layer_idx:
                    conv_outputs = x
                    tape.watch(conv_outputs)

            predictions = x
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_channel = predictions[:, class_idx]

        grads = tape.gradient(class_channel, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=tuple(range(len(grads.shape) - 1)))
        conv_output_single = conv_outputs[0]
        heatmap = conv_output_single @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    def overlay_heatmap(self, img_2d, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay a heatmap on a 2D image.

        Args:
            img_2d: 2D input image (H, W), values in [0, 1] or [0, 255].
            heatmap: 2D heatmap (H, W), values in [0, 1].
            alpha: Transparency of the overlay.
            colormap: OpenCV colormap ID.

        Returns:
            overlay: RGB image with heatmap overlay.
        """
        # Normalize image to [0, 255] if needed
        if img_2d.max() <= 1.0:
            img_2d = (img_2d * 255).astype(np.uint8)
        else:
            img_2d = img_2d.astype(np.uint8)

        # Ensure heatmap is in [0, 255]
        heatmap_scaled = (heatmap * 255).astype(np.uint8)

        # Resize heatmap to match image dimensions if needed
        if heatmap_scaled.shape != img_2d.shape:
            heatmap_scaled = cv2.resize(
                heatmap_scaled, (img_2d.shape[1], img_2d.shape[0])
            )

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_scaled, colormap)

        # Blend: overlay heatmap on grayscale image (convert to 3-channel)
        img_bgr = cv2.cvtColor(img_2d, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay
