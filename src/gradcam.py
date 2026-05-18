"""
Grad-CAM implementation for the RopModel.

Grad-CAM produces a heatmap highlighting which spatial regions in the
input image contributed most to the model's decision. We hook the last
convolutional block of EfficientNet-B0 and weight its feature maps by
the gradient of the logit with respect to those maps.

The output is a single channel float array in [0, 1] of size 224 x 224
(matching the model input). Use overlay_heatmap() to compose it on top of
the original image.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        # EfficientNet-B0's features is a Sequential. The last block is index 8.
        self.target_layer = model.backbone.features[-1]
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def compute(
        self,
        image_tensor: torch.Tensor,
        meta_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """Return a [H, W] heatmap normalized to [0, 1].

        image_tensor: shape [1, 3, H, W]
        meta_tensor:  shape [1, META_DIM]
        target_class: 1 for ROP, 0 for not-ROP. The sign of the gradient is
                      flipped for the not-ROP case so the heatmap always
                      indicates pixels that pushed the prediction toward
                      the chosen class.
        """
        self.model.eval()
        self.model.zero_grad()
        image_tensor.requires_grad_(True)
        logit = self.model(image_tensor, meta_tensor)
        score = logit if target_class == 1 else -logit
        score.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations.")

        # Global-average pooled gradients give per-channel weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(image_tensor.shape[2], image_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_min = float(cam_np.min())
        cam_max = float(cam_np.max())
        if cam_max - cam_min < 1e-8:
            return np.zeros_like(cam_np)
        return (cam_np - cam_min) / (cam_max - cam_min)


def overlay_heatmap(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Compose a JET-like colored heatmap on top of pil_image.

    pil_image: original (any size) RGB image.
    heatmap:   [H, W] float in [0, 1], will be resized to match pil_image.
    Returns an RGB PIL Image with the same size as pil_image.
    """
    base = pil_image.convert("RGB")
    h_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        base.size, resample=Image.BILINEAR
    )
    cam_arr = np.array(h_img, dtype=np.float32) / 255.0
    colored = _jet_colormap(cam_arr)  # [H, W, 3] in [0, 1]
    base_arr = np.array(base, dtype=np.float32) / 255.0
    out = (1 - alpha) * base_arr + alpha * colored
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """A simple Jet-style colormap that does not depend on matplotlib."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)
