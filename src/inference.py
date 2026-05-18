"""
Inference utilities for the trained ROP classifier.

Used by both the command line and the Flask web portal.

Example:
    from inference import RopInference
    engine = RopInference()
    result = engine.predict(
        image_path="samples/rop/Stage_2_ROP_44.jpg",
        gestational_age_days=200,
        birth_weight_g=1100,
    )
    print(result["probability_rop"])
    result["heatmap_image"].save("heatmap.png")

The class is safe to instantiate once at server startup and reuse for
many requests.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from gradcam import GradCAM, overlay_heatmap
from model import RopModel, normalize_meta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


class RopInference:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        normalization_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        model_path = Path(model_path) if model_path else MODEL_DIR / "best_model.pt"
        norm_path = Path(normalization_path) if normalization_path else MODEL_DIR / "normalization.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                "Train the model first by running `python src/train.py`."
            )
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found at {norm_path}. "
                "Re-run training so this file gets created."
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.config = json.loads(norm_path.read_text())
        self.image_size = int(self.config.get("image_size", 224))

        self.model = RopModel(pretrained=False)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.config["imagenet_mean"], self.config["imagenet_std"]
                ),
            ]
        )

    def _load_image(self, image: Union[str, Path, Image.Image, bytes]) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image)!r}")

    def predict(
        self,
        image: Union[str, Path, Image.Image, bytes],
        gestational_age_days: Optional[float] = None,
        birth_weight_g: Optional[float] = None,
        compute_heatmap: bool = True,
    ) -> dict:
        """Run a prediction on a single image and return a result dict.

        Keys in the returned dict:
            probability_rop      float in [0, 1]
            probability_not_rop  float in [0, 1]
            predicted_label      'rop' or 'not_rop'
            confidence_percent   highest class probability x 100, rounded
            heatmap_image        PIL.Image with heatmap overlaid on input
                                 (only if compute_heatmap is True)
            heatmap_array        numpy [H, W] in [0, 1] of the raw cam
            metadata_used        bool, whether age/weight were supplied
            reasons              dict with 'for_rop' and 'against_rop' lists
        """
        pil_image = self._load_image(image)
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        meta_vec = normalize_meta(gestational_age_days, birth_weight_g, self.config)
        meta_tensor = torch.tensor(meta_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        metadata_used = bool(meta_vec[2] < 0.5)  # missing flag is 0 when both provided

        with torch.no_grad():
            logit = self.model(image_tensor, meta_tensor).item()
        prob_rop = float(1.0 / (1.0 + np.exp(-logit)))
        prob_not_rop = 1.0 - prob_rop
        predicted_label = "rop" if prob_rop >= 0.5 else "not_rop"
        confidence_percent = round(max(prob_rop, prob_not_rop) * 100, 1)

        heatmap_image = None
        heatmap_array = None
        if compute_heatmap:
            with GradCAM(self.model) as cam:
                heatmap_array = cam.compute(
                    image_tensor.clone(),
                    meta_tensor.clone(),
                    target_class=1 if predicted_label == "rop" else 0,
                )
            heatmap_image = overlay_heatmap(pil_image, heatmap_array)

        reasons = self._explain(
            prob_rop=prob_rop,
            predicted_label=predicted_label,
            heatmap_array=heatmap_array,
            gestational_age_days=gestational_age_days,
            birth_weight_g=birth_weight_g,
        )

        return {
            "probability_rop": prob_rop,
            "probability_not_rop": prob_not_rop,
            "predicted_label": predicted_label,
            "confidence_percent": confidence_percent,
            "heatmap_image": heatmap_image,
            "heatmap_array": heatmap_array,
            "metadata_used": metadata_used,
            "reasons": reasons,
        }

    def _explain(
        self,
        prob_rop: float,
        predicted_label: str,
        heatmap_array: Optional[np.ndarray],
        gestational_age_days: Optional[float],
        birth_weight_g: Optional[float],
    ) -> dict:
        """Generate human readable reasons for and against an ROP diagnosis."""
        for_rop: list[str] = []
        against_rop: list[str] = []

        if prob_rop >= 0.8:
            for_rop.append(
                f"The CNN gave a high ROP probability of {prob_rop * 100:.1f} percent."
            )
        elif prob_rop >= 0.5:
            for_rop.append(
                f"The CNN leans toward ROP with a probability of {prob_rop * 100:.1f} percent."
            )
        elif prob_rop >= 0.2:
            against_rop.append(
                f"The CNN leans away from ROP with an ROP probability of only "
                f"{prob_rop * 100:.1f} percent."
            )
        else:
            against_rop.append(
                f"The CNN is confident this is not ROP. ROP probability is "
                f"{prob_rop * 100:.1f} percent."
            )

        if heatmap_array is not None:
            region = describe_hot_region(heatmap_array)
            if region and predicted_label == "rop":
                for_rop.append(
                    "Grad-CAM highlights suspicious features in the "
                    f"{region} of the image where the ridge or neovascular "
                    "pattern typical of active ROP would appear."
                )
            elif region and predicted_label == "not_rop":
                against_rop.append(
                    "The most informative pixels for this prediction lie in "
                    f"the {region}, and the model interpreted them as benign "
                    "retinal background rather than ROP changes."
                )

        if gestational_age_days is not None:
            weeks = gestational_age_days / 7.0
            if weeks <= 30:
                for_rop.append(
                    f"Gestational age at birth was {weeks:.1f} weeks, which is "
                    "in the highest risk band for ROP (most cases occur below 31 weeks)."
                )
            elif weeks <= 32:
                for_rop.append(
                    f"Gestational age at birth was {weeks:.1f} weeks, in the "
                    "moderate risk band for ROP."
                )
            else:
                against_rop.append(
                    f"Gestational age at birth was {weeks:.1f} weeks, above "
                    "the usual ROP screening threshold of 32 weeks."
                )

        if birth_weight_g is not None:
            if birth_weight_g <= 1250:
                for_rop.append(
                    f"Birth weight was {birth_weight_g:.0f} g, below the "
                    "1500 g screening threshold, which raises ROP risk."
                )
            elif birth_weight_g <= 1500:
                for_rop.append(
                    f"Birth weight was {birth_weight_g:.0f} g, within the "
                    "ROP screening range (under 1500 g)."
                )
            else:
                against_rop.append(
                    f"Birth weight was {birth_weight_g:.0f} g, above the "
                    "1500 g threshold typically used for ROP screening."
                )

        return {"for_rop": for_rop, "against_rop": against_rop}


def describe_hot_region(heatmap: np.ndarray, threshold: float = 0.6) -> str:
    """Return a short string describing where the heatmap is most intense.

    Splits the heatmap into a 3 x 3 grid and reports the cell with the
    highest activation. Returns '' if no region is sufficiently activated.
    """
    if heatmap.max() < threshold:
        return ""
    h, w = heatmap.shape
    rows = ["top", "middle", "bottom"]
    cols = ["left", "center", "right"]
    r_size = h // 3
    c_size = w // 3
    best = (-1.0, "")
    for ri, r_name in enumerate(rows):
        for ci, c_name in enumerate(cols):
            block = heatmap[
                ri * r_size : (ri + 1) * r_size,
                ci * c_size : (ci + 1) * c_size,
            ]
            mean_val = float(block.mean())
            if mean_val > best[0]:
                label = f"{r_name} {c_name}".replace("middle center", "center")
                best = (mean_val, label)
    return best[1]
