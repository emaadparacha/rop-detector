"""
ROP detection web portal.

Run locally:
    python web/app.py
    # then open http://127.0.0.1:5000

The portal exposes:
    GET  /          drag and drop upload form
    POST /predict   returns a JSON response with prediction and a base64
                    encoded heatmap overlay
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from inference import RopInference  # noqa: E402

app = Flask(__name__, static_folder="static", template_folder="templates")

# Lazy load the model so importing the module does not crash if the model
# has not been trained yet. The first prediction request triggers loading.
_engine: RopInference | None = None


def get_engine() -> RopInference:
    global _engine
    if _engine is None:
        _engine = RopInference()
    return _engine


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Use the 'image' form field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        pil = Image.open(file.stream).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not read image: {exc}"}), 400

    ga_weeks_raw = _parse_optional_float(request.form.get("gestational_age_weeks"))
    ga_days_raw = _parse_optional_float(request.form.get("gestational_age_days"))
    if ga_weeks_raw is None and ga_days_raw is None:
        gestational_age_days = None
    else:
        gestational_age_days = (ga_weeks_raw or 0) * 7 + (ga_days_raw or 0)
    birth_weight_g = _parse_optional_float(request.form.get("birth_weight_g"))

    try:
        engine = get_engine()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    result = engine.predict(
        image=pil,
        gestational_age_days=gestational_age_days,
        birth_weight_g=birth_weight_g,
        compute_heatmap=True,
    )

    heatmap_b64 = _pil_to_base64(result["heatmap_image"]) if result["heatmap_image"] else None
    original_b64 = _pil_to_base64(pil)

    return jsonify(
        {
            "probability_rop_percent": round(result["probability_rop"] * 100, 1),
            "probability_not_rop_percent": round(result["probability_not_rop"] * 100, 1),
            "predicted_label": result["predicted_label"],
            "confidence_percent": result["confidence_percent"],
            "metadata_used": result["metadata_used"],
            "reasons_for_rop": result["reasons"]["for_rop"],
            "reasons_against_rop": result["reasons"]["against_rop"],
            "heatmap_png_base64": heatmap_b64,
            "original_png_base64": original_b64,
            "inputs": {
                "gestational_age_days": gestational_age_days,
                "birth_weight_g": birth_weight_g,
            },
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
