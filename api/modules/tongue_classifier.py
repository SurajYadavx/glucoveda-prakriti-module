# api/modules/tongue_classifier.py

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODEL_PATH    = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'tongue_model.pth')
_METADATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'tongue_metadata.json')

# ── Load model once at import time ────────────────────────────────────────────
_model     = None
_device    = None
_transform = None
_metadata  = None


def _load_model():
    global _model, _device, _transform, _metadata

    if _model is not None:
        return  # already loaded

    with open(_METADATA_PATH) as f:
        _metadata = json.load(f)

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Rebuild same architecture as training
    base = models.resnet50(weights=None)
    base.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    base.load_state_dict(torch.load(_METADATA_PATH.replace('tongue_metadata.json', 'tongue_model.pth'),
                                    map_location=_device))
    base.eval()
    _model = base.to(_device)

    img_size = _metadata.get('img_size', 224)
    norm     = _metadata.get('normalize', {
        'mean': [0.485, 0.456, 0.406],
        'std':  [0.229, 0.224, 0.225]
    })

    _transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm['mean'], norm['std']),
    ])

    print(f"[tongue_classifier] Model loaded on {_device} | "
          f"Best acc: {_metadata.get('best_val_acc')}%")


def _validate_tongue_image(image_bgr: np.ndarray) -> tuple[bool, str]:
    """
    Basic quality checks before inference.
    Returns (is_valid, reason_if_invalid)
    """
    if image_bgr is None or image_bgr.size == 0:
        return False, "Empty image"

    h, w = image_bgr.shape[:2]
    if h < 80 or w < 80:
        return False, "Image too small — get closer"

    # Check image is not mostly black (no tongue detected)
    gray       = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    non_black  = np.sum(gray > 30)
    total_px   = h * w
    if non_black / total_px < 0.10:
        return False, "Tongue not visible — show tongue clearly"

    # Check for reddish/pink hue (tongue color sanity)
    hsv        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    avg_h      = float(np.mean(hsv[:, :, 0][gray > 30]))
    if avg_h > 50 and avg_h < 150:
        return False, "Not a tongue image — wrong color range detected"

    return True, "ok"


def analyze_tongue(image_bgr: np.ndarray = None,
                   image_path: str = None) -> dict:
    """
    Main function. Accepts either:
      - image_bgr  : OpenCV BGR frame (from webcam)
      - image_path : file path string (for testing)

    Returns structured tongue analysis dict with dosha_scores.
    """
    try:
        _load_model()

        # ── Load image ────────────────────────────────────────────────────────
        if image_path:
            image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {"error": "No image provided", "dosha_scores": None}

        # ── Quality check ─────────────────────────────────────────────────────
        valid, reason = _validate_tongue_image(image_bgr)
        if not valid:
            return {"error": reason, "dosha_scores": None}

        # ── Preprocess ────────────────────────────────────────────────────────
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor  = _transform(pil_img).unsqueeze(0).to(_device)

        # ── Inference ─────────────────────────────────────────────────────────
        with torch.no_grad():
            logits = _model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        vata_prob  = float(probs[0])
        pitta_prob = float(probs[1])

        # ── 2-class → 3-dosha mapping ─────────────────────────────────────────
        # If model is very confident about one class, residual goes to kapha
        # If borderline (both ~0.5), split residual to all three equally
        confidence_gap = abs(vata_prob - pitta_prob)

        if confidence_gap > 0.4:
            # Strong prediction — kapha gets the residual
            kapha_prob = max(0.0, 1.0 - vata_prob - pitta_prob)
        else:
            # Borderline — distribute residual more evenly
            kapha_prob  = 0.15
            vata_prob   = vata_prob  * 0.85
            pitta_prob  = pitta_prob * 0.85

        # Normalize to sum = 1.0
        total      = vata_prob + pitta_prob + kapha_prob
        vata_prob  = round(vata_prob  / total, 4)
        pitta_prob = round(pitta_prob / total, 4)
        kapha_prob = round(kapha_prob / total, 4)

        predicted_class = "vata" if probs[0] > probs[1] else "pitta"

        # ── HSV features for additional context ───────────────────────────────
        hsv     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) > 30
        avg_hsv = [
            round(float(np.mean(hsv[:, :, i][mask])), 1)
            for i in range(3)
        ]

        return {
            "predicted_class":  predicted_class,
            "vata_raw_prob":    round(float(probs[0]), 4),
            "pitta_raw_prob":   round(float(probs[1]), 4),
            "tongue_hsv":       avg_hsv,
            "confidence_gap":   round(confidence_gap, 3),
            "dosha_scores": {
                "vata":  vata_prob,
                "pitta": pitta_prob,
                "kapha": kapha_prob,
            }
        }

    except Exception as e:
        return {"error": f"Tongue analysis failed: {str(e)}", "dosha_scores": None}