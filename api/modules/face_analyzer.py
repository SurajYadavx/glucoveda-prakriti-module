# api/modules/face_analyzer.py

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime
import json

mp_face_mesh = mp.solutions.face_mesh


# ─── Landmark Index Reference ────────────────────────────────────────────────
# Face width  : landmarks 234 (right cheek) ↔ 454 (left cheek)
# Face height : landmarks 10 (forehead)     ↔ 152 (chin)
# Left eye    : landmarks 33, 133, 159, 145, 153, 380
# Right eye   : landmarks 362, 263, 386, 374, 380, 373
# Lip top     : landmark 13
# Lip bottom  : landmark 14
# ─────────────────────────────────────────────────────────────────────────────


def _landmark_dist(lm, i, j, w, h):
    """Euclidean distance between two landmarks in pixel space."""
    a = np.array([lm[i].x * w, lm[i].y * h])
    b = np.array([lm[j].x * w, lm[j].y * h])
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(lm, indices, w, h):
    """EAR = vertical / horizontal — small value = small/narrow eye."""
    p1 = np.array([lm[indices[1]].x * w, lm[indices[1]].y * h])
    p2 = np.array([lm[indices[2]].x * w, lm[indices[2]].y * h])
    p3 = np.array([lm[indices[3]].x * w, lm[indices[3]].y * h])
    p4 = np.array([lm[indices[4]].x * w, lm[indices[4]].y * h])
    p5 = np.array([lm[indices[0]].x * w, lm[indices[0]].y * h])
    p6 = np.array([lm[indices[5]].x * w, lm[indices[5]].y * h])
    vertical   = (np.linalg.norm(p2 - p4) + np.linalg.norm(p3 - p5)) / 2
    horizontal = np.linalg.norm(p1 - p6)
    return float(vertical / (horizontal + 1e-6))


def _classify_face_shape(face_index: float) -> str:
    """face_index = height / width."""
    if face_index > 1.40:
        return "long_angular"   # Vata
    elif face_index < 1.10:
        return "round_full"     # Kapha
    else:
        return "oval_medium"    # Pitta


def _classify_eye_size(ear: float) -> str:
    if ear < 0.22:
        return "small"    # Vata
    elif ear > 0.30:
        return "large"    # Kapha
    else:
        return "medium"   # Pitta


def _skin_tone_dosha(hsv_mean: np.ndarray) -> dict:
    """
    Indian skin tone HSV ranges (BGR camera space):
    Vata  : Dusky/dark brown   H:10-25, S:60-120, V:60-130
    Pitta : Warm reddish-fair  H:5-18,  S:80-160, V:130-200
    Kapha : Pale/cool/oily     H:15-30, S:30-80,  V:160-230
    
    Sanity check: if H > 40, not a skin tone → return neutral
    """
    h, s, v = float(hsv_mean[0]), float(hsv_mean[1]), float(hsv_mean[2])

    # Sanity check — if hue is outside skin range, return neutral
    if h > 40 or h < 0:
        return {"vata": 0.34, "pitta": 0.33, "kapha": 0.33}

    vata_score  = 0.0
    pitta_score = 0.0
    kapha_score = 0.0

    # Darkness → Vata
    if v < 130:
        vata_score += 0.5
    if 10 <= h <= 25 and s > 60:
        vata_score += 0.3

    # Redness + warmth + brightness → Pitta
    if h <= 18 and s > 80 and v > 130:
        pitta_score += 0.7
    elif h <= 18 and v > 130:
        pitta_score += 0.4

    # Pale + cool + low saturation → Kapha
    if v > 160 and s < 80:
        kapha_score += 0.6
    if h > 18 and v > 140:
        kapha_score += 0.2

    total = vata_score + pitta_score + kapha_score
    if total < 0.1:
        return {"vata": 0.34, "pitta": 0.33, "kapha": 0.33}

    return {
        "vata":  round(vata_score  / total, 3),
        "pitta": round(pitta_score / total, 3),
        "kapha": round(kapha_score / total, 3),
    }


def _get_skin_hsv(image_bgr: np.ndarray, landmarks, w: int, h: int) -> np.ndarray:
    """
    Sample skin tone from cheek region — more reliable than forehead.
    Uses left cheek area between landmarks 116, 117, 118, 119, 100, 36.
    Falls back to a direct pixel sample if region is invalid.
    """
    # Cheek landmarks — lower face, avoids hair and background
    cheek_lms = [116, 117, 118, 119, 100, 36, 205, 187]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                    for i in cheek_lms])

    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)

    # Validate: at least 100 pixels in mask
    if cv2.countNonZero(mask) < 100:
        # Fallback: sample center of face bounding box
        cx = int(landmarks[1].x * w)
        cy = int(landmarks[1].y * h)
        patch = image_bgr[max(0, cy-20):cy+20, max(0, cx-20):cx+20]
        if patch.size == 0:
            return np.array([15.0, 80.0, 150.0])  # default warm skin
        image_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return np.array(cv2.mean(image_hsv)[:3])

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv  = cv2.mean(image_hsv, mask=mask)[:3]
    return np.array(mean_hsv)


def analyze_face(image_bgr: np.ndarray) -> dict:
    """
    Main function — takes BGR image, returns structured face analysis dict.
    """
    h, w = image_bgr.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return {"error": "No face detected", "dosha_scores": None}

        lm = results.multi_face_landmarks[0].landmark

        # ── Face shape ───────────────────────────────────────────────────────
        face_width  = _landmark_dist(lm, 234, 454, w, h)
        face_height = _landmark_dist(lm, 10,  152, w, h)
        face_index  = face_height / (face_width + 1e-6)
        face_shape  = _classify_face_shape(face_index)

        # ── Eye size ─────────────────────────────────────────────────────────
        left_eye_idx  = [33, 133, 159, 145, 153, 380]
        right_eye_idx = [362, 263, 386, 374, 380, 373]
        ear_left  = _eye_aspect_ratio(lm, left_eye_idx,  w, h)
        ear_right = _eye_aspect_ratio(lm, right_eye_idx, w, h)
        avg_ear   = (ear_left + ear_right) / 2
        eye_size  = _classify_eye_size(avg_ear)

        # ── Skin tone ─────────────────────────────────────────────────────────
        hsv_mean   = _get_skin_hsv(image_bgr, lm, w, h)
        skin_dosha = _skin_tone_dosha(hsv_mean)

        # ── Per-feature dosha vote ────────────────────────────────────────────
        face_shape_scores = {
            "long_angular": {"vata": 0.7, "pitta": 0.2, "kapha": 0.1},
            "oval_medium":  {"vata": 0.2, "pitta": 0.6, "kapha": 0.2},
            "round_full":   {"vata": 0.1, "pitta": 0.2, "kapha": 0.7},
        }[face_shape]

        eye_size_scores = {
            "small":  {"vata": 0.7, "pitta": 0.2, "kapha": 0.1},
            "medium": {"vata": 0.2, "pitta": 0.6, "kapha": 0.2},
            "large":  {"vata": 0.1, "pitta": 0.2, "kapha": 0.7},
        }[eye_size]

        # Weighted fusion of 3 face sub-signals
        weights = {"face_shape": 0.35, "eye_size": 0.30, "skin_tone": 0.35}

        combined = {}
        for dosha in ["vata", "pitta", "kapha"]:
            combined[dosha] = round(
                face_shape_scores[dosha] * weights["face_shape"] +
                eye_size_scores[dosha]   * weights["eye_size"]   +
                skin_dosha[dosha]        * weights["skin_tone"],
                3
            )

        # Normalise to sum = 1
        total = sum(combined.values())
        combined = {k: round(v / total, 3) for k, v in combined.items()}

        return {
            "face_shape":        face_shape,
            "face_index":        round(face_index, 3),
            "eye_size":          eye_size,
            "eye_aspect_ratio":  round(avg_ear, 3),
            "skin_hsv":          [round(x, 1) for x in hsv_mean.tolist()],
            "skin_dosha_raw":    skin_dosha,
            "dosha_scores": {
                "vata":  combined["vata"],
                "pitta": combined["pitta"],
                "kapha": combined["kapha"],
            },
        }