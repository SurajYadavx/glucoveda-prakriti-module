# api/modules/body_analyzer.py

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _visible(lm, idx, threshold=0.6):
    return lm[idx].visibility > threshold


def _classify_build(shoulder_height_ratio: float) -> str:
    """
    shoulder_height_ratio = shoulder_width / body_height
    Thin  (Vata)  : < 0.22
    Medium(Pitta) : 0.22 – 0.28
    Broad (Kapha) : > 0.28
    """
    if shoulder_height_ratio < 0.22:
        return "thin"
    elif shoulder_height_ratio > 0.28:
        return "broad"
    else:
        return "medium"


def analyze_body(image_bgr: np.ndarray) -> dict:
    """
    Takes BGR full-body image → returns body build analysis and dosha scores.
    """
    h, w = image_bgr.shape[:2]

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results   = pose.process(image_rgb)

        if not results.pose_landmarks:
            return {"error": "No body detected", "dosha_scores": None}

        lm = results.pose_landmarks.landmark

        # Require key landmarks to be visible
        required = [11, 12, 23, 24, 0, 27, 28]
        if not all(_visible(lm, i) for i in required):
            return {"error": "Body not fully visible — step back more",
                    "dosha_scores": None}

        # ── Key points ───────────────────────────────────────────────────────
        def px(idx):
            return (lm[idx].x * w, lm[idx].y * h)

        l_shoulder = px(11)
        r_shoulder = px(12)
        l_hip      = px(23)
        r_hip      = px(24)
        nose       = px(0)
        l_ankle    = px(27)
        r_ankle    = px(28)

        # ── Measurements ─────────────────────────────────────────────────────
        shoulder_width = _dist(l_shoulder, r_shoulder)
        hip_width      = _dist(l_hip, r_hip)
        body_height    = _dist(nose, (
            (l_ankle[0] + r_ankle[0]) / 2,
            (l_ankle[1] + r_ankle[1]) / 2
        ))

        shoulder_height_ratio = shoulder_width / (body_height + 1e-6)
        shoulder_hip_ratio    = shoulder_width / (hip_width + 1e-6)

        build = _classify_build(shoulder_height_ratio)

        # ── Dosha scores from build ───────────────────────────────────────────
        build_scores = {
            "thin":   {"vata": 0.70, "pitta": 0.20, "kapha": 0.10},
            "medium": {"vata": 0.20, "pitta": 0.60, "kapha": 0.20},
            "broad":  {"vata": 0.10, "pitta": 0.15, "kapha": 0.75},
        }[build]

        return {
            "build":                  build,
            "shoulder_width_px":      round(shoulder_width, 1),
            "hip_width_px":           round(hip_width, 1),
            "body_height_px":         round(body_height, 1),
            "shoulder_height_ratio":  round(shoulder_height_ratio, 3),
            "shoulder_hip_ratio":     round(shoulder_hip_ratio, 3),
            "dosha_scores": {
                "vata":  build_scores["vata"],
                "pitta": build_scores["pitta"],
                "kapha": build_scores["kapha"],
            },
        }