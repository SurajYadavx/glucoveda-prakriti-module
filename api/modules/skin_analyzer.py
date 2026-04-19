# api/modules/skin_analyzer.py

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# ── Landmark indices for 5 skin sampling zones ────────────────────────────────
# Forehead, left cheek, right cheek, chin, nose bridge
SKIN_ZONES = {
    "forehead":    [10, 338, 297, 332, 284],
    "left_cheek":  [116, 117, 118, 119, 100],
    "right_cheek": [345, 346, 347, 348, 329],
    "chin":        [175, 199, 200, 175, 152],
    "nose":        [1, 2, 5, 4, 19],
}

# ── Dosha skin classification rules (Ayurvedic + HSV research) ────────────────
# Vata  : Dusky/dark, dry, low saturation — H:8-25, S:40-100, V:60-130
# Pitta : Warm reddish, sensitive — H:5-18, S:90-180, V:120-200
# Kapha : Pale/oily, cool tone — H:15-35, S:20-70, V:160-235

def _hsv_to_dosha(h, s, v) -> dict:
    """
    Multi-factor HSV → dosha scoring.
    Each factor votes independently, then weighted sum.
    """
    vata = pitta = kapha = 0.0

    # ── Hue factor ────────────────────────────────────────────────────────────
    if 8 <= h <= 20:
        pitta += 0.4   # warm reddish hue
        vata  += 0.2
    elif h < 8 or (20 < h <= 28):
        vata  += 0.35
    elif h > 28:
        kapha += 0.35  # cooler/yellowish hue

    # ── Saturation factor ─────────────────────────────────────────────────────
    if s > 110:
        pitta += 0.35  # high saturation = Pitta (redness, inflammation)
    elif s > 60:
        vata  += 0.20
        pitta += 0.15
    else:
        kapha += 0.30  # low saturation = Kapha (pale, oily)
        vata  += 0.10

    # ── Value (brightness) factor ─────────────────────────────────────────────
    if v < 110:
        vata  += 0.35  # dark skin = Vata
    elif v < 160:
        pitta += 0.20
        vata  += 0.10
    else:
        kapha += 0.35  # very bright/pale = Kapha

    # ── Sanity check — if H outside skin range entirely ───────────────────────
    if h > 45 or h < 0:
        return {"vata": 0.34, "pitta": 0.33, "kapha": 0.33}

    total = vata + pitta + kapha
    if total < 0.1:
        return {"vata": 0.34, "pitta": 0.33, "kapha": 0.33}

    return {
        "vata":  round(vata  / total, 4),
        "pitta": round(pitta / total, 4),
        "kapha": round(kapha / total, 4),
    }


def _sample_zone(image_bgr, landmarks, zone_lms, h, w):
    """Sample mean HSV from a polygon region defined by landmark indices."""
    pts = np.array([
        [int(landmarks[i].x * w), int(landmarks[i].y * h)]
        for i in zone_lms
    ], dtype=np.int32)

    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)

    if cv2.countNonZero(mask) < 50:
        return None

    hsv      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
    return mean_hsv


def analyze_skin(image_bgr: np.ndarray) -> dict:
    """
    Main function. Accepts BGR frame from webcam.
    Samples 5 facial skin zones → per-zone dosha → weighted average.
    """
    try:
        h, w = image_bgr.shape[:2]

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:

            rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                return {"error": "No face detected for skin analysis",
                        "dosha_scores": None}

            landmarks = result.multi_face_landmarks[0].landmark

            # ── Sample each zone ──────────────────────────────────────────────
            zone_results   = {}
            all_vata  = []
            all_pitta = []
            all_kapha = []
            all_hsv   = []

            # Zone weights — cheeks most reliable, forehead can have shine
            zone_weights = {
                "forehead":    0.15,
                "left_cheek":  0.30,
                "right_cheek": 0.30,
                "chin":        0.15,
                "nose":        0.10,
            }

            for zone_name, lm_indices in SKIN_ZONES.items():
                hsv_mean = _sample_zone(image_bgr, landmarks, lm_indices, h, w)
                if hsv_mean is None:
                    continue

                hz, sz, vz = hsv_mean
                scores     = _hsv_to_dosha(hz, sz, vz)
                weight     = zone_weights[zone_name]

                all_vata.append(scores["vata"]  * weight)
                all_pitta.append(scores["pitta"] * weight)
                all_kapha.append(scores["kapha"] * weight)
                all_hsv.append([round(hz, 1), round(sz, 1), round(vz, 1)])

                zone_results[zone_name] = {
                    "hsv":    [round(hz, 1), round(sz, 1), round(vz, 1)],
                    "scores": scores,
                }

            if not all_vata:
                return {"error": "Could not sample skin zones",
                        "dosha_scores": None}

            # ── Weighted fusion across zones ──────────────────────────────────
            fused_vata  = sum(all_vata)
            fused_pitta = sum(all_pitta)
            fused_kapha = sum(all_kapha)

            total = fused_vata + fused_pitta + fused_kapha
            final = {
                "vata":  round(fused_vata  / total, 4),
                "pitta": round(fused_pitta / total, 4),
                "kapha": round(fused_kapha / total, 4),
            }

            # Mean HSV across all zones
            mean_hsv_all = np.mean(all_hsv, axis=0).tolist()

            return {
                "zones_sampled":  len(zone_results),
                "mean_skin_hsv":  [round(x, 1) for x in mean_hsv_all],
                "zone_breakdown": zone_results,
                "dosha_scores":   final,
            }

    except Exception as e:
        return {"error": f"Skin analysis failed: {str(e)}", "dosha_scores": None}