# api/modules/fusion.py

# ─── Module Weights ────────────────────────────────────────────────────────────
# Based on DoshaMitra feature importance + research validation scores
# Face + Body are highest because they're geometry-based (no training bias)
# Voice is lowest because mic quality varies per device
# Tongue + Skin placeholders (0.0) until models are trained on Colab
# Total always = 1.0
# ──────────────────────────────────────────────────────────────────────────────

MODULE_WEIGHTS = {
    "face":    0.30,
    "body":    0.30,
    "voice":   0.15,
    "tongue":  0.15,   # 0.0 until trained model ready
    "skin":    0.10,   # 0.0 until trained model ready
}

# ─── Dosha Label Map ──────────────────────────────────────────────────────────
DOSHA_LABELS = {
    (True,  False, False): "Vata",
    (False, True,  False): "Pitta",
    (False, False, True ): "Kapha",
    (True,  True,  False): "Vata-Pitta",
    (True,  False, True ): "Vata-Kapha",
    (False, True,  True ): "Pitta-Kapha",
}


def _normalize(scores: dict) -> dict:
    """Ensure vata+pitta+kapha = 1.0"""
    total = sum(scores.values())
    if total < 1e-6:
        return {"vata": 0.34, "pitta": 0.33, "kapha": 0.33}
    return {k: round(v / total, 4) for k, v in scores.items()}


def _get_dominant_label(vata: float, pitta: float, kapha: float) -> str:
    """
    Returns Prakriti label.
    Single dominant: score > 0.50
    Dual dominant : two scores both > 0.30 and within 0.15 of each other
    """
    scores = {"vata": vata, "pitta": pitta, "kapha": kapha}
    sorted_doshas = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top1_name,  top1_val  = sorted_doshas[0]
    top2_name,  top2_val  = sorted_doshas[1]

    # Single dominant
    if top1_val > 0.50:
        return top1_name.capitalize()

    # Dual dominant — both meaningful and close
    if top1_val > 0.30 and top2_val > 0.25 and (top1_val - top2_val) < 0.18:
        pair = tuple(sorted([top1_name, top2_name]))
        label_map = {
            ("pitta", "vata"):  "Vata-Pitta",
            ("kapha", "vata"):  "Vata-Kapha",
            ("kapha", "pitta"): "Pitta-Kapha",
        }
        return label_map.get(pair, f"{top1_name.capitalize()}-{top2_name.capitalize()}")

    return top1_name.capitalize()


def _confidence_score(module_results: dict) -> float:
    """
    Confidence = inverse of disagreement between modules.
    If all modules agree → high confidence.
    If modules split → low confidence.
    Measured as 1 - std_deviation of per-module dominant dosha votes.
    """
    vata_votes  = []
    pitta_votes = []
    kapha_votes = []

    for module_name, result in module_results.items():
        if result and result.get("dosha_scores"):
            ds = result["dosha_scores"]
            vata_votes.append(ds.get("vata",  0))
            pitta_votes.append(ds.get("pitta", 0))
            kapha_votes.append(ds.get("kapha", 0))

    if len(vata_votes) < 2:
        return 60.0  # not enough data for confidence calc

    import numpy as np
    avg_std = float(np.mean([
        np.std(vata_votes),
        np.std(pitta_votes),
        np.std(kapha_votes)
    ]))

    # std of 0 → 100% confidence, std of 0.3+ → ~40% confidence
    confidence = max(40.0, min(97.0, round((1.0 - avg_std * 2.5) * 100, 1)))
    return confidence


def fuse_modules(module_results: dict) -> dict:
    """
    Main fusion function.

    Args:
        module_results: dict with keys face, body, voice, tongue, skin
                        Each value is the output dict from that module.
                        Pass None for modules not yet available.

    Returns:
        Complete fusion result ready to send to GlucoVeda website API.

    Example input:
        {
            "face":   analyze_face(frame),
            "body":   analyze_body(frame),
            "voice":  analyze_voice(audio_path=path),
            "tongue": None,   # not trained yet
            "skin":   None,   # not trained yet
        }
    """
    available_modules = {
        name: result
        for name, result in module_results.items()
        if result is not None and result.get("dosha_scores") is not None
    }

    if not available_modules:
        return {"error": "No module results available", "dosha_scores": None}

    # ── Recalculate weights for available modules only (renormalize) ──────────
    active_weights = {
        name: MODULE_WEIGHTS[name]
        for name in available_modules
    }
    weight_total = sum(active_weights.values())
    norm_weights = {k: v / weight_total for k, v in active_weights.items()}

    # ── Weighted fusion ───────────────────────────────────────────────────────
    fused_vata  = 0.0
    fused_pitta = 0.0
    fused_kapha = 0.0

    module_breakdown = {}

    for name, result in available_modules.items():
        ds     = result["dosha_scores"]
        weight = norm_weights[name]

        fused_vata  += ds["vata"]  * weight
        fused_pitta += ds["pitta"] * weight
        fused_kapha += ds["kapha"] * weight

        module_breakdown[name] = {
            "vata":   ds["vata"],
            "pitta":  ds["pitta"],
            "kapha":  ds["kapha"],
            "weight": round(weight, 3),
        }

    # Normalize final
    final = _normalize({
        "vata":  fused_vata,
        "pitta": fused_pitta,
        "kapha": fused_kapha,
    })

    dominant_label = _get_dominant_label(
        final["vata"], final["pitta"], final["kapha"]
    )
    confidence = _confidence_score(available_modules)

    # ── Percentage form for website ───────────────────────────────────────────
    vata_pct  = round(final["vata"]  * 100, 1)
    pitta_pct = round(final["pitta"] * 100, 1)
    kapha_pct = round(final["kapha"] * 100, 1)

    return {
        # ── This is the ONE object the website receives ────────────────────
        "video_partial_scores": {
            "vata_visual":  vata_pct,
            "pitta_visual": pitta_pct,
            "kapha_visual": kapha_pct,
        },
        "dominant_from_video": dominant_label,
        "confidence":          confidence,
        "modules_used":        list(available_modules.keys()),
        "modules_missing":     [n for n in MODULE_WEIGHTS if n not in available_modules],

        # ── Detailed breakdown — for doctor view and debugging ─────────────
        "module_breakdown":    module_breakdown,

        # ── Raw scores — website fusion layer uses these ───────────────────
        "raw_scores": {
            "vata":  final["vata"],
            "pitta": final["pitta"],
            "kapha": final["kapha"],
        },
    }