# api/modules/voice_analyzer.py

import numpy as np
import librosa
import soundfile as sf
import io
import tempfile
import os


# ─── Dosha-Voice Mapping (classical Ayurvedic texts + research) ──────────────
# Vata  : High pitch, fast speech, irregular rhythm, thin voice
# Pitta : Medium-sharp pitch, moderate-fast, clear articulate, strong
# Kapha : Low-deep pitch, slow speech, very steady rhythm, resonant
# ─────────────────────────────────────────────────────────────────────────────


def _classify_pitch(mean_hz: float) -> dict:
    """
    Fundamental frequency → dosha weight.
    Female voices are higher overall, so ranges are generous.
    """
    if mean_hz > 220:
        return {"vata": 0.65, "pitta": 0.25, "kapha": 0.10}   # high/thin
    elif mean_hz > 150:
        return {"vata": 0.25, "pitta": 0.55, "kapha": 0.20}   # medium-sharp
    else:
        return {"vata": 0.10, "pitta": 0.25, "kapha": 0.65}   # low/deep


def _classify_speech_speed(syllable_rate: float) -> dict:
    """
    Syllable rate per second → dosha weight.
    Fast talker = Vata, Moderate = Pitta, Slow = Kapha.
    """
    if syllable_rate > 4.5:
        return {"vata": 0.65, "pitta": 0.25, "kapha": 0.10}
    elif syllable_rate > 2.8:
        return {"vata": 0.20, "pitta": 0.60, "kapha": 0.20}
    else:
        return {"vata": 0.10, "pitta": 0.20, "kapha": 0.70}


def _classify_rhythm(ioi_std: float) -> dict:
    """
    Inter-onset interval standard deviation → rhythm regularity.
    High std = erratic (Vata), Low std = very steady (Kapha).
    """
    if ioi_std > 0.25:
        return {"vata": 0.65, "pitta": 0.20, "kapha": 0.15}
    elif ioi_std > 0.12:
        return {"vata": 0.20, "pitta": 0.60, "kapha": 0.20}
    else:
        return {"vata": 0.10, "pitta": 0.25, "kapha": 0.65}


def _classify_energy_variance(energy_std: float) -> dict:
    """
    RMS energy variance → voice steadiness.
    High variance = bursts of energy (Vata/Pitta), Low = steady (Kapha).
    """
    if energy_std > 0.05:
        return {"vata": 0.50, "pitta": 0.35, "kapha": 0.15}
    elif energy_std > 0.02:
        return {"vata": 0.20, "pitta": 0.55, "kapha": 0.25}
    else:
        return {"vata": 0.10, "pitta": 0.20, "kapha": 0.70}


def analyze_voice(audio_bytes: bytes = None, audio_path: str = None) -> dict:
    """
    Main function.
    Accepts either:
      - audio_bytes : raw bytes from FastAPI UploadFile
      - audio_path  : file path string (for testing)

    Returns structured voice analysis dict with dosha_scores.
    """
    try:
        # ── Load audio ────────────────────────────────────────────────────────
        if audio_path:
            y, sr = librosa.load(audio_path, sr=22050, duration=10.0)
        elif audio_bytes:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            y, sr = librosa.load(tmp_path, sr=22050, duration=10.0)
            os.unlink(tmp_path)
        else:
            return {"error": "No audio input provided", "dosha_scores": None}

        # Minimum duration check (need at least 1 second)
        if len(y) < sr:
            return {"error": "Audio too short — speak for at least 2 seconds",
                    "dosha_scores": None}

        # ── Feature 1: Pitch (fundamental frequency) ──────────────────────────
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        mean_pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 5 else 180.0

        # ── Feature 2: Speech speed (onset-based syllable rate) ───────────────
        onsets     = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        duration   = librosa.get_duration(y=y, sr=sr)
        syllable_rate = len(onsets) / (duration + 1e-6)

        # ── Feature 3: Rhythm regularity (inter-onset interval std) ──────────
        if len(onsets) > 3:
            ioi     = np.diff(onsets)
            ioi_std = float(np.std(ioi))
        else:
            ioi_std = 0.15  # default to medium if too few onsets

        # ── Feature 4: Energy variance ────────────────────────────────────────
        rms        = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        energy_std = float(np.std(rms))

        # ── Per-feature dosha scores ──────────────────────────────────────────
        pitch_scores   = _classify_pitch(mean_pitch)
        speed_scores   = _classify_speech_speed(syllable_rate)
        rhythm_scores  = _classify_rhythm(ioi_std)
        energy_scores  = _classify_energy_variance(energy_std)

        # ── Weighted fusion of 4 voice sub-signals ────────────────────────────
        weights = {
            "pitch":  0.35,
            "speed":  0.30,
            "rhythm": 0.20,
            "energy": 0.15,
        }

        combined = {}
        for dosha in ["vata", "pitta", "kapha"]:
            combined[dosha] = (
                pitch_scores[dosha]  * weights["pitch"]  +
                speed_scores[dosha]  * weights["speed"]  +
                rhythm_scores[dosha] * weights["rhythm"] +
                energy_scores[dosha] * weights["energy"]
            )

        total = sum(combined.values())
        combined = {k: round(v / total, 3) for k, v in combined.items()}

        return {
            "mean_pitch_hz":   round(mean_pitch, 1),
            "syllable_rate":   round(syllable_rate, 2),
            "rhythm_ioi_std":  round(ioi_std, 3),
            "energy_std":      round(energy_std, 4),
            "pitch_class":     "high" if mean_pitch > 220 else ("medium" if mean_pitch > 150 else "low"),
            "speed_class":     "fast" if syllable_rate > 4.5 else ("moderate" if syllable_rate > 2.8 else "slow"),
            "rhythm_class":    "erratic" if ioi_std > 0.25 else ("steady" if ioi_std > 0.12 else "very_steady"),
            "dosha_scores": {
                "vata":  combined["vata"],
                "pitta": combined["pitta"],
                "kapha": combined["kapha"],
            },
        }

    except Exception as e:
        return {"error": f"Audio analysis failed: {str(e)}", "dosha_scores": None}