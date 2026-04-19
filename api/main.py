# api/main.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import tempfile
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from api.modules.face_analyzer    import analyze_face
from api.modules.body_analyzer    import analyze_body
from api.modules.voice_analyzer   import analyze_voice
from api.modules.tongue_classifier import analyze_tongue
from api.modules.skin_analyzer    import analyze_skin
from api.modules.fusion           import fuse_modules
from api.download_models import ensure_models
ensure_models()
from api.download_models import ensure_models
ensure_models()

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "GlucoVeda Prakriti API",
    description = "Video-based Prakriti analysis — Vata/Pitta/Kapha dosha scoring",
    version     = "2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict to your website domain in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Patient photo storage ──────────────────────────────────────────────────────
PHOTOS_DIR = os.path.join(os.path.dirname(__file__), '..', 'patient_data', 'photos')
os.makedirs(PHOTOS_DIR, exist_ok=True)
app.mount("/patient-photos", StaticFiles(directory=PHOTOS_DIR), name="photos")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes → OpenCV BGR array."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def _save_patient_photo(patient_id: str, module: str,
                        image_bytes: bytes, ext: str = "jpg") -> str:
    """Save photo to patient_data/photos/{patient_id}/{session_date}/"""
    session_date = datetime.now().strftime("%Y-%m-%d")
    save_dir     = os.path.join(PHOTOS_DIR, patient_id, session_date)
    os.makedirs(save_dir, exist_ok=True)

    filename  = f"{patient_id}_{module}_{datetime.now().strftime('%H%M%S')}.{ext}"
    save_path = os.path.join(save_dir, filename)

    with open(save_path, 'wb') as f:
        f.write(image_bytes)

    return save_path


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "GlucoVeda Prakriti API",
        "version": "2.0.0",
        "status":  "running",
        "endpoints": {
            "analyze": "POST /analyze-prakriti",
            "photos":  "GET  /patient-photos/{patient_id}",
            "health":  "GET  /health"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/analyze-prakriti")
async def analyze_prakriti(
    face_image:   UploadFile = File(None),
    body_image:   UploadFile = File(None),
    tongue_image: UploadFile = File(None),
    audio_clip:   UploadFile = File(None),
    patient_id:   str        = Form(default="anonymous"),
    save_photos:  bool       = Form(default=True),
):
    """
    Main endpoint. Website sends images + audio, gets dosha JSON back.

    All fields are optional — send whatever you have.
    Fusion layer handles missing modules automatically.

    Returns:
        video_partial_scores: { vata_visual, pitta_visual, kapha_visual }
        dominant_from_video:  "Pitta"
        confidence:           74.3
        modules_used:         ["face", "tongue", "voice"]
        modules_missing:      ["body", "skin"]
    """
    results = {
        "face":   None,
        "body":   None,
        "voice":  None,
        "tongue": None,
        "skin":   None,
    }
    errors  = {}

    # ── Face ──────────────────────────────────────────────────────────────────
    if face_image:
        try:
            face_bytes = await face_image.read()
            face_bgr   = _bytes_to_bgr(face_bytes)

            results["face"] = analyze_face(face_bgr)
            results["skin"] = analyze_skin(face_bgr)  # skin uses same frame

            if save_photos and patient_id != "anonymous":
                _save_patient_photo(patient_id, "face", face_bytes)

        except Exception as e:
            errors["face"] = str(e)

    # ── Body ──────────────────────────────────────────────────────────────────
    if body_image:
        try:
            body_bytes      = await body_image.read()
            body_bgr        = _bytes_to_bgr(body_bytes)
            results["body"] = analyze_body(body_bgr)

            if save_photos and patient_id != "anonymous":
                _save_patient_photo(patient_id, "body", body_bytes)

        except Exception as e:
            errors["body"] = str(e)

    # ── Tongue ────────────────────────────────────────────────────────────────
    if tongue_image:
        try:
            tongue_bytes      = await tongue_image.read()
            tongue_bgr        = _bytes_to_bgr(tongue_bytes)
            results["tongue"] = analyze_tongue(image_bgr=tongue_bgr)

            if save_photos and patient_id != "anonymous":
                _save_patient_photo(patient_id, "tongue", tongue_bytes)

        except Exception as e:
            errors["tongue"] = str(e)

    # ── Voice ─────────────────────────────────────────────────────────────────
    if audio_clip:
        try:
            audio_bytes = await audio_clip.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            results["voice"] = analyze_voice(audio_path=tmp_path)
            os.unlink(tmp_path)

        except Exception as e:
            errors["voice"] = str(e)

    # ── Check at least one module has data ────────────────────────────────────
    if all(v is None for v in results.values()):
        raise HTTPException(
            status_code = 422,
            detail      = "No valid input provided. Send at least face_image."
        )

    # ── Fuse all modules ──────────────────────────────────────────────────────
    fusion_result = fuse_modules(results)

    # ── Add session metadata ──────────────────────────────────────────────────
    fusion_result["patient_id"]  = patient_id
    fusion_result["session_id"]  = str(uuid.uuid4())[:8]
    fusion_result["timestamp"]   = datetime.now().isoformat()
    if errors:
        fusion_result["module_errors"] = errors

    return JSONResponse(content=fusion_result)


@app.get("/patient-photos/{patient_id}")
def get_patient_photos(patient_id: str):
    """Returns list of all session photos for a patient."""
    patient_dir = os.path.join(PHOTOS_DIR, patient_id)
    if not os.path.exists(patient_dir):
        raise HTTPException(status_code=404, detail=f"No photos found for {patient_id}")

    sessions = {}
    for session_date in os.listdir(patient_dir):
        session_path = os.path.join(patient_dir, session_date)
        if os.path.isdir(session_path):
            sessions[session_date] = [
                f"/patient-photos/{patient_id}/{session_date}/{f}"
                for f in os.listdir(session_path)
            ]

    return {"patient_id": patient_id, "sessions": sessions}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)