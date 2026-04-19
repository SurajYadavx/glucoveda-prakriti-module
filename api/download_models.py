# api/download_models.py
# Auto-downloads model from Google Drive if not present locally

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ← Paste your Google Drive direct download link here
TONGUE_MODEL_URL = "https://drive.google.com/uc?id=1-TC4AriBkbZ5eMMv3vNrXjMo_AWlFMIi"

def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'tongue_model.pth')

    if not os.path.exists(model_path):
        print("[startup] tongue_model.pth not found — downloading from Drive...")
        try:
            urllib.request.urlretrieve(TONGUE_MODEL_URL, model_path)
            size = os.path.getsize(model_path) / (1024*1024)
            print(f"[startup] ✅ Downloaded tongue_model.pth ({size:.1f} MB)")
        except Exception as e:
            print(f"[startup] ❌ Failed to download model: {e}")
    else:
        print("[startup] ✅ tongue_model.pth already present")

if __name__ == "__main__":
    ensure_models()