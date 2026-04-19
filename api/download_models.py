import os
import urllib.request

MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH   = os.path.join(MODELS_DIR, 'tongue_model.pth')
METADATA_PATH = os.path.join(MODELS_DIR, 'tongue_metadata.json')

# Direct download from Google Drive
TONGUE_MODEL_URL = "https://drive.google.com/uc?id=1-TC4AriBkbZ5eMMv3vNrXjMo_AWlFMIi"

def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("[startup] Downloading tongue_model.pth from Google Drive...")
        try:
            urllib.request.urlretrieve(TONGUE_MODEL_URL, MODEL_PATH)
            size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"[startup] ✅ Downloaded tongue_model.pth ({size:.1f} MB)")
        except Exception as e:
            print(f"[startup] ❌ Model download failed: {e}")
    else:
        size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[startup] ✅ tongue_model.pth already present ({size:.1f} MB)")

if __name__ == "__main__":
    ensure_models()