# GlucoVeda Prakriti Module

This module analyses a person's Ayurvedic body type (Prakriti) using their camera and microphone.
It looks at 5 things — face structure, body shape, tongue color, skin tone, and voice pitch —
and outputs a percentage score for Vata, Pitta, and Kapha dosha.

---

## What This Does

1. User opens the website and clicks "Start Analysis"
2. The camera captures face, body, and tongue photos automatically
3. The microphone records 4 seconds of voice
4. All data is sent to the Python API running on port 8000
5. The API runs 5 analysis modules and combines the results
6. Website receives JSON with Vata/Pitta/Kapha percentages

---

## Folder Structure

glucoveda-prakriti-module/
│
├── api/ ← Python backend
│ ├── main.py ← Start this to run the API
│ └── modules/
│ ├── face_analyzer.py ← Reads face shape using 468 landmarks
│ ├── body_analyzer.py ← Reads body proportions using pose detection
│ ├── voice_analyzer.py ← Reads pitch and tone from voice recording
│ ├── tongue_classifier.py ← AI model trained on tongue images
│ ├── skin_analyzer.py ← Reads skin tone from 5 face zones
│ └── fusion.py ← Combines all 5 scores into final result
│
├── models/
│ ├── tongue_model.pth ← Trained AI model file (ResNet50)
│ └── tongue_metadata.json ← Model settings and class names
│
├── frontend/
│ ├── capture_engine.js ← The widget your teammate adds to the website
│ └── test.html ← Test page to try everything locally
│
├── training/
│ ├── train_tongue.ipynb ← Google Colab notebook used to train the model
│ └── test_modules.py ← Run this to test all 5 modules at once
│
├── patient_data/
│ └── photos/ ← Captured photos get saved here automatically
│
├── requirements.txt ← All Python packages needed
└── README.md ← This file

---

## How to Run

### Step 1 — Install packages

```bash
pip install -r requirements.txt
```

### Step 2 — Start the API

Open a terminal and run:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

You will see: Uvicorn running on http://0.0.0.0:8000

Keep this terminal open.

### Step 3 — Test it in browser

Open a second terminal and run:

```bash
cd frontend
python -m http.server 3000
```

Then open your browser and go to: http://localhost:3000/test.html

Click Start Analysis, go through all 4 steps, and you will see the result JSON appear below the widget.

### Step 4 — View API docs

Go to:http://localhost:8000/docs

This shows all API endpoints with a live test interface.

---

## API — How to Use

### Endpoint

POST http://localhost:8000/analyze-prakriti

### What to Send

| Field          | Type       | Description                              |
| -------------- | ---------- | ---------------------------------------- |
| `face_image`   | image file | Photo of face (jpg or png)               |
| `body_image`   | image file | Full body photo (optional)               |
| `tongue_image` | image file | Photo of tongue (optional)               |
| `audio_clip`   | audio file | Voice recording in wav format (optional) |
| `patient_id`   | text       | Any ID to identify the patient           |

Send at least `face_image`. All others are optional — the system works with whatever you give it.

### What You Get Back

```json
{
  "video_partial_scores": {
    "vata_visual": 24.3,
    "pitta_visual": 48.4,
    "kapha_visual": 27.3
  },
  "dominant_from_video": "Pitta",
  "confidence": 67.6,
  "modules_used": ["face", "skin", "voice", "tongue"],
  "modules_missing": ["body"],
  "patient_id": "PT001",
  "session_id": "7bf809c9",
  "timestamp": "2026-04-20T01:00:00"
}
```

---

## Website Integration

### Your teammate needs to do only 3 things:

**1. Copy `capture_engine.js` into the website folder**

**2. Add this div where the widget should appear:**

```html
<div id="glucoveda-capture"></div>
```

**3. Add this script at the bottom of the page:**

```html
<script src="capture_engine.js"></script>
<script>
  GlucoVedaPrakriti.mount("glucoveda-capture", {
    patientId: "PT001",
    onComplete: function (result) {
      console.log(result.video_partial_scores);
      // result.video_partial_scores.vata_visual  → number like 24.3
      // result.video_partial_scores.pitta_visual → number like 48.4
      // result.video_partial_scores.kapha_visual → number like 27.3
      // result.dominant_from_video               → "Vata" or "Pitta" or "Kapha"
    },
  });
</script>
```

That's it. The widget handles camera, recording, sending to API, and showing the result automatically.

---

## The 5 Analysis Modules

| Module | What it looks at                                    | Weight in final score |
| ------ | --------------------------------------------------- | --------------------- |
| Face   | Eye spacing, jaw width, forehead height, face shape | 35–50%                |
| Skin   | Skin tone and color from 5 zones on the face        | 10–20%                |
| Voice  | Pitch frequency, tone richness, speech rhythm       | 15–25%                |
| Tongue | Color and texture using trained AI model            | 15–25%                |
| Body   | Height-to-width ratio, shoulder and hip width       | 10–20%                |

If a module fails or is missing, its weight is redistributed to the others automatically.

---

## The Tongue AI Model

- Built with: ResNet50 (a standard image classification network)
- Trained on: 567 tongue photos — 285 Vata class, 282 Pitta class
- Trained on: Google Colab with T4 GPU, 25 training rounds
- Accuracy: 100% on validation set
- What it detects: Pale/dry tongue = Vata, Red/moist tongue = Pitta, Kapha is calculated from the remaining probability

---

## Before Going Live on the Website

**Two things to change:**

1. In `frontend/capture_engine.js`, line 4:

```javascript
// Change this:
const API_BASE = "http://localhost:8000";

// To your live server URL:
const API_BASE = "https://your-api-server.com";
```

2. In `api/main.py`, around line 20:

```python
# Change this:
allow_origins = ["*"]

# To your website URL only:
allow_origins = ["https://your-website.com"]
```

---

## Common Errors

| Error                       | Cause                           | Fix                                                      |
| --------------------------- | ------------------------------- | -------------------------------------------------------- |
| Camera denied               | Browser blocked camera          | Click Allow when browser asks, or check site permissions |
| API error on port 8000      | API server not running          | Run Terminal 1 command again                             |
| Body not detected           | User not far enough             | Step back until full body is in frame                    |
| Tongue score low confidence | Poor lighting or partial tongue | Better lighting, stick tongue out fully                  |
| `ModuleNotFoundError`       | Package not installed           | Run `pip install -r requirements.txt`                    |
