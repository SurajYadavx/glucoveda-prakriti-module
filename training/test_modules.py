# training/test_modules.py
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from api.modules.tongue_classifier import analyze_tongue
from api.modules.skin_analyzer import analyze_skin

import cv2, time, tempfile
from api.modules.face_analyzer  import analyze_face
from api.modules.body_analyzer  import analyze_body
from api.modules.voice_analyzer import analyze_voice
from api.modules.fusion         import fuse_modules

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  sounddevice not installed — voice skipped")

MODES        = ["FACE", "BODY", "VOICE", "TONGUE", "SKIN"] if AUDIO_AVAILABLE else ["FACE", "BODY", "TONGUE"]
TIMERS       = {"FACE": 5, "BODY": 10, "VOICE": 5, "TONGUE": 5, "SKIN": 5}
INSTRUCTIONS = {
    "FACE":   "Look straight at camera — good lighting on face",
    "BODY":   "Step BACK — full body visible head to ankle",
    "VOICE":  "Say your full name clearly and slowly",
    "TONGUE": "Open mouth wide — stick tongue out fully",
    "SKIN": "Face the camera — normal expression, good lighting",
}

def overlay_text(frame, mode, seconds_left, done=False):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
    if done:
        cv2.putText(frame, "DONE — Check terminal for results | Q = quit",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"[{mode}] {INSTRUCTIONS[mode]}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 255), 2)
        cv2.putText(frame, f"Capturing in {seconds_left}s  |  Q = quit",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    return frame

def record_audio_clip(duration=4, sr=22050):
    print(f"  🎤 Recording {duration}s...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, recording, sr)
    return tmp.name

def run_test():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera"); return

    mode_idx     = 0
    capture_done = False
    all_done     = False
    capture_time = time.time() + TIMERS[MODES[0]]
    all_results  = {"face": None, "body": None, "voice": None,
                    "tongue": None, "skin": None}

    print("\n=== GlucoVeda — Full Module Test + Fusion ===\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        if all_done:
            display = overlay_text(frame.copy(), "", 0, done=True)
        else:
            mode         = MODES[mode_idx]
            seconds_left = max(0, int(capture_time - time.time()))
            display      = overlay_text(frame.copy(), mode, seconds_left)

            if time.time() >= capture_time and not capture_done:
                capture_done = True
                print(f"\n{'='*45}\n  {mode} RESULT\n{'='*45}")

                if mode == "FACE":
                    result = analyze_face(frame)
                    all_results["face"] = result
                elif mode == "BODY":
                    result = analyze_body(frame)
                    all_results["body"] = result
                elif mode == "VOICE":
                    audio_path = record_audio_clip(duration=4)
                    result = analyze_voice(audio_path=audio_path)
                    all_results["voice"] = result
                elif mode == "TONGUE":
                    result = analyze_tongue(image_bgr=frame)
                    all_results["tongue"] = result
                elif mode == "SKIN":
                    result = analyze_skin(image_bgr=frame)
                    all_results["skin"] = result

                if result.get("dosha_scores"):
                    print(f"  ✅ DOSHA SCORES: {result['dosha_scores']}")
                else:
                    print(f"  ❌ {result.get('error')}")

                mode_idx += 1
                if mode_idx >= len(MODES):
                    all_done = True

                    # ── FUSION ────────────────────────────────────────────
                    print(f"\n{'='*45}")
                    print(f"  FUSION RESULT  (video_partial_scores)")
                    print(f"{'='*45}")
                    fusion = fuse_modules(all_results)

                    vp = fusion["video_partial_scores"]
                    print(f"\n  Vata  : {vp['vata_visual']}%")
                    print(f"  Pitta : {vp['pitta_visual']}%")
                    print(f"  Kapha : {vp['kapha_visual']}%")
                    print(f"\n  Dominant      : {fusion['dominant_from_video']}")
                    print(f"  Confidence    : {fusion['confidence']}%")
                    print(f"  Modules used  : {fusion['modules_used']}")
                    print(f"  Modules missing: {fusion['modules_missing']}")
                    print(f"\n  Per-module breakdown:")
                    for mod, breakdown in fusion["module_breakdown"].items():
                        print(f"    {mod:8s} → V:{breakdown['vata']:.2f} "
                              f"P:{breakdown['pitta']:.2f} "
                              f"K:{breakdown['kapha']:.2f} "
                              f"(weight {breakdown['weight']:.2f})")
                    print(f"\n  → JSON for website: {fusion['video_partial_scores']}")
                    print(f"\n✅ Done. Press Q to quit.")
                else:
                    capture_done = False
                    next_mode    = MODES[mode_idx]
                    capture_time = time.time() + TIMERS[next_mode]
                    print(f"\n⏳ {next_mode} in {TIMERS[next_mode]}s → {INSTRUCTIONS[next_mode]}")

        cv2.imshow("GlucoVeda Prakriti Module", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()