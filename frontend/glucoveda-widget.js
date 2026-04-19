// glucoveda-widget.js
// GlucoVeda Prakriti — Drop-in Modal Widget
// Usage: <script src="glucoveda-widget.js"></script>
//        <button onclick="GlucoVeda.open()">Start Live Analysis</button>

(function () {
  "use strict";

  const API_BASE = "http://localhost:8000"; // ← change to live URL before deploy

  const STEPS = [
    { id:"face",   label:"Face Scan",   instruction:"Look straight at camera — good lighting on your face.", icon:"😐", duration:4000, type:"image" },
    { id:"body",   label:"Body Scan",   instruction:"Step back so your full body is visible — head to ankles.", icon:"🧍", duration:6000, type:"image" },
    { id:"tongue", label:"Tongue Scan", instruction:"Open your mouth wide and stick your tongue out fully.", icon:"👅", duration:4000, type:"image" },
    { id:"voice",  label:"Voice Sample",instruction:"Say your full name clearly and slowly.", icon:"🎤", duration:5000, type:"audio" },
  ];

  let stream = null, mediaRecorder = null, audioChunks = [];
  let captures = {}, currentStep = 0;
  let _patientId = "anonymous", _onComplete = null;

  // ── Inject CSS once ────────────────────────────────────────────────────────
  function injectStyles() {
    if (document.getElementById("gv-modal-styles")) return;
    const s = document.createElement("style");
    s.id = "gv-modal-styles";
    s.textContent = `
      #gv-modal-backdrop {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.75);
        z-index: 99998;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 16px;
        animation: gv-fade-in 0.2s ease;
      }
      @keyframes gv-fade-in {
        from { opacity: 0; }
        to   { opacity: 1; }
      }
      #gv-modal-box {
        position: relative;
        width: 100%;
        max-width: 460px;
        background: #0f172a;
        border-radius: 20px;
        overflow: hidden;
        color: #f8fafc;
        box-shadow: 0 32px 80px rgba(0,0,0,0.6);
        font-family: 'Segoe UI', sans-serif;
        animation: gv-slide-up 0.3s cubic-bezier(0.34,1.56,0.64,1);
      }
      @keyframes gv-slide-up {
        from { transform: translateY(40px); opacity: 0; }
        to   { transform: translateY(0);    opacity: 1; }
      }
      #gv-close-btn {
        position: absolute;
        top: 12px; right: 14px;
        z-index: 10;
        background: rgba(255,255,255,0.1);
        border: none;
        color: #fff;
        width: 30px; height: 30px;
        border-radius: 50%;
        font-size: 16px;
        cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        transition: background 0.2s;
      }
      #gv-close-btn:hover { background: rgba(255,255,255,0.2); }
      #gv-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 48px 16px 20px;
        background: #1e293b;
        border-bottom: 1px solid #334155;
      }
      #gv-logo { font-size: 15px; font-weight: 700; color: #4ade80; }
      #gv-step-counter { font-size: 12px; color: #94a3b8; }
      #gv-progress-bar { height: 3px; background: #0f172a; }
      #gv-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #4ade80, #22d3ee);
        width: 0%;
        transition: width 0.5s ease;
      }
      #gv-camera-wrap {
        position: relative;
        width: 100%;
        aspect-ratio: 4/3;
        background: #000;
      }
      #gv-video {
        width: 100%; height: 100%;
        object-fit: cover;
        transform: scaleX(-1);
      }
      #gv-overlay {
        position: absolute; inset: 0;
        display: flex; flex-direction: column;
        align-items: center; justify-content: flex-end;
        padding-bottom: 20px;
        background: linear-gradient(to top, rgba(0,0,0,0.72) 0%, transparent 55%);
        pointer-events: none;
      }
      #gv-step-icon { font-size: 32px; margin-bottom: 6px; }
      #gv-instruction {
        font-size: 13px; text-align: center;
        padding: 0 20px; color: #e2e8f0;
        line-height: 1.5; margin-bottom: 10px;
      }
      #gv-timer-ring {
        position: absolute; top: 14px; right: 14px;
        width: 52px; height: 52px;
      }
      #gv-timer-ring svg { width: 100%; height: 100%; }
      #gv-timer-num {
        position: absolute; inset: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; font-weight: 700; color: #fff;
      }
      #gv-flash {
        position: absolute; inset: 0;
        background: #fff; opacity: 0;
        pointer-events: none; transition: opacity 0.1s;
      }
      #gv-flash.flash { opacity: 0.75; }
      #gv-status-bar {
        padding: 9px 18px;
        background: #1e293b;
        font-size: 12px; color: #94a3b8;
        min-height: 36px;
      }
      #gv-controls { padding: 14px 18px; }
      #gv-start-btn {
        width: 100%; padding: 13px;
        background: linear-gradient(135deg, #4ade80, #22d3ee);
        color: #0f172a; font-size: 15px; font-weight: 700;
        border: none; border-radius: 10px;
        cursor: pointer; transition: opacity 0.2s;
      }
      #gv-start-btn:hover { opacity: 0.88; }
      #gv-start-btn:disabled { opacity: 0.35; cursor: not-allowed; }
      #gv-results {
        display: none;
        padding: 20px 18px 24px;
      }
      #gv-result-title {
        font-size: 13px; color: #4ade80;
        font-weight: 700; margin-bottom: 16px;
        text-transform: uppercase; letter-spacing: 0.5px;
      }
      .gv-dosha-row { margin-bottom: 12px; }
      .gv-dosha-label {
        display: flex; justify-content: space-between;
        font-size: 13px; margin-bottom: 4px; color: #cbd5e1;
      }
      .gv-dosha-track {
        height: 8px; background: #1e293b;
        border-radius: 4px; overflow: hidden;
      }
      .gv-dosha-fill {
        height: 100%; border-radius: 4px;
        transition: width 1.1s cubic-bezier(0.34,1.56,0.64,1);
      }
      .gv-fill-vata  { background: linear-gradient(90deg,#a78bfa,#7c3aed); }
      .gv-fill-pitta { background: linear-gradient(90deg,#fb923c,#dc2626); }
      .gv-fill-kapha { background: linear-gradient(90deg,#4ade80,#0891b2); }
      #gv-dominant-label {
        margin-top: 14px; font-size: 20px;
        font-weight: 800; color: #f8fafc;
      }
      #gv-confidence {
        font-size: 12px; color: #64748b;
        margin-top: 3px; margin-bottom: 16px;
      }
      #gv-retry-btn {
        width: 100%; padding: 11px;
        background: #1e293b;
        color: #94a3b8; font-size: 13px;
        border: 1px solid #334155; border-radius: 8px;
        cursor: pointer; transition: background 0.2s;
      }
      #gv-retry-btn:hover { background: #334155; color: #f8fafc; }
    `;
    document.head.appendChild(s);
  }

  // ── Build modal HTML ───────────────────────────────────────────────────────
  function buildModal() {
    const backdrop = document.createElement("div");
    backdrop.id = "gv-modal-backdrop";
    backdrop.innerHTML = `
      <div id="gv-modal-box">
        <button id="gv-close-btn" aria-label="Close">✕</button>
        <div id="gv-header">
          <div id="gv-logo">🌿 GlucoVeda Prakriti</div>
          <div id="gv-step-counter">Step <span id="gv-step-num">1</span> of ${STEPS.length}</div>
        </div>
        <div id="gv-progress-bar"><div id="gv-progress-fill"></div></div>
        <div id="gv-camera-wrap">
          <video id="gv-video" autoplay playsinline muted></video>
          <canvas id="gv-canvas" style="display:none"></canvas>
          <div id="gv-overlay">
            <div id="gv-step-icon"></div>
            <div id="gv-instruction"></div>
            <div id="gv-timer-ring">
              <svg viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#ffffff22" stroke-width="6"/>
                <circle id="gv-timer-arc" cx="50" cy="50" r="45" fill="none"
                        stroke="#4ade80" stroke-width="6"
                        stroke-dasharray="283" stroke-dashoffset="283"
                        stroke-linecap="round" transform="rotate(-90 50 50)"/>
              </svg>
              <div id="gv-timer-num">-</div>
            </div>
          </div>
          <div id="gv-flash"></div>
        </div>
        <div id="gv-status-bar"><span id="gv-status-text">Ready to start</span></div>
        <div id="gv-controls"><button id="gv-start-btn">▶ Start Analysis</button></div>
        <div id="gv-results">
          <div id="gv-result-title">Your Prakriti Result</div>
          <div id="gv-dosha-bars"></div>
          <div id="gv-dominant-label"></div>
          <div id="gv-confidence"></div>
          <button id="gv-retry-btn">↺ Retake Analysis</button>
        </div>
      </div>`;
    return backdrop;
  }

  // ── Open modal ─────────────────────────────────────────────────────────────
  function open(options = {}) {
    _patientId  = options.patientId  || "anonymous";
    _onComplete = options.onComplete || null;

    injectStyles();

    // Remove old modal if exists
    const old = document.getElementById("gv-modal-backdrop");
    if (old) old.remove();

    const modal = buildModal();
    document.body.appendChild(modal);

    // Bind buttons
    document.getElementById("gv-start-btn").addEventListener("click", startCapture);
    document.getElementById("gv-close-btn").addEventListener("click", closeModal);
    document.getElementById("gv-retry-btn").addEventListener("click", resetModal);

    // Close on backdrop click
    modal.addEventListener("click", function(e) {
      if (e.target === modal) closeModal();
    });

    // Prevent body scroll
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    stopStream();
    const modal = document.getElementById("gv-modal-backdrop");
    if (modal) modal.remove();
    document.body.style.overflow = "";
  }

  function resetModal() {
    stopStream();
    captures = {}; currentStep = 0;
    document.getElementById("gv-results").style.display  = "none";
    document.getElementById("gv-controls").style.display = "block";
    document.getElementById("gv-start-btn").disabled     = false;
    document.getElementById("gv-progress-fill").style.width = "0%";
    document.getElementById("gv-step-num").textContent   = "1";
    document.getElementById("gv-step-icon").textContent  = "";
    document.getElementById("gv-instruction").textContent = "";
    document.getElementById("gv-timer-num").textContent  = "-";
    setStatus("Ready to start");
  }

  // ── Camera ─────────────────────────────────────────────────────────────────
  function stopStream() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  }

  function captureFrame() {
    const video  = document.getElementById("gv-video");
    const canvas = document.getElementById("gv-canvas");
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const flash = document.getElementById("gv-flash");
    flash.classList.add("flash");
    setTimeout(() => flash.classList.remove("flash"), 200);
    return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.92));
  }

  async function recordAudio(durationMs) {
    audioChunks = [];
    const audioStream = new MediaStream(stream.getAudioTracks());
    mediaRecorder = new MediaRecorder(audioStream);
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    return new Promise(resolve => {
      mediaRecorder.onstop = () => resolve(new Blob(audioChunks, { type: "audio/wav" }));
      mediaRecorder.start();
      setTimeout(() => mediaRecorder.stop(), durationMs);
    });
  }

  // ── Sequence ───────────────────────────────────────────────────────────────
  async function startCapture() {
    document.getElementById("gv-start-btn").disabled = true;
    setStatus("Starting camera...");
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: true });
      document.getElementById("gv-video").srcObject = stream;
      await document.getElementById("gv-video").play();
    } catch(e) {
      setStatus("❌ Camera denied — please allow camera access and try again");
      document.getElementById("gv-start-btn").disabled = false;
      return;
    }
    captures = {}; currentStep = 0;
    await runStep();
  }

  async function runStep() {
    if (currentStep >= STEPS.length) { await submitToAPI(); return; }
    const step = STEPS[currentStep];
    document.getElementById("gv-step-num").textContent    = currentStep + 1;
    document.getElementById("gv-step-icon").textContent   = step.icon;
    document.getElementById("gv-instruction").textContent = step.instruction;
    document.getElementById("gv-progress-fill").style.width = (currentStep / STEPS.length * 100) + "%";
    setStatus(`${step.label} — get ready...`);
    await countdown(step.duration);
    if (step.type === "image") {
      captures[step.id] = await captureFrame();
      setStatus(`✅ ${step.label} captured`);
    } else {
      setStatus("🎤 Recording voice...");
      captures[step.id] = await recordAudio(step.duration - 500);
      setStatus("✅ Voice recorded");
    }
    currentStep++;
    await delay(400);
    await runStep();
  }

  function countdown(durationMs) {
    return new Promise(resolve => {
      const arc   = document.getElementById("gv-timer-arc");
      const numEl = document.getElementById("gv-timer-num");
      const start = performance.now();
      function tick(now) {
        const elapsed   = now - start;
        const remaining = Math.max(0, durationMs - elapsed);
        arc.setAttribute("stroke-dashoffset", 283 * (1 - elapsed / durationMs));
        numEl.textContent = Math.ceil(remaining / 1000);
        remaining > 0 ? requestAnimationFrame(tick) : (numEl.textContent = "✓", resolve());
      }
      requestAnimationFrame(tick);
    });
  }

  // ── API Submit ─────────────────────────────────────────────────────────────
  async function submitToAPI() {
    document.getElementById("gv-progress-fill").style.width = "90%";
    setStatus("🔄 Analysing your Prakriti...");
    const fd = new FormData();
    fd.append("patient_id",  _patientId);
    fd.append("save_photos", "true");
    if (captures.face)   fd.append("face_image",   captures.face,   "face.jpg");
    if (captures.body)   fd.append("body_image",   captures.body,   "body.jpg");
    if (captures.tongue) fd.append("tongue_image", captures.tongue, "tongue.jpg");
    if (captures.voice)  fd.append("audio_clip",   captures.voice,  "voice.wav");
    try {
      const res  = await fetch(`${API_BASE}/analyze-prakriti`, { method: "POST", body: fd });
      const data = await res.json();
      document.getElementById("gv-progress-fill").style.width = "100%";
      setStatus("✅ Analysis complete");
      showResults(data);
      if (typeof _onComplete === "function") _onComplete(data);
    } catch(e) {
      setStatus("❌ Could not reach analysis server. Please try again.");
      document.getElementById("gv-start-btn").disabled = false;
    }
  }

  // ── Show Results ───────────────────────────────────────────────────────────
  function showResults(data) {
    document.getElementById("gv-controls").style.display = "none";
    document.getElementById("gv-results").style.display  = "block";
    const vp    = data.video_partial_scores || {};
    const vata  = vp.vata_visual  || 0;
    const pitta = vp.pitta_visual || 0;
    const kapha = vp.kapha_visual || 0;
    document.getElementById("gv-dosha-bars").innerHTML = `
      <div class="gv-dosha-row">
        <div class="gv-dosha-label"><span>Vata</span><span>${vata}%</span></div>
        <div class="gv-dosha-track"><div class="gv-dosha-fill gv-fill-vata" style="width:0%" data-target="${vata}%"></div></div>
      </div>
      <div class="gv-dosha-row">
        <div class="gv-dosha-label"><span>Pitta</span><span>${pitta}%</span></div>
        <div class="gv-dosha-track"><div class="gv-dosha-fill gv-fill-pitta" style="width:0%" data-target="${pitta}%"></div></div>
      </div>
      <div class="gv-dosha-row">
        <div class="gv-dosha-label"><span>Kapha</span><span>${kapha}%</span></div>
        <div class="gv-dosha-track"><div class="gv-dosha-fill gv-fill-kapha" style="width:0%" data-target="${kapha}%"></div></div>
      </div>`;
    document.getElementById("gv-dominant-label").textContent =
      `Prakriti: ${data.dominant_from_video || "—"}`;
    document.getElementById("gv-confidence").textContent =
      `Confidence: ${data.confidence || 0}%  •  Modules used: ${(data.modules_used || []).join(", ")}`;
    setTimeout(() => {
      document.querySelectorAll(".gv-dosha-fill")
              .forEach(el => el.style.width = el.dataset.target);
    }, 100);
  }

  // ── Helpers ────────────────────────────────────────────────────────────────
  function setStatus(msg) {
    const el = document.getElementById("gv-status-text");
    if (el) el.textContent = msg;
  }
  function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ── Public API ─────────────────────────────────────────────────────────────
  window.GlucoVeda = { open, close: closeModal };

})();