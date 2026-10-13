/**
 * AI Vision Assistant — Interactive Cyber-Aura HUD Client Logic (Phase 5)
 * Features:
 * - Web Audio API initialization on explicit user gesture
 * - Synthesized futuristic earcons (wake, listening, thinking, answer ready)
 * - Spacebar hotkey with e.preventDefault() to eliminate scrolling
 * - Window focus / blur warning detector
 * - Real-time WebSocket telemetry & voice state synchronization
 * - Navigation between Live Walk Mode, Test Lab, and Model Evaluation
 * - Test Lab: sample picker, file drag/drop upload, media detection preview
 * - Test Lab: "Analyse movement" background runner with live progress bar and timeline
 * - Test Lab: query execution with measured latency pipeline trace
 * - Model Evaluation: dynamic benchmark reader from experiment_02_summary.json
 */

(function () {
  'use strict';

  // --- Audio Context & Sound Synthesizer ---
  let audioCtx = null;
  let audioUnlocked = false;

  function initAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        audioCtx = new AudioContextClass();
      }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    audioUnlocked = true;
  }

  // Pure Web Audio synthesized earcons (futuristic high-tech tones)
  function playEarcon(type) {
    if (!audioCtx || !audioUnlocked) return;
    try {
      const now = audioCtx.currentTime;

      if (type === 'wake') {
        // High rising two-tone chime (587Hz -> 880Hz, D5 -> A5)
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(587.33, now);
        osc.frequency.exponentialRampToValueAtTime(880.00, now + 0.15);
        gain.gain.setValueAtTime(0.18, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.35);
      } else if (type === 'listen') {
        // Crisp high-tech prompt tone (750Hz sine pulse)
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(750, now);
        gain.gain.setValueAtTime(0.2, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.2);
      } else if (type === 'thinking') {
        // Soft resonant thinking blip (440Hz warm pulse)
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(440, now);
        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.25);
      } else if (type === 'answer') {
        // Harmonious chord completion chime (659Hz + 987Hz, E5 + B5)
        [659.25, 987.77].forEach((freq) => {
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(freq, now);
          gain.gain.setValueAtTime(0.12, now);
          gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          osc.start(now);
          osc.stop(now + 0.45);
        });
      }
    } catch (e) {
      console.warn('Audio earcon error:', e);
    }
  }

  // --- DOM Elements ---
  const startOverlay = document.getElementById('startOverlay');
  const startBtn = document.getElementById('startBtn');
  const unfocusedWarning = document.getElementById('unfocusedWarning');

  const camFpsVal = document.getElementById('camFpsVal');
  const detFpsVal = document.getElementById('detFpsVal');
  const detLatVal = document.getElementById('detLatVal');

  const statusOrb = document.getElementById('statusOrb');
  const statusLabel = document.getElementById('statusLabel');

  const heardText = document.getElementById('heardText');
  const assistantResponseText = document.getElementById('assistantResponseText');
  const responseLatency = document.getElementById('responseLatency');
  const responseIntent = document.getElementById('responseIntent');
  const replayBtn = document.getElementById('replayBtn');

  const zoneLeft = document.getElementById('zoneLeft');
  const zoneCenter = document.getElementById('zoneCenter');
  const zoneRight = document.getElementById('zoneRight');
  const zoneLeftCount = document.getElementById('zoneLeftCount');
  const zoneCenterCount = document.getElementById('zoneCenterCount');
  const zoneRightCount = document.getElementById('zoneRightCount');

  const queryForm = document.getElementById('queryForm');
  const queryInput = document.getElementById('queryInput');
  const promptChips = document.querySelectorAll('.prompt-chip');

  // Read answers aloud switches & TTS notes
  const readAloudToggle = document.getElementById('readAloudToggle');
  const testLabReadAloudToggle = document.getElementById('testLabReadAloudToggle');
  const ttsNoteWalk = document.getElementById('ttsNoteWalk');
  const ttsNoteTestLab = document.getElementById('ttsNoteTestLab');
  const STORAGE_KEY_READ_ALOUD = 'ai_vision_read_aloud';

  function isReadAloudEnabled() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY_READ_ALOUD);
      return stored !== 'false';
    } catch (e) {
      return true;
    }
  }

  function setReadAloudEnabled(enabled) {
    try {
      localStorage.setItem(STORAGE_KEY_READ_ALOUD, enabled ? 'true' : 'false');
    } catch (e) {}
    if (readAloudToggle) readAloudToggle.checked = enabled;
    if (testLabReadAloudToggle) testLabReadAloudToggle.checked = enabled;
  }

  const initialReadAloud = isReadAloudEnabled();
  if (readAloudToggle) {
    readAloudToggle.checked = initialReadAloud;
    readAloudToggle.addEventListener('change', () => setReadAloudEnabled(readAloudToggle.checked));
  }
  if (testLabReadAloudToggle) {
    testLabReadAloudToggle.checked = initialReadAloud;
    testLabReadAloudToggle.addEventListener('change', () => setReadAloudEnabled(testLabReadAloudToggle.checked));
  }

  async function checkTtsAvailability() {
    try {
      const res = await fetch('/api/tts/status');
      const data = await res.json();
      const available = data && data.available;
      if (ttsNoteWalk) ttsNoteWalk.classList.toggle('hidden', available);
      if (ttsNoteTestLab) ttsNoteTestLab.classList.toggle('hidden', available);
    } catch (err) {
      if (ttsNoteWalk) ttsNoteWalk.classList.remove('hidden');
      if (ttsNoteTestLab) ttsNoteTestLab.classList.remove('hidden');
    }
  }
  checkTtsAvailability();

  // Mode Tabs & View Containers
  const tabWalk = document.getElementById('tabWalk');
  const tabTestLab = document.getElementById('tabTestLab');
  const tabEval = document.getElementById('tabEval');
  const viewWalkMode = document.getElementById('viewWalkMode');
  const viewTestLab = document.getElementById('viewTestLab');
  const viewEvaluation = document.getElementById('viewEvaluation');

  let currentMode = 'walk'; // 'walk' | 'test-lab' | 'eval'

  // --- Mode Tab Switching ---
  async function setMode(mode) {
    currentMode = mode;

    // Update Tab Classes
    [tabWalk, tabTestLab, tabEval].forEach((t) => {
      if (t) {
        const isActive = t.dataset.tab === mode;
        t.classList.toggle('active', isActive);
        t.setAttribute('aria-selected', isActive ? 'true' : 'false');
      }
    });

    // Update View Containers
    if (viewWalkMode) viewWalkMode.classList.toggle('hidden', mode !== 'walk');
    if (viewTestLab) viewTestLab.classList.toggle('hidden', mode !== 'test-lab');
    if (viewEvaluation) viewEvaluation.classList.toggle('hidden', mode !== 'eval');

    // Notify Backend
    const backendMode = mode === 'test-lab' ? 'test_lab' : 'walk';
    try {
      await fetch('/api/set_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: backendMode })
      });
    } catch (e) {
      console.warn('Mode change sync error:', e);
    }

    if (mode === 'eval') {
      loadEvaluationMetrics();
    } else if (mode === 'test-lab') {
      loadCurrentTestLabMedia();
    }
  }

  if (tabWalk) tabWalk.addEventListener('click', () => setMode('walk'));
  if (tabTestLab) tabTestLab.addEventListener('click', () => setMode('test-lab'));
  if (tabEval) tabEval.addEventListener('click', () => setMode('eval'));

  // --- Audio Unlock & Greeting ---
  function unlockAudioAndStart() {
    initAudioContext();
    if (startOverlay) {
      startOverlay.classList.add('fading');
      setTimeout(() => startOverlay.remove(), 400);
    }
    playEarcon('wake');

    // Spoken startup greeting
    if ('speechSynthesis' in window) {
      const greetingUtterance = new SpeechSynthesisUtterance(
        'AI Vision Assistant initialized. Walk mode active. Say Hey Vision or press Spacebar to ask a question.'
      );
      greetingUtterance.rate = 1.05;
      greetingUtterance.pitch = 1.0;
      window.speechSynthesis.speak(greetingUtterance);
    }
  }

  if (startBtn) {
    startBtn.addEventListener('click', unlockAudioAndStart);
  }

  window.addEventListener('keydown', function onFirstKey(e) {
    if (!audioUnlocked) {
      unlockAudioAndStart();
      window.removeEventListener('keydown', onFirstKey);
    }
  });

  // --- Window Focus Detection ---
  window.addEventListener('blur', () => {
    if (unfocusedWarning) unfocusedWarning.classList.remove('hidden');
  });

  window.addEventListener('focus', () => {
    if (unfocusedWarning) unfocusedWarning.classList.add('hidden');
  });

  // --- Spacebar Hotkey ---
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
      e.preventDefault();
      triggerSpacebarAction();
    }
  });

  function triggerSpacebarAction() {
    fetch('/api/tts/stop', { method: 'POST' }).catch(() => {});
    playEarcon('listen');
    setOrbState('LISTENING');
    heardText.textContent = 'Spacebar pressed: Listening for query...';

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'spacebar' }));
    } else {
      fetch('/api/voice/trigger_space', { method: 'POST' }).catch(() => {});
    }
  }

  // --- Status Orb State Renderer ---
  function setOrbState(state) {
    if (!statusOrb || !statusLabel) return;
    const cleanState = (state || 'STANDBY').toUpperCase();
    statusLabel.textContent = cleanState;

    statusOrb.className = 'status-orb';
    if (cleanState === 'LISTENING') {
      statusOrb.classList.add('status-listening');
    } else if (cleanState === 'ANALYZING') {
      statusOrb.classList.add('status-analyzing');
    } else if (cleanState === 'SPEAKING') {
      statusOrb.classList.add('status-speaking');
    } else {
      statusOrb.classList.add('status-standby');
    }
  }

  // --- Assistant Spoken & Text Responses ---
  let lastSpokenText = '';

  function displayAssistantResponse(data) {
    playEarcon('answer');
    const text = data.response || 'Ready.';
    lastSpokenText = text;

    if (currentMode === 'test-lab') {
      const tlText = document.getElementById('testLabResponseText');
      const tlLat = document.getElementById('testLabResponseLatency');
      const tlIntent = document.getElementById('testLabResponseIntent');
      if (tlText) tlText.textContent = text;
      if (tlLat) tlLat.textContent = `Latency: ${(data.time_to_answer_ms || 0).toFixed(0)} ms`;
      if (tlIntent && data.result) tlIntent.textContent = `Intent: ${data.result.intent || 'QUERY'}`;
    } else {
      if (assistantResponseText) assistantResponseText.textContent = text;
      if (responseLatency) responseLatency.textContent = `Latency: ${(data.time_to_answer_ms || 0).toFixed(0)} ms`;
      if (responseIntent && data.result) responseIntent.textContent = `Intent: ${data.result.intent || 'QUERY'}`;
    }

    if (isReadAloudEnabled()) {
      setOrbState('SPEAKING');
    } else {
      setOrbState('STANDBY');
    }
  }

  async function replayLastSpeech() {
    if (!lastSpokenText) return;
    try {
      setOrbState('SPEAKING');
      const res = await fetch('/api/tts/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: lastSpokenText })
      });
      const data = await res.json();
      if (data && data.status === 'success') {
        if (ttsNoteWalk) ttsNoteWalk.classList.add('hidden');
        if (ttsNoteTestLab) ttsNoteTestLab.classList.add('hidden');
        return;
      }
    } catch (err) {
      console.warn('Backend TTS replay request error:', err);
    }

    // Fallback to Web Speech API
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(lastSpokenText);
      utterance.rate = 1.05;
      utterance.onend = () => setOrbState('STANDBY');
      window.speechSynthesis.speak(utterance);
    } else {
      if (ttsNoteWalk) ttsNoteWalk.classList.remove('hidden');
      if (ttsNoteTestLab) ttsNoteTestLab.classList.remove('hidden');
      setOrbState('STANDBY');
    }
  }

  if (replayBtn) replayBtn.addEventListener('click', replayLastSpeech);
  const testLabReplayBtn = document.getElementById('testLabReplayBtn');
  if (testLabReplayBtn) testLabReplayBtn.addEventListener('click', replayLastSpeech);

  // --- Query Submission (Walk Mode) ---
  if (queryForm) {
    queryForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const query = queryInput.value.trim();
      if (!query) return;
      queryInput.value = '';
      sendQuery(query);
    });
  }

  promptChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const q = chip.dataset.query;
      if (q) sendQuery(q);
    });
  });

  async function sendQuery(query) {
    // Starting a new query stops any answer still being spoken
    fetch('/api/tts/stop', { method: 'POST' }).catch(() => {});

    setOrbState('ANALYZING');
    playEarcon('thinking');
    heardText.textContent = `Processing query: "${query}"`;

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, read_aloud: isReadAloudEnabled() })
      });
      const data = await res.json();
      if (data.status === 'success') {
        displayAssistantResponse(data);
      }
    } catch (err) {
      console.warn('Query failed:', err);
      setOrbState('STANDBY');
    }
  }

  // --- WebSocket Connection ---
  let ws = null;
  let wsReconnectTimer = null;

  function connectWebSocket() {
    if (wsReconnectTimer) clearTimeout(wsReconnectTimer);
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[WebSocket] Connected to AI Vision Assistant HUD server.');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
      } catch (err) {
        console.warn('WS parse error:', err);
      }
    };

    ws.onclose = () => {
      console.warn('[WebSocket] Closed. Reconnecting in 2s...');
      wsReconnectTimer = setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
      console.error('[WebSocket] Error:', err);
      ws.close();
    };
  }

  function handleServerMessage(msg) {
    if (msg.type === 'telemetry') {
      // Metrics Bar
      camFpsVal.textContent = (msg.camera_fps || 0).toFixed(1);
      detFpsVal.textContent = (msg.detection_fps || 0).toFixed(1);
      detLatVal.textContent = `${(msg.detection_latency_ms || 0).toFixed(1)} ms`;

      // Status Orb (if not in an override state)
      if (msg.assistant_status) {
        setOrbState(msg.assistant_status);
      }

      // Live Heard Line
      if (msg.last_heard) {
        heardText.textContent = msg.last_heard;
      }

      // Update Spatial Zones
      updateSpatialZones(msg.detections || []);
    } else if (msg.type === 'voice_state') {
      setOrbState(msg.state);
      if (msg.data && msg.data.text) {
        heardText.textContent = `Heard: "${msg.data.text}"`;
      }
      if (msg.state === 'AWAKE') {
        playEarcon('wake');
        heardText.textContent = 'Wake detected! Listening for question...';
      } else if (msg.state === 'LISTENING') {
        playEarcon('listen');
      } else if (msg.state === 'ANALYZING') {
        playEarcon('thinking');
      }
    } else if (msg.type === 'assistant_response') {
      displayAssistantResponse(msg);
    } else if (msg.type === 'dqn_decision') {
      renderDqnDecision(msg.decision);
    }
  }

  let totalAlertsCount = 0;
  const dqnTraceContainer = document.getElementById('dqnTraceContainer');
  const alertsCountBadge = document.getElementById('alertsCountBadge');

  function renderDqnDecision(d) {
    if (!d || !dqnTraceContainer) return;

    // Remove empty msg if present
    const emptyMsg = dqnTraceContainer.querySelector('.dqn-empty-msg');
    if (emptyMsg) emptyMsg.remove();

    if (d.spoke_alert) {
      totalAlertsCount++;
      if (alertsCountBadge) alertsCountBadge.textContent = `${totalAlertsCount} Alerts`;
    }

    const entry = document.createElement('div');
    const actClass = (d.effective_action || 'WAIT').toLowerCase();
    entry.className = `dqn-entry entry-${actClass}`;

    const qStr = (d.q_values || []).map((v) => v.toFixed(2)).join(', ');
    let extraNote = '';
    if (d.in_cooldown) extraNote = ` [Cooldown: ${d.cooldown_remaining_sec}s]`;
    else if (d.suppressed) extraNote = ' [Suppressed by Query]';

    entry.innerHTML = `
      <div class="dqn-row-top">
        <span class="dqn-obj-label">${d.object.toUpperCase()} (${d.position})</span>
        <span class="action-badge action-${actClass}">${d.effective_action}</span>
      </div>
      <div class="dqn-row-details">
        <span>Risk: ${d.risk} (${d.risk_confidence})</span>
        <span>Q: [${qStr}]${extraNote}</span>
      </div>
    `;

    dqnTraceContainer.prepend(entry);
    while (dqnTraceContainer.children.length > 30) {
      dqnTraceContainer.removeChild(dqnTraceContainer.lastChild);
    }
  }

  // --- Video Source Toggle Buttons (Live Walk Mode) ---
  const sourceWebcamBtn = document.getElementById('sourceWebcamBtn');
  const sourceVideoBtn = document.getElementById('sourceVideoBtn');
  const sourceLabel = document.getElementById('sourceLabel');
  const telemCam = document.getElementById('telemCam');

  async function switchVideoSource(type) {
    try {
      const res = await fetch('/api/video_source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: type })
      });
      const data = await res.json();
      if (data.status === 'success') {
        if (type === 'video') {
          sourceVideoBtn.classList.add('active');
          sourceWebcamBtn.classList.remove('active');
          if (sourceLabel) sourceLabel.textContent = 'DEMO VIDEO LIVE';
          if (telemCam) telemCam.textContent = 'Looped Video File (test_video.mp4)';
        } else {
          sourceWebcamBtn.classList.add('active');
          sourceVideoBtn.classList.remove('active');
          if (sourceLabel) sourceLabel.textContent = 'WEBCAM LIVE';
          if (telemCam) telemCam.textContent = 'Webcam DirectShow (Idx 0)';
        }
      }
    } catch (err) {
      console.warn('Failed to switch video source:', err);
    }
  }

  if (sourceWebcamBtn) sourceWebcamBtn.addEventListener('click', () => switchVideoSource('webcam'));
  if (sourceVideoBtn) sourceVideoBtn.addEventListener('click', () => switchVideoSource('video'));

  function updateSpatialZones(detections) {
    let leftCount = 0;
    let centerCount = 0;
    let rightCount = 0;

    detections.forEach((det) => {
      const pos = (det.spatial_position || '').toLowerCase();
      if (pos.includes('left')) leftCount++;
      else if (pos.includes('right')) rightCount++;
      else centerCount++;
    });

    zoneLeftCount.textContent = leftCount;
    zoneCenterCount.textContent = centerCount;
    zoneRightCount.textContent = rightCount;

    zoneLeft.classList.toggle('has-obstacle', leftCount > 0);
    zoneCenter.classList.toggle('has-obstacle', centerCount > 0);
    zoneRight.classList.toggle('has-obstacle', rightCount > 0);
  }

  // ==========================================================================
  // PHASE 5: TEST LAB LOGIC
  // ==========================================================================
  const activeMediaBadge = document.getElementById('activeMediaBadge');
  const sampleButtons = document.querySelectorAll('.sample-btn');
  const uploadDropzone = document.getElementById('uploadDropzone');
  const mediaFileInput = document.getElementById('mediaFileInput');
  const browseBtn = document.getElementById('browseBtn');

  const testEmptyPlaceholder = document.getElementById('testEmptyPlaceholder');
  const movementMutedNotice = document.getElementById('movementMutedNotice');
  const testImageWrapper = document.getElementById('testImageWrapper');
  const testVideoWrapper = document.getElementById('testVideoWrapper');
  const testLabImage = document.getElementById('testLabImage');
  const testLabVideo = document.getElementById('testLabVideo');
  const viewportMediaTitle = document.getElementById('viewportMediaTitle');
  const viewportMediaSub = document.getElementById('viewportMediaSub');
  const streamVideoBtn = document.getElementById('streamVideoBtn');
  const mediaDetectionsSummary = document.getElementById('mediaDetectionsSummary');

  const movementAnalysisCard = document.getElementById('movementAnalysisCard');
  const startAnalysisBtn = document.getElementById('startAnalysisBtn');
  const analysisProgressSection = document.getElementById('analysisProgressSection');
  const analysisProgressBar = document.getElementById('analysisProgressBar');
  const analysisStatusText = document.getElementById('analysisStatusText');
  const analysisPercentText = document.getElementById('analysisPercentText');
  const analysisElapsedText = document.getElementById('analysisElapsedText');
  const timelineContainer = document.getElementById('timelineContainer');
  const timelineCountBadge = document.getElementById('timelineCountBadge');

  const testLabQueryForm = document.getElementById('testLabQueryForm');
  const testLabQueryInput = document.getElementById('testLabQueryInput');
  const testLabPromptChips = document.querySelectorAll('.test-lab-prompt-chip');

  // Latency trace elements
  const pipelineTotalMs = document.getElementById('pipelineTotalMs');
  const barNlp = document.getElementById('barNlp');
  const latNlp = document.getElementById('latNlp');
  const barYolo = document.getElementById('barYolo');
  const latYolo = document.getElementById('latYolo');
  const barSpatial = document.getElementById('barSpatial');
  const latSpatial = document.getElementById('latSpatial');
  const barRisk = document.getElementById('barRisk');
  const latRisk = document.getElementById('latRisk');
  const barDqn = document.getElementById('barDqn');
  const latDqn = document.getElementById('latDqn');

  // Sample Picker Click
  sampleButtons.forEach((btn) => {
    btn.addEventListener('click', async () => {
      const filename = btn.dataset.sample;
      if (!filename) return;

      sampleButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      if (testLabVideo) testLabVideo.pause();

      try {
        const res = await fetch('/api/test_lab/select_sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: filename })
        });
        const data = await res.json();
        if (data.status === 'success') {
          renderTestLabMedia(data.media);
        }
      } catch (err) {
        console.warn('Sample selection error:', err);
      }
    });
  });

  // Browse File Button
  if (browseBtn && mediaFileInput) {
    browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      mediaFileInput.click();
    });
  }

  if (uploadDropzone && mediaFileInput) {
    uploadDropzone.addEventListener('click', () => mediaFileInput.click());

    ['dragenter', 'dragover'].forEach((eventName) => {
      uploadDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        uploadDropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      uploadDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        uploadDropzone.classList.remove('dragover');
      });
    });

    uploadDropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleUploadFile(e.dataTransfer.files[0]);
      }
    });

    mediaFileInput.addEventListener('change', (e) => {
      if (mediaFileInput.files && mediaFileInput.files.length > 0) {
        handleUploadFile(mediaFileInput.files[0]);
      }
    });
  }

  // Upload File Handler
  async function handleUploadFile(file) {
    if (!file) return;
    if (testLabVideo) testLabVideo.pause();

    // Client-side size validation (max 50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      alert(`File exceeds 50MB limit (${(file.size / (1024 * 1024)).toFixed(1)} MB). Please select a smaller media file.`);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      activeMediaBadge.textContent = 'Uploading & Validating...';
      const res = await fetch('/api/test_lab/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.status === 'success') {
        sampleButtons.forEach((b) => b.classList.remove('active'));
        renderTestLabMedia(data.media);
      } else {
        alert(data.error || 'Upload failed.');
        loadCurrentTestLabMedia();
      }
    } catch (err) {
      alert('Upload request error: ' + err);
      loadCurrentTestLabMedia();
    }
  }

  // Load Current Media on Tab Open
  async function loadCurrentTestLabMedia() {
    try {
      const res = await fetch('/api/test_lab/current_media');
      const data = await res.json();
      if (data.status === 'success' && data.media) {
        renderTestLabMedia(data.media);
      }
    } catch (err) {
      console.warn('Failed to load current test lab media:', err);
    }
  }

  // Render Test Lab Media Viewport
  function renderTestLabMedia(media) {
    if (!media || (!media.type && !media.path)) {
      if (testEmptyPlaceholder) testEmptyPlaceholder.classList.remove('hidden');
      if (testImageWrapper) testImageWrapper.classList.add('hidden');
      if (testVideoWrapper) testVideoWrapper.classList.add('hidden');
      if (movementAnalysisCard) movementAnalysisCard.classList.add('hidden');
      if (movementMutedNotice) movementMutedNotice.classList.add('hidden');
      if (streamVideoBtn) streamVideoBtn.classList.add('hidden');
      if (testLabVideo) {
        testLabVideo.pause();
        testLabVideo.removeAttribute('src');
        testLabVideo.load();
      }
      if (testLabImage) testLabImage.removeAttribute('src');
      if (activeMediaBadge) activeMediaBadge.textContent = 'No media loaded';
      if (viewportMediaTitle) viewportMediaTitle.textContent = 'Media detection viewport';
      if (viewportMediaSub) viewportMediaSub.textContent = 'Select a sample asset or upload media';
      if (mediaDetectionsSummary) mediaDetectionsSummary.innerHTML = '';
      return;
    }

    if (testEmptyPlaceholder) testEmptyPlaceholder.classList.add('hidden');

    const isVideo = media.type === 'video';
    const filename = media.name || 'media';

    if (activeMediaBadge) {
      activeMediaBadge.textContent = `${filename} (${isVideo ? 'Video' : 'Image'})`;
    }

    if (viewportMediaTitle) {
      viewportMediaTitle.textContent = `${filename} — ${isVideo ? 'Video' : 'Image'}`;
    }

    if (viewportMediaSub) {
      if (isVideo) {
        viewportMediaSub.textContent = `Video resolution: ${media.width}x${media.height} | ${media.total_frames || 0} frames @ ${media.fps || 30} FPS`;
      } else {
        viewportMediaSub.textContent = `Image resolution: ${media.width}x${media.height} | Detected ${(media.detections || []).length} objects`;
      }
    }

    if (isVideo) {
      // VIDEO ONLY: Reset and hide image view
      if (testImageWrapper) testImageWrapper.classList.add('hidden');
      if (testLabImage) testLabImage.removeAttribute('src');

      // Show video player and load video
      if (testVideoWrapper) testVideoWrapper.classList.remove('hidden');
      if (testLabVideo) {
        testLabVideo.src = `/api/test_lab/media_file?t=${Date.now()}`;
        testLabVideo.load();
      }

      // Show movement analysis section and hide muted notice
      if (movementAnalysisCard) movementAnalysisCard.classList.remove('hidden');
      if (movementMutedNotice) movementMutedNotice.classList.add('hidden');
      if (streamVideoBtn) streamVideoBtn.classList.remove('hidden');
    } else {
      // IMAGE ONLY: Reset and pause video player
      if (testVideoWrapper) testVideoWrapper.classList.add('hidden');
      if (testLabVideo) {
        testLabVideo.pause();
        testLabVideo.removeAttribute('src');
        testLabVideo.load();
      }

      // Show image view and set src
      if (testImageWrapper) testImageWrapper.classList.remove('hidden');
      if (testLabImage) {
        testLabImage.src = `/api/test_lab/annotated_image?t=${Date.now()}`;
      }

      // Hide movement analysis section and show muted notice
      if (movementAnalysisCard) movementAnalysisCard.classList.add('hidden');
      if (movementMutedNotice) movementMutedNotice.classList.remove('hidden');
      if (streamVideoBtn) streamVideoBtn.classList.add('hidden');
    }

    // Populate Detection Summary Chips
    if (mediaDetectionsSummary) {
      mediaDetectionsSummary.innerHTML = '';
      const dets = media.detections || [];
      if (dets.length > 0) {
        dets.forEach((d) => {
          const chip = document.createElement('span');
          chip.className = 'det-summary-chip';
          chip.textContent = `${d.object} (${(d.confidence * 100).toFixed(0)}%) - ${d.position}`;
          mediaDetectionsSummary.appendChild(chip);
        });
      }
    }
  }

  // Stream Video in Live View Button
  if (streamVideoBtn) {
    streamVideoBtn.addEventListener('click', () => {
      setMode('walk');
      switchVideoSource('video');
    });
  }

  // --- "Analyse Movement" Background Job & Timeline ---
  let analysisPollTimer = null;

  if (startAnalysisBtn) {
    startAnalysisBtn.addEventListener('click', async () => {
      startAnalysisBtn.disabled = true;
      if (analysisProgressSection) analysisProgressSection.classList.remove('hidden');
      if (analysisStatusText) analysisStatusText.textContent = 'Starting real process_video() with LSTM...';
      if (timelineContainer) {
        timelineContainer.innerHTML = '<div class="timeline-empty-msg">Processing video frames with YOLO tracking and Approaching LSTM...</div>';
      }

      try {
        const res = await fetch('/api/test_lab/analyze_movement', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'started' || data.status === 'running') {
          pollMovementProgress();
        } else if (data.error) {
          alert('Analysis error: ' + data.error);
          startAnalysisBtn.disabled = false;
        }
      } catch (err) {
        alert('Analysis request failed: ' + err);
        startAnalysisBtn.disabled = false;
      }
    });
  }

  function pollMovementProgress() {
    if (analysisPollTimer) clearInterval(analysisPollTimer);

    analysisPollTimer = setInterval(async () => {
      try {
        const res = await fetch('/api/test_lab/movement_progress');
        const data = await res.json();
        const job = data.job || {};

        if (analysisProgressBar) {
          analysisProgressBar.style.width = `${job.percent || 0}%`;
        }
        if (analysisPercentText) {
          analysisPercentText.textContent = `${(job.percent || 0).toFixed(1)}%`;
        }
        if (analysisStatusText) {
          analysisStatusText.textContent = `Analyzing Frame ${job.current_frame || 0} / ${job.total_frames || 0}`;
        }
        if (analysisElapsedText) {
          analysisElapsedText.textContent = `Elapsed: ${(job.elapsed_sec || 0).toFixed(1)}s`;
        }

        if (job.status === 'completed') {
          clearInterval(analysisPollTimer);
          if (startAnalysisBtn) startAnalysisBtn.disabled = false;
          if (analysisStatusText) {
            analysisStatusText.textContent = `Completed in ${job.elapsed_sec}s (${(job.events || []).length} events detected)`;
          }
          renderMovementTimeline(job.events || []);
        } else if (job.status === 'error') {
          clearInterval(analysisPollTimer);
          if (startAnalysisBtn) startAnalysisBtn.disabled = false;
          if (analysisStatusText) {
            analysisStatusText.textContent = `Error: ${job.error_message}`;
          }
        }
      } catch (e) {
        console.warn('Progress poll error:', e);
      }
    }, 500);
  }

  function renderMovementTimeline(events) {
    if (!timelineContainer) return;
    timelineContainer.innerHTML = '';

    if (timelineCountBadge) {
      timelineCountBadge.textContent = `${events.length} Events`;
    }

    if (events.length === 0) {
      timelineContainer.innerHTML = '<div class="timeline-empty-msg">No high-risk movement alerts triggered in this clip.</div>';
      return;
    }

    events.forEach((ev) => {
      const card = document.createElement('div');
      const movLower = (ev.movement || 'stationary').toLowerCase().replace(' ', '-');
      card.className = `timeline-card event-${movLower}`;

      const movConf = (ev.movement_confidence * 100).toFixed(0);
      const riskConf = (ev.risk_confidence * 100).toFixed(0);

      card.innerHTML = `
        <div class="timeline-left">
          <span class="timeline-frame-label">FRAME ${ev.frame} • TRACK #${ev.track_id}</span>
          <span class="timeline-obj-title">${(ev.object || 'object').toUpperCase()} (${ev.position})</span>
          <div class="timeline-badges-row">
            <span class="timeline-badge badge-${movLower}">${ev.movement.toUpperCase()} (${movConf}%)</span>
            <span class="timeline-badge badge-risk-${(ev.risk || 'low').toLowerCase()}">RISK: ${ev.risk} (${riskConf}%)</span>
          </div>
        </div>
        <div class="timeline-right">
          <span class="timeline-badge badge-action-${(ev.final_action || 'wait').toLowerCase()}">${ev.final_action || 'WAIT'}</span>
        </div>
      `;
      timelineContainer.appendChild(card);
    });
  }

  // --- Test Lab Queries & Pipeline Latency Trace ---
  if (testLabQueryForm) {
    testLabQueryForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const q = testLabQueryInput.value.trim();
      if (!q) return;
      testLabQueryInput.value = '';
      sendTestLabQuery(q);
    });
  }

  testLabPromptChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const q = chip.dataset.query;
      if (q) sendTestLabQuery(q);
    });
  });

  async function sendTestLabQuery(query) {
    // Starting a new query stops any answer still being spoken
    fetch('/api/tts/stop', { method: 'POST' }).catch(() => {});

    setOrbState('ANALYZING');
    playEarcon('thinking');

    try {
      const res = await fetch('/api/test_lab/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, read_aloud: isReadAloudEnabled() })
      });
      const data = await res.json();
      if (data.status === 'success') {
        displayAssistantResponse(data);
        renderPipelineTrace(data.stage_latencies || {}, data.time_to_answer_ms || 0);
      }
    } catch (err) {
      console.warn('Test Lab query failed:', err);
      setOrbState('STANDBY');
    }
  }

  function renderPipelineTrace(latencies, totalMs) {
    if (pipelineTotalMs) {
      pipelineTotalMs.textContent = `Total: ${totalMs.toFixed(1)} ms`;
    }

    const nlp = latencies.nlp_ms || 0;
    const yolo = latencies.yolo_ms || 0;
    const spatial = latencies.spatial_ms || 0;
    const risk = latencies.risk_ms || 0;
    const dqn = latencies.dqn_ms || 0;

    const maxVal = Math.max(1, totalMs);

    if (latNlp) latNlp.textContent = `${nlp.toFixed(1)} ms`;
    if (barNlp) barNlp.style.width = `${Math.min(100, (nlp / maxVal) * 100)}%`;

    if (latYolo) latYolo.textContent = `${yolo.toFixed(1)} ms`;
    if (barYolo) barYolo.style.width = `${Math.min(100, (yolo / maxVal) * 100)}%`;

    if (latSpatial) latSpatial.textContent = `${spatial.toFixed(1)} ms`;
    if (barSpatial) barSpatial.style.width = `${Math.min(100, (spatial / maxVal) * 100)}%`;

    if (latRisk) latRisk.textContent = `${risk.toFixed(1)} ms`;
    if (barRisk) barRisk.style.width = `${Math.min(100, (risk / maxVal) * 100)}%`;

    if (latDqn) latDqn.textContent = `${dqn.toFixed(1)} ms`;
    if (barDqn) barDqn.style.width = `${Math.min(100, (dqn / maxVal) * 100)}%`;
  }

  // ==========================================================================
  // PHASE 5: MODEL EVALUATION DASHBOARD
  // ==========================================================================
  const refreshEvalBtn = document.getElementById('refreshEvalBtn');
  if (refreshEvalBtn) {
    refreshEvalBtn.addEventListener('click', loadEvaluationMetrics);
  }

  async function loadEvaluationMetrics() {
    try {
      const res = await fetch('/api/evaluation');
      const resp = await res.json();
      if (resp.status !== 'success' || !resp.data) return;

      const d = resp.data;

      // Executive Metrics
      const overall = d.overall_results || {};
      const evalPrec = document.getElementById('evalPrecision');
      const evalRec = document.getElementById('evalRecall');
      const evalMap50 = document.getElementById('evalMap50');
      const evalMap5095 = document.getElementById('evalMap5095');

      if (evalPrec) evalPrec.textContent = (overall.precision !== undefined) ? `${(overall.precision * 100).toFixed(1)}%` : '--';
      if (evalRec) evalRec.textContent = (overall.recall !== undefined) ? `${(overall.recall * 100).toFixed(1)}%` : '--';
      if (evalMap50) evalMap50.textContent = (overall.mAP50 !== undefined) ? `${(overall.mAP50 * 100).toFixed(1)}%` : '--';
      if (evalMap5095) evalMap5095.textContent = (overall.mAP50_95 !== undefined) ? `${(overall.mAP50_95 * 100).toFixed(1)}%` : '--';

      // Door Detection Specifics
      const door = d.door_results || {};
      const evalDoorPrec = document.getElementById('evalDoorPrecision');
      const evalDoorRec = document.getElementById('evalDoorRecall');
      const evalDoorMap50 = document.getElementById('evalDoorMap50');
      const evalDoorMap5095 = document.getElementById('evalDoorMap5095');

      if (evalDoorPrec) evalDoorPrec.textContent = (door.precision !== undefined) ? `${(door.precision * 100).toFixed(1)}%` : '--';
      if (evalDoorRec) evalDoorRec.textContent = (door.recall !== undefined) ? `${(door.recall * 100).toFixed(1)}%` : '--';
      if (evalDoorMap50) evalDoorMap50.textContent = (door.mAP50 !== undefined) ? `${(door.mAP50 * 100).toFixed(1)}%` : '--';
      if (evalDoorMap5095) evalDoorMap5095.textContent = (door.mAP50_95 !== undefined) ? `${(door.mAP50_95 * 100).toFixed(1)}%` : '--';

      // Comparison Table
      const comp = d.comparison_with_experiment_01 || {};
      const compBody = document.getElementById('evalComparisonTableBody');
      if (compBody) {
        compBody.innerHTML = `
          <tr>
            <td><strong>mAP @ 0.50</strong></td>
            <td>${(comp.experiment_01_mAP50 * 100).toFixed(1)}%</td>
            <td>${(comp.experiment_02_mAP50 * 100).toFixed(1)}%</td>
            <td><span class="delta-badge">+${((comp.experiment_02_mAP50 - comp.experiment_01_mAP50) * 100).toFixed(2)}%</span></td>
          </tr>
          <tr>
            <td><strong>mAP @ 0.50:0.95</strong></td>
            <td>${(comp.experiment_01_mAP50_95 * 100).toFixed(1)}%</td>
            <td>${(comp.experiment_02_mAP50_95 * 100).toFixed(1)}%</td>
            <td><span class="delta-badge">+${((comp.experiment_02_mAP50_95 - comp.experiment_01_mAP50_95) * 100).toFixed(2)}%</span></td>
          </tr>
          <tr>
            <td><strong>Overall Recall</strong></td>
            <td>${(comp.experiment_01_recall * 100).toFixed(1)}%</td>
            <td>${(comp.experiment_02_recall * 100).toFixed(1)}%</td>
            <td><span class="delta-badge">+${((comp.experiment_02_recall - comp.experiment_01_recall) * 100).toFixed(2)}%</span></td>
          </tr>
        `;
      }

      // Dataset Info
      const ds = d.dataset || {};
      const dsInfo = document.getElementById('evalDatasetInfo');
      if (dsInfo) {
        dsInfo.innerHTML = `
          <div class="info-row"><span class="info-key">Base Dataset:</span><span class="info-val">${ds.base}</span></div>
          <div class="info-row"><span class="info-key">Custom Extension:</span><span class="info-val">${ds.added_dataset}</span></div>
          <div class="info-row"><span class="info-key">Total Model Classes:</span><span class="info-val">${ds.classes}</span></div>
          <div class="info-row"><span class="info-key">Train Images / Instances:</span><span class="info-val">${ds.train_images} / ${ds.train_instances}</span></div>
          <div class="info-row"><span class="info-key">Validation Instances:</span><span class="info-val">${ds.val_instances} (COCO: ${ds.coco_val_instances}, Door: ${ds.door_val_instances})</span></div>
        `;
      }

      // Training Hyperparameters
      const tr = d.training || {};
      const trInfo = document.getElementById('evalTrainingInfo');
      if (trInfo) {
        trInfo.innerHTML = `
          <div class="info-row"><span class="info-key">Model Base Architecture:</span><span class="info-val">${d.model} (${d.pretrained ? 'Pretrained' : 'Custom'})</span></div>
          <div class="info-row"><span class="info-key">Training Epochs:</span><span class="info-val">${tr.epochs}</span></div>
          <div class="info-row"><span class="info-key">Input Image Size:</span><span class="info-val">${tr.image_size} x ${tr.image_size}</span></div>
          <div class="info-row"><span class="info-key">Batch Size:</span><span class="info-val">${tr.batch_size}</span></div>
          <div class="info-row"><span class="info-key">Optimizer:</span><span class="info-val">${tr.optimizer} (Seed: ${tr.seed})</span></div>
        `;
      }

      // CNN Architecture Badges
      const cnnGrid = document.getElementById('evalCnnGrid');
      if (cnnGrid) {
        cnnGrid.innerHTML = '';
        const cnnList = d.cnn_architectures_implemented || ['VGG', 'RESNET', 'MOBILENET', 'LENET5', 'GOOGLENET', 'ALEXNET'];
        cnnList.forEach((arch) => {
          const chip = document.createElement('div');
          chip.className = 'cnn-card-chip';
          chip.innerHTML = `<span class="cnn-chip-dot"></span><span>${arch}</span>`;
          cnnGrid.appendChild(chip);
        });
      }

      // Notes List
      const notesList = document.getElementById('evalNotesList');
      if (notesList) {
        notesList.innerHTML = '';
        (d.notes || []).forEach((note) => {
          const li = document.createElement('li');
          li.textContent = note;
          notesList.appendChild(li);
        });
      }
    } catch (err) {
      console.warn('Failed to load evaluation report:', err);
    }
  }

  // Connect WebSocket on load
  window.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
  });
})();
