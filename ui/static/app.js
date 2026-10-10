/**
 * AI Vision Assistant — Interactive Cyber-Aura HUD Client Logic
 * Features:
 * - Web Audio API initialization on explicit user gesture
 * - Synthesized futuristic earcons (wake, listening, thinking, answer ready)
 * - Spacebar hotkey with e.preventDefault() to eliminate scrolling
 * - Window focus / blur warning detector
 * - Real-time WebSocket telemetry & voice state synchronization
 * - Direct query submission & prompt chip dispatch
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

  const zoneLeftCount = document.getElementById('zoneLeftCount');
  const zoneCenterCount = document.getElementById('zoneCenterCount');
  const zoneRightCount = document.getElementById('zoneRightCount');
  const zoneLeft = document.getElementById('zoneLeft');
  const zoneCenter = document.getElementById('zoneCenter');
  const zoneRight = document.getElementById('zoneRight');

  const responseText = document.getElementById('assistantResponseText');
  const responseLatency = document.getElementById('responseLatency');
  const responseIntent = document.getElementById('responseIntent');
  const replayBtn = document.getElementById('replayBtn');

  const queryForm = document.getElementById('queryForm');
  const queryInput = document.getElementById('queryInput');
  const promptChips = document.querySelectorAll('.prompt-chip');

  // --- Startup & Overlay Unlock ---
  function unlockAssistant() {
    if (audioUnlocked) return;
    initAudioContext();
    playEarcon('wake');

    // Spoken startup greeting
    if ('speechSynthesis' in window) {
      const greetingUtterance = new SpeechSynthesisUtterance('AI Vision Assistant initialized.');
      greetingUtterance.rate = 1.05;
      greetingUtterance.pitch = 1.0;
      window.speechSynthesis.speak(greetingUtterance);
    }

    startOverlay.classList.add('dissolve');
    setTimeout(() => {
      startOverlay.style.display = 'none';
    }, 500);

    // Focus window
    window.focus();
  }

  startBtn.addEventListener('click', unlockAssistant);
  startOverlay.addEventListener('click', unlockAssistant);

  // --- Window Focus & Blur Warning Handling ---
  window.addEventListener('blur', () => {
    unfocusedWarning.classList.remove('hidden');
  });

  window.addEventListener('focus', () => {
    unfocusedWarning.classList.add('hidden');
  });

  // --- Spacebar & Keyboard Hotkey Management ---
  window.addEventListener('keydown', (e) => {
    // If overlay is still present, any key unlocks it
    if (!audioUnlocked) {
      unlockAssistant();
      if (e.code === 'Space') e.preventDefault();
      return;
    }

    // Space key fallback handling
    if (e.code === 'Space') {
      const activeEl = document.activeElement;
      const isInput = activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA');

      // Only intercept space if not currently typing in a text field
      if (!isInput) {
        e.preventDefault(); // Prevent scrolling the webpage!
        triggerSpacebarListening();
      }
    }
  });

  function triggerSpacebarListening() {
    playEarcon('listen');
    setOrbState('LISTENING');
    heardText.textContent = 'Spacebar active: Listening for direct speech query...';

    // Notify backend via WebSocket or API
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'spacebar' }));
    } else {
      fetch('/api/voice/trigger_space', { method: 'POST' }).catch(() => {});
    }
  }

  // --- State Orb Visualizer ---
  function setOrbState(state) {
    statusOrb.className = 'status-orb';
    const s = (state || 'STANDBY').toUpperCase();
    statusLabel.textContent = s;

    switch (s) {
      case 'LISTENING':
        statusOrb.classList.add('status-listening');
        break;
      case 'AWAKE':
        statusOrb.classList.add('status-awake');
        break;
      case 'ANALYZING':
        statusOrb.classList.add('status-analyzing');
        break;
      case 'SPEAKING':
        statusOrb.classList.add('status-speaking');
        break;
      case 'STANDBY':
      default:
        statusOrb.classList.add('status-standby');
        break;
    }
  }

  // --- Query Dispatch ---
  async function submitQuery(queryStr) {
    const cleanQuery = (queryStr || '').trim();
    if (!cleanQuery) return;

    playEarcon('thinking');
    setOrbState('ANALYZING');
    heardText.textContent = `Query: "${cleanQuery}"`;

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: cleanQuery })
      });
      const data = await res.json();
      displayAssistantResponse(data);
    } catch (err) {
      console.error('Query failed:', err);
      responseText.textContent = 'Error processing query. System ready for next question.';
      setOrbState('STANDBY');
    }
  }

  function displayAssistantResponse(data) {
    playEarcon('answer');
    setOrbState('SPEAKING');

    const reply = data.response || 'No response generated.';
    responseText.textContent = reply;
    responseLatency.textContent = `Latency: ${data.time_to_answer_ms || 0} ms`;

    const intent = (data.result && data.result.intent) || 'ANSWER_READY';
    responseIntent.textContent = `Intent: ${intent}`;

    setTimeout(() => {
      setOrbState('STANDBY');
    }, 3000);
  }

  queryForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = queryInput.value;
    queryInput.value = '';
    submitQuery(query);
  });

  promptChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      submitQuery(query);
    });
  });

  replayBtn.addEventListener('click', () => {
    const text = responseText.textContent;
    if (text && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(utterance);
    }
  });

  // --- WebSocket Real-Time Telemetry & Event Stream ---
  let ws = null;
  let wsReconnectTimer = null;

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[WebSocket] Connected to vision server.');
      if (wsReconnectTimer) clearTimeout(wsReconnectTimer);
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

    const qStr = (d.q_values || []).map(v => v.toFixed(2)).join(', ');
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
    // Keep max 30 visible entries in UI log
    while (dqnTraceContainer.children.length > 30) {
      dqnTraceContainer.removeChild(dqnTraceContainer.lastChild);
    }
  }

  // --- Video Source Toggle Buttons ---
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

  if (sourceWebcamBtn) {
    sourceWebcamBtn.addEventListener('click', () => switchVideoSource('webcam'));
  }
  if (sourceVideoBtn) {
    sourceVideoBtn.addEventListener('click', () => switchVideoSource('video'));
  }

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

  // Connect WebSocket on page load
  window.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
  });
})();
