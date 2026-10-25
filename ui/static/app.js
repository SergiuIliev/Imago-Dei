/**
 * Personal AI Time Capsule — Sergiu
 * Web Interface Controller & Audio UX Manager
 * Inspired by Constantin Brâncuși: "Simplicity is complexity resolved."
 */

// In-memory cache for memories by ID to avoid inline HTML quoting/newline syntax errors
window._memoriesCache = {};

// --- Global Singleton Audio Controller (No Concurrent Overlaps, Toggle Stop, Visual State) ---
class CapsuleAudioController {
  constructor() {
    this.activeAudio = null;
    this.activeUrl = null;
    this.activeButton = null;
    this.activeButtonOriginalHtml = null;
  }

  stop() {
    if (this.activeAudio) {
      try {
        this.activeAudio.pause();
        this.activeAudio.currentTime = 0;
      } catch (e) {
        console.warn("Audio pause error:", e);
      }
      this.activeAudio = null;
    }
    this.resetButtonState();
    this.activeUrl = null;
  }

  resetButtonState() {
    if (this.activeButton) {
      this.activeButton.classList.remove("is-playing");
      if (this.activeButtonOriginalHtml !== null) {
        this.activeButton.innerHTML = this.activeButtonOriginalHtml;
      }
      this.activeButton = null;
      this.activeButtonOriginalHtml = null;
    }
  }

  toggle(url, triggerButton = null) {
    if (!url) return;

    // Resolve trigger button if event occurred without explicit argument
    if (!triggerButton && window.event) {
      const target = window.event.currentTarget || window.event.target;
      if (target) {
        triggerButton = target.closest("button") || target;
      }
    }

    // If clicking the SAME clip that is currently playing -> PAUSE and STOP
    if (this.activeAudio && this.activeUrl === url && !this.activeAudio.paused) {
      this.stop();
      return;
    }

    // STOP previous audio before starting new one (NO CONCURRENT OVERLAPS)
    this.stop();

    // Pause studio audio player if playing
    const studioPlayer = document.getElementById("ttsAudioPlayer");
    if (studioPlayer && !studioPlayer.paused) {
      try { studioPlayer.pause(); } catch (_) {}
    }

    this.activeUrl = url;
    const audio = new Audio(url);
    this.activeAudio = audio;

    // Toggle button visual state
    if (triggerButton) {
      this.activeButton = triggerButton;
      this.activeButtonOriginalHtml = triggerButton.innerHTML;
      triggerButton.classList.add("is-playing");
      const isSmall = triggerButton.classList.contains("hear-clip-btn");
      triggerButton.innerHTML = `
        <span class="material-symbols-outlined" style="font-size:${isSmall ? '13px' : '16px'};">stop</span>
        <span>Stop</span>
      `;
    }

    audio.onended = () => {
      this.stop();
    };

    audio.onerror = (e) => {
      console.error("Audio playback error:", e);
      this.stop();
    };

    audio.play().catch(err => {
      console.warn("Audio play prevented or interrupted:", err);
      this.stop();
    });
  }
}

const audioController = new CapsuleAudioController();

// Universal global functions for HTML onclick handlers
window.playAudio = function(url, triggerButton = null) {
  audioController.toggle(url, triggerButton);
};

window.toggleAudio = function(url, triggerButton = null) {
  audioController.toggle(url, triggerButton);
};

window.stopAllAudio = function() {
  audioController.stop();
};

document.addEventListener("DOMContentLoaded", () => {
  // Navigation & Page Title Sync
  const navItems = document.querySelectorAll(".m3-nav-item, .nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const pageTitle = document.getElementById("pageTitle");
  const pageSubhead = document.getElementById("pageSubhead");

  const tabMeta = {
    "conversation": {
      title: "Dialogue with My Past & Present Self",
      subhead: "Evidence-grounded conversational synthesis across chronological eras"
    },
    "memories": {
      title: "Memory Vault & Evidence Index",
      subhead: "Inspect, search, verify, and correct atomic memories with full provenance"
    },
    "beliefs": {
      title: "Temporal Beliefs & Value Evolution",
      subhead: "Tracking how philosophies, habits, and priorities evolved over time without erasing the past"
    },
    "ingestion": {
      title: "Archive Ingestion Pipeline",
      subhead: "Import personal vlogs, run Whisper transcription, and manage media checksums"
    },
    "voice-avatar": {
      title: "Expressive Voice & Avatar Studio",
      subhead: "Acoustic prosody controls, local demonstration voice, and future video avatar rendering"
    },
    "system": {
      title: "System Configuration & Disaster Recovery",
      subhead: "Host specifications, model registry status, local invariants, and backup management"
    },
    "about": {
      title: "About & Architectural Evolution",
      subhead: "Chronicle of behind-the-scenes LLM fine-tuning, memory transformations, and acoustic engineering across versions"
    }
  };

  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const targetTab = item.getAttribute("data-tab");
      navItems.forEach(n => n.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      item.classList.add("active");
      const targetPane = document.getElementById(`tab-${targetTab}`);
      if (targetPane) targetPane.classList.add("active");

      // Update Top App Bar titles
      if (tabMeta[targetTab]) {
        if (pageTitle) pageTitle.innerText = tabMeta[targetTab].title;
        if (pageSubhead) pageSubhead.innerText = tabMeta[targetTab].subhead;
      }

      // Tab specific refresh
      if (targetTab === "memories") loadMemories();
      if (targetTab === "beliefs") loadBeliefs();
      if (targetTab === "ingestion") loadMediaInventory();
    });
  });

  // Stop global audio if user navigates tabs to prevent background phantom audio
  navItems.forEach(item => {
    item.addEventListener("click", () => {
      audioController.stop();
    });
  });

  // Era Selector
  const eraSelect = document.getElementById("eraSelect");
  const currentEraTag = document.getElementById("currentEraTag");
  const eraDescription = document.getElementById("eraDescription");

  const eraDescriptions = {
    "early_20s": "Early 20s (2014-2018): High curiosity, exploratory coding, eager experimentation, discovering independence.",
    "mid_late_20s": "Mid-to-Late 20s (2019-2022): Scalable systems, smart home automation, infrastructure resilience, pragmatic architecture.",
    "present": "Present Day (2023-Present): High clarity, autonomy, agentic AI systems, long-term personal preservation, and calm strategic thinking."
  };

  eraSelect.addEventListener("change", () => {
    const selected = eraSelect.value;
    if (eraDescriptions[selected]) {
      eraDescription.innerText = eraDescriptions[selected];
    }
    const selectedText = eraSelect.options[eraSelect.selectedIndex].text;
    if (currentEraTag) currentEraTag.innerText = selectedText;
  });

  // Chat Logic & Multi-Turn Session
  const chatMessages = document.getElementById("chatMessages");
  const chatInput = document.getElementById("chatInput");
  const sendMessageBtn = document.getElementById("sendMessageBtn");
  const voiceAutoPlay = document.getElementById("voiceAutoPlay");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const micRecordBtn = document.getElementById("micRecordBtn");
  const micStatusIndicator = document.getElementById("micStatusIndicator");
  const micStatusText = document.getElementById("micStatusText");
  const micIcon = document.getElementById("micIcon");

  let currentSessionId = null;
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;

  // Initialize or fetch conversational session
  async function initSession() {
    try {
      const res = await fetch("/api/chat/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "Web Dialogue" })
      });
      if (res.ok) {
        const data = await res.json();
        currentSessionId = data.session_id;
      }
    } catch (e) {
      console.warn("Session init fallback:", e);
    }
  }
  initSession();

  // Push-to-Talk Microphone Recording with MediaRecorder API
  if (micRecordBtn) {
    micRecordBtn.addEventListener("click", async () => {
      if (!isRecording) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          audioChunks = [];
          const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") 
            ? "audio/webm;codecs=opus" 
            : (MediaRecorder.isTypeSupported("audio/mp4") ? "audio/mp4" : "");
          mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

          mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
          };

          mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            if (audioChunks.length === 0) return;
            const blobType = mediaRecorder.mimeType || "audio/webm";
            const audioBlob = new Blob(audioChunks, { type: blobType });
            await transcribeAudioAndSend(audioBlob);
          };

          mediaRecorder.start();
          isRecording = true;
          micRecordBtn.classList.add("recording");
          if (micIcon) micIcon.innerText = "stop";
          if (micStatusIndicator) micStatusIndicator.style.display = "flex";
          if (micStatusText) micStatusText.innerText = "Listening... Speak naturally, then click mic to transcribe.";
        } catch (err) {
          alert(`Microphone access error: ${err.message}`);
        }
      } else {
        isRecording = false;
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
          mediaRecorder.stop();
        }
        micRecordBtn.classList.remove("recording");
        if (micIcon) micIcon.innerText = "mic";
        if (micStatusText) micStatusText.innerText = "Transcribing speech via local Whisper...";
      }
    });
  }

  async function transcribeAudioAndSend(audioBlob) {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");

    try {
      if (micStatusIndicator) micStatusIndicator.style.display = "flex";
      if (micStatusText) micStatusText.innerText = "Transcribing with local Whisper on Apple Silicon...";

      const resp = await fetch("/api/asr/transcribe", {
        method: "POST",
        body: formData
      });

      if (!resp.ok) throw new Error("Whisper transcription failed");
      const result = await resp.json();

      if (result.transcription && result.transcription.trim()) {
        chatInput.value = result.transcription.trim();
        if (micStatusText) micStatusText.innerText = `Transcribed: "${result.transcription.trim()}"`;
        setTimeout(() => {
          if (micStatusIndicator) micStatusIndicator.style.display = "none";
        }, 2000);
        sendMessage();
      } else {
        if (micStatusIndicator) {
          micStatusIndicator.style.display = "flex";
          if (micStatusText) micStatusText.innerText = "No intelligible speech detected. Please speak closer to the mic.";
          setTimeout(() => {
            if (micStatusIndicator) micStatusIndicator.style.display = "none";
          }, 4000);
        }
      }
    } catch (err) {
      console.warn("Transcription note:", err);
      if (micStatusIndicator) {
        micStatusIndicator.style.display = "flex";
        if (micStatusText) micStatusText.innerText = `Microphone ASR: ${err.message}`;
        setTimeout(() => {
          if (micStatusIndicator) micStatusIndicator.style.display = "none";
        }, 4000);
      }
    }
  }

  async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Append User Bubble
    appendMessage("user", "You", query);
    chatInput.value = "";
    chatInput.focus();

    // Loading Indicator
    const loadingElem = appendLoadingMessage();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          era_id: eraSelect.value,
          synthesize_audio: voiceAutoPlay.checked,
          session_id: currentSessionId
        })
      });

      if (!res.ok) throw new Error("Failed to process conversation turn.");
      const data = await res.json();

      loadingElem.remove();
      appendAssistantMessage(data);

      if (data.audio_url && voiceAutoPlay.checked) {
        audioController.toggle(data.audio_url);
      }
    } catch (err) {
      loadingElem.remove();
      appendMessage("assistant", "System Error", `Unable to generate response: ${err.message}`);
    }
  }

  sendMessageBtn.addEventListener("click", sendMessage);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  clearChatBtn.addEventListener("click", async () => {
    audioController.stop();
    await initSession();
    chatMessages.innerHTML = `
      <div class="m3-bubble assistant message">
        <div class="bubble-top-meta">
          <span class="author-name">
            <span class="material-symbols-outlined" style="font-size:16px;">smart_toy</span>
            <span>Sergiu (Time Capsule)</span>
          </span>
          <span class="m3-era-chip">${eraSelect.options[eraSelect.selectedIndex].text}</span>
        </div>
        <div class="msg-body">
          Conversation refreshed with a clean session. Ask me about any period in my life, personal decisions, or recorded events.
        </div>
      </div>
    `;
  });

  function formatInlineMarkdown(safeText) {
    // Bold: **text**
    let s = safeText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic: *text* (excluding already formatted strong)
    s = s.replace(/(^|[^*])\*([^*]+)\*([^*]|$)/g, '$1<em>$2</em>$3');
    // Inline code: `code`
    s = s.replace(/`([^`]+)`/g, '<code class="chat-inline-code">$1</code>');
    // Line breaks within a paragraph
    s = s.replace(/\n/g, '<br/>');
    return s;
  }

  function formatMessageText(text) {
    if (!text) return "";

    // 1. Separate into raw paragraphs by double newline
    let rawParagraphs = text.trim().split(/\n\s*\n/);

    // If text has no double newlines but is long (> 240 chars) with multiple sentences,
    // intelligently split into natural conversational paragraphs
    if (rawParagraphs.length === 1 && rawParagraphs[0].length > 240) {
      const sents = rawParagraphs[0].match(/[^.!?]+[.!?]+(?:\s+|$)/g);
      if (sents && sents.length >= 3) {
        const splitParas = [];
        let cur = "";
        for (let i = 0; i < sents.length; i++) {
          cur += sents[i];
          // Break into paragraphs after 2-3 sentences (~160+ chars)
          if (cur.length >= 160 && i < sents.length - 1) {
            splitParas.push(cur.trim());
            cur = "";
          }
        }
        if (cur.trim()) splitParas.push(cur.trim());
        if (splitParas.length > 1) {
          rawParagraphs = splitParas;
        }
      }
    }

    // 2. Render each paragraph
    return rawParagraphs.map(para => {
      let p = para.trim();
      if (!p) return "";

      // Check for bullet lists
      if (p.startsWith("- ") || p.startsWith("* ") || p.includes("\n- ") || p.includes("\n* ")) {
        const items = p.split(/\n(?=[-*]\s+)/).map(it => {
          const cleanItem = it.replace(/^[-*]\s+/, "");
          return `<li>${formatInlineMarkdown(escapeHtml(cleanItem))}</li>`;
        });
        return `<ul class="chat-list">${items.join("")}</ul>`;
      }

      // Check for standardized parametric disclaimer banner
      if (p.includes("Memory not found") && p.includes("closest memories are below")) {
        return `<div class="chat-disclaimer-banner">${formatInlineMarkdown(escapeHtml(p))}</div>`;
      }

      return `<p class="chat-paragraph">${formatInlineMarkdown(escapeHtml(p))}</p>`;
    }).join("");
  }

  function appendMessage(role, author, text) {
    const msg = document.createElement("div");
    msg.className = `m3-bubble ${role} message`;
    msg.innerHTML = `
      <div class="bubble-top-meta">
        <span class="author-name">
          <span class="material-symbols-outlined" style="font-size:16px;">${role === 'user' ? 'person' : 'smart_toy'}</span>
          <span>${author}</span>
        </span>
      </div>
      <div class="msg-body">${formatMessageText(text)}</div>
    `;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
  }

  function appendLoadingMessage() {
    const msg = document.createElement("div");
    msg.className = "m3-bubble assistant message";
    msg.innerHTML = `
      <div class="bubble-top-meta">
        <span class="author-name">
          <span class="material-symbols-outlined spin-icon" style="font-size:16px;">sync</span>
          <span>Retrieving memories & reasoning...</span>
        </span>
      </div>
      <div class="msg-body" style="font-style:italic; color:var(--md-sys-color-outline);">Consulting vector embeddings and temporal belief state...</div>
    `;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
  }

  function appendAssistantMessage(data) {
    const msg = document.createElement("div");
    msg.className = "m3-bubble assistant message";
    const msgId = "msg_" + Math.random().toString(36).substr(2, 9);
    msg.id = msgId;

    const eraText = eraSelect.options[eraSelect.selectedIndex].text;
    let evidenceHtml = "";

    if (data.evidence_used && data.evidence_used.length > 0) {
      const isDirect = data.has_sufficient_evidence;
      const titleLabel = isDirect
        ? `📚 Grounded Source Evidence (${data.evidence_used.length} sources)`
        : `📚 Closest Memories (${data.evidence_used.length} retrieved candidates)`;
      const subLabel = isDirect
        ? ''
        : `<span style="font-size:11px; color:var(--text-muted); font-weight:400; margin-left:6px;">· Consulted via fine-tuned weights</span>`;

      evidenceHtml = `
        <div class="m3-evidence-card">
          <div class="evidence-title-row">
            <span>${titleLabel}</span>
            ${subLabel}
          </div>
          ${data.evidence_used.map(ev => `
            <div class="evidence-citation-line" style="display:flex; align-items:center; justify-content:space-between; gap:8px; margin: 4px 0;">
              <span>• ${escapeHtml(ev.citation)}</span>
              ${ev.audio_clip_url ? `
                <button class="hear-clip-btn" onclick="playAudio('${escapeAttr(ev.audio_clip_url)}', this)" title="Listen to original vlog recording">
                  <span class="material-symbols-outlined" style="font-size:13px;">headphones</span>
                  <span>Hear Clip</span>
                </button>
              ` : ''}
            </div>
          `).join("")}
        </div>
      `;
    } else {
      evidenceHtml = `
        <div class="m3-evidence-card" style="border-left-color: var(--md-sys-color-primary);">
          <div class="evidence-title-row" style="color: var(--color-accent);">
            <span>💡 Parametric Corpus Knowledge: Direct memory not found</span>
          </div>
          <div class="evidence-citation-line">Answered authoritatively from autobiographical weights fine-tuned on Sergiu's full video log archive.</div>
        </div>
      `;
    }

    const emotion = data.recommended_delivery ? data.recommended_delivery.emotion : 'reflective';
    let voiceControlsHtml = "";
    if (data.audio_url) {
      voiceControlsHtml = `
        <div style="margin-top:10px; display:flex; align-items:center; gap:8px;">
          <button class="m3-btn m3-btn-tonal audio-trigger-btn" onclick="playAudio('${escapeAttr(data.audio_url)}', this)">
            <span class="material-symbols-outlined" style="font-size:16px;">volume_up</span>
            <span>Play Voice (${escapeHtml(emotion)})</span>
          </button>
        </div>
      `;
    } else {
      voiceControlsHtml = `
        <div style="margin-top:10px; display:flex; align-items:center; gap:8px;">
          <button class="m3-btn m3-btn-outlined" id="synthBtn_${msgId}" onclick="synthesizeMessageVoice('${msgId}', ${escapeAttr(JSON.stringify(data.text))}, '${escapeAttr(emotion)}')">
            <span class="material-symbols-outlined" style="font-size:16px;">graphic_eq</span>
            <span>Synthesize Voice (${escapeHtml(emotion)})</span>
          </button>
          <div id="audioSlot_${msgId}"></div>
        </div>
      `;
    }

    msg.innerHTML = `
      <div class="bubble-top-meta">
        <span class="author-name">
          <span class="material-symbols-outlined" style="font-size:16px;">smart_toy</span>
          <span>Sergiu</span>
        </span>
        <span class="m3-era-chip">${escapeHtml(eraText)}</span>
      </div>
      <div class="msg-body">
        <div class="response-text-flow">
          ${formatMessageText(data.text)}
        </div>
        ${evidenceHtml}
        ${voiceControlsHtml}
      </div>
    `;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // On-demand Voice Synthesis for message bubble
  window.synthesizeMessageVoice = async function(msgId, text, emotion) {
    const btn = document.getElementById(`synthBtn_${msgId}`);
    const slot = document.getElementById(`audioSlot_${msgId}`);
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `<span class="material-symbols-outlined spin-icon" style="font-size:16px;">sync</span><span>Synthesizing...</span>`;
    }

    try {
      const res = await fetch("/api/tts/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          emotion: emotion || "reflective",
          speech_rate: 1.0
        })
      });
      if (!res.ok) throw new Error("Synthesis failed");
      const data = await res.json();
      if (btn) btn.style.display = "none";
      if (slot && data.audio_url) {
        slot.innerHTML = `
          <button class="m3-btn m3-btn-tonal audio-trigger-btn" onclick="playAudio('${escapeAttr(data.audio_url)}', this)">
            <span class="material-symbols-outlined" style="font-size:16px;">volume_up</span>
            <span>Play Voice (${escapeHtml(emotion)})</span>
          </button>
        `;
        const playBtn = slot.querySelector("button");
        playAudio(data.audio_url, playBtn);
      }
    } catch (e) {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-outlined" style="font-size:16px;">error</span><span>Error: ${e.message}</span>`;
      }
    }
  };

  // Memory Vault & M3 Filter Chips
  const memoryGrid = document.getElementById("memoryGrid");
  const memoryStatusFilter = document.getElementById("memoryStatusFilter");
  const memorySearchInput = document.getElementById("memorySearchInput");
  const filterChips = document.querySelectorAll(".m3-filter-chip");

  filterChips.forEach(chip => {
    chip.addEventListener("click", () => {
      filterChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const val = chip.getAttribute("data-filter") || "";
      if (memoryStatusFilter) memoryStatusFilter.value = val;
      loadMemories();
    });
  });

  async function loadMemories() {
    try {
      const status = memoryStatusFilter ? memoryStatusFilter.value : "";
      const url = status ? `/api/memories?status=${encodeURIComponent(status)}` : `/api/memories`;
      const res = await fetch(url);
      const memories = await res.json();
      renderMemories(memories);
    } catch (err) {
      memoryGrid.innerHTML = `<div class="m3-card" style="grid-column: 1/-1; color:var(--md-sys-color-error);">Error loading memories: ${err.message}</div>`;
    }
  }

  function renderMemories(memories) {
    // Populate in-memory cache to guarantee safe event dispatching without inline unescaped strings
    window._memoriesCache = {};
    memories.forEach(m => {
      window._memoriesCache[m.id] = m;
    });

    const filterText = (memorySearchInput.value || "").toLowerCase();
    const filtered = memories.filter(m => 
      !filterText || m.content.toLowerCase().includes(filterText) || (m.summary && m.summary.toLowerCase().includes(filterText))
    );

    if (filtered.length === 0) {
      memoryGrid.innerHTML = `<div class="m3-card" style="grid-column: 1/-1; text-align:center; padding:36px; color:var(--md-sys-color-on-surface-variant);">No memories match the current filter.</div>`;
      return;
    }

    memoryGrid.innerHTML = filtered.map(m => {
      let badgeClass = "m3-badge-unverified";
      const isVerified = (m.verification_status === "personal_verified");
      if (isVerified) badgeClass = "m3-badge-verified";
      else if (m.verification_status === "synthetic_fixture") badgeClass = "m3-badge-synthetic";

      const hasAudioClip = m.source_id && (m.start_seconds !== null && m.start_seconds !== undefined) && (m.end_seconds !== null && m.end_seconds !== undefined);
      const clipUrl = hasAudioClip ? `/api/media/${encodeURIComponent(m.source_id)}/clip?start=${m.start_seconds}&end=${m.end_seconds}` : null;

      return `
        <div class="m3-card memory-card" id="mem_card_${escapeAttr(m.id)}">
          <div class="m3-card-header">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="m3-badge ${badgeClass}">${escapeHtml(m.verification_status)}</span>
              ${m.category ? `<span class="memory-category-tag">${escapeHtml(m.category)}</span>` : ''}
            </div>
            <span style="font-size:12px; color:var(--md-sys-color-on-surface-variant); font-family:var(--md-sys-typescale-mono);">${escapeHtml(m.date_occurred || 'Date Unknown')}</span>
          </div>
          <div class="m3-card-body">
            <p style="white-space: pre-wrap;">${escapeHtml(m.content)}</p>
            ${m.summary && m.summary !== m.content ? `<p style="margin-top:8px; font-size:12px; color:var(--md-sys-color-on-surface-variant); font-style:italic;">${escapeHtml(m.summary)}</p>` : ''}
          </div>
          <div class="m3-card-footer">
            <div style="display:flex; align-items:center; gap:10px;">
              <span style="font-size:12px; color:var(--md-sys-color-outline);">Tone: <strong>${escapeHtml(m.emotional_tone || 'neutral')}</strong></span>
              ${clipUrl ? `
                <button class="hear-clip-btn" onclick="playAudio('${escapeAttr(clipUrl)}', this)" title="Listen to original vlog sound clip">
                  <span class="material-symbols-outlined" style="font-size:13px;">headphones</span>
                  <span>Listen Clip</span>
                </button>
              ` : ''}
            </div>
            <div class="card-action-buttons" style="display:flex; align-items:center; gap:8px;">
              ${!isVerified ? `
                <button class="m3-btn m3-btn-filled m3-btn-verify-direct" onclick="directVerifyMemory('${escapeAttr(m.id)}', this)" title="Direct one-click approval & verification">
                  <span class="material-symbols-outlined" style="font-size:16px;">verified</span>
                  <span>Verify</span>
                </button>
              ` : `
                <span class="verified-check-indicator" title="Personal Verified">
                  <span class="material-symbols-outlined" style="font-size:18px; color:var(--md-sys-color-success);">check_circle</span>
                </span>
              `}
              <button class="m3-btn m3-btn-tonal" onclick="openCorrectionModal('${escapeAttr(m.id)}')" title="Edit text or add notes">
                <span class="material-symbols-outlined" style="font-size:16px;">edit_note</span>
                <span>Edit</span>
              </button>
            </div>
          </div>
        </div>
      `;
    }).join("");
  }

  if (memoryStatusFilter) memoryStatusFilter.addEventListener("change", loadMemories);
  if (memorySearchInput) memorySearchInput.addEventListener("input", loadMemories);

  // Correction Modal Dialog
  const correctionModal = document.getElementById("correctionModal");
  const modalMemoryId = document.getElementById("modalMemoryId");
  const modalMemoryContent = document.getElementById("modalMemoryContent");
  const modalCorrectionReason = document.getElementById("modalCorrectionReason");
  const cancelCorrectionBtn = document.getElementById("cancelCorrectionBtn");
  const saveCorrectionBtn = document.getElementById("saveCorrectionBtn");

  // Open modal using ID lookup from in-memory cache
  window.openCorrectionModal = function(id) {
    const mem = window._memoriesCache[id];
    if (!mem) {
      console.warn("Memory item not found in cache for id:", id);
      return;
    }
    modalMemoryId.value = mem.id;
    modalMemoryContent.value = mem.content || "";
    modalCorrectionReason.value = "";
    correctionModal.style.display = "flex";
  };

  // Direct one-click verification action
  window.directVerifyMemory = async function(id, btn) {
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `<span class="material-symbols-outlined spin-icon" style="font-size:16px;">sync</span><span>Verifying...</span>`;
    }

    try {
      const res = await fetch(`/api/memories/${encodeURIComponent(id)}/verify`, {
        method: "POST"
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Verification failed");
      }

      const data = await res.json();
      if (data.memory && window._memoriesCache[id]) {
        window._memoriesCache[id] = data.memory;
      }
      await loadMemories();
      updateSidebarStats();
    } catch (err) {
      alert(`Verification error: ${err.message}`);
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-outlined" style="font-size:16px;">verified</span><span>Verify</span>`;
      }
    }
  };

  cancelCorrectionBtn.addEventListener("click", () => {
    correctionModal.style.display = "none";
  });

  // Close modal when clicking on scrim outside dialog card
  correctionModal.addEventListener("click", (e) => {
    if (e.target === correctionModal) {
      correctionModal.style.display = "none";
    }
  });

  saveCorrectionBtn.addEventListener("click", async () => {
    const memId = modalMemoryId.value;
    const newContent = modalMemoryContent.value.trim();
    const reason = modalCorrectionReason.value.trim() || "User manual clarification";

    if (!newContent) return;

    saveCorrectionBtn.disabled = true;
    saveCorrectionBtn.innerHTML = `<span class="material-symbols-outlined spin-icon" style="font-size:16px;">sync</span><span>Saving...</span>`;

    try {
      const res = await fetch("/api/memories/correct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          memory_id: memId,
          corrected_content: newContent,
          correction_reason: reason
        })
      });

      if (!res.ok) throw new Error("Correction failed");
      correctionModal.style.display = "none";
      loadMemories();
      updateSidebarStats();
    } catch (err) {
      alert(`Error saving correction: ${err.message}`);
    } finally {
      saveCorrectionBtn.disabled = false;
      saveCorrectionBtn.innerHTML = `<span>Save & Verify</span>`;
    }
  });

  // Beliefs Timeline
  const beliefsTimeline = document.getElementById("beliefsTimeline");
  async function loadBeliefs() {
    try {
      const res = await fetch("/api/beliefs");
      const beliefs = await res.json();

      if (beliefs.length === 0) {
        beliefsTimeline.innerHTML = `<div class="m3-card" style="text-align:center; padding:36px; color:var(--md-sys-color-on-surface-variant);">No beliefs recorded yet.</div>`;
        return;
      }

      beliefsTimeline.innerHTML = beliefs.map(b => {
        const itemClass = b.active ? "active" : "superseded";
        const statusBadge = b.active 
          ? `<span class="m3-badge m3-badge-verified">Active Core Belief</span>` 
          : `<span class="m3-badge m3-badge-synthetic">Superseded Historical Belief</span>`;

        return `
          <div class="m3-timeline-item ${itemClass}">
            <div class="timeline-header-row">
              <span class="timeline-topic-title">${escapeHtml(b.topic)}</span>
              <span class="timeline-era-label">${escapeHtml(b.era_label)}</span>
            </div>
            <div style="font-size:15px; font-style:italic; color:var(--md-sys-color-on-surface); margin:10px 0;">
              "${escapeHtml(b.belief_statement)}"
            </div>
            ${b.reason_for_change ? `
              <div style="font-size:13px; color:var(--md-sys-color-on-surface-variant); margin-top:10px; padding-left:14px; border-left:2px solid var(--md-sys-color-outline-variant);">
                <strong>Evolution Context:</strong> ${escapeHtml(b.reason_for_change)}
              </div>
            ` : ''}
            ${(b.supporting_evidence && b.supporting_evidence.length > 0) ? `
              <div style="margin-top:12px; font-size:12px; background:rgba(0,0,0,0.15); border-radius:8px; padding:10px;">
                <div style="font-weight:600; color:var(--md-sys-color-primary); margin-bottom:6px;">
                  <span class="material-symbols-rounded" style="font-size:14px; vertical-align:middle;">verified</span> Grounded Supporting Evidence (${b.supporting_evidence.length})
                </div>
                ${b.supporting_evidence.map(ev => `
                  <div style="margin-bottom:6px; color:var(--md-sys-color-on-surface-variant); padding-left:8px; border-left:2px solid var(--md-sys-color-primary);">
                    <span style="font-weight:500; color:var(--md-sys-color-outline);">${escapeHtml(ev.timestamp_str || 'Archive')}${ev.start_seconds !== undefined && ev.start_seconds !== null ? ` (${ev.start_seconds}s-${ev.end_seconds}s)` : ''}</span>: "${escapeHtml(ev.quote)}"
                  </div>
                `).join('')}
              </div>
            ` : ''}
            ${(b.counter_arguments && b.counter_arguments.length > 0) ? `
              <div style="margin-top:10px; font-size:12px; background:rgba(255,180,168,0.08); border-radius:8px; padding:10px;">
                <div style="font-weight:600; color:var(--md-sys-color-tertiary); margin-bottom:6px;">
                  <span class="material-symbols-rounded" style="font-size:14px; vertical-align:middle;">balance</span> Dialectical Counterarguments & Nuances (${b.counter_arguments.length})
                </div>
                ${b.counter_arguments.map(ca => `
                  <div style="margin-bottom:6px; color:var(--md-sys-color-on-surface-variant); padding-left:8px; border-left:2px solid var(--md-sys-color-tertiary);">
                    <span style="font-weight:500; color:var(--md-sys-color-outline);">${escapeHtml(ca.timestamp_str || 'Observation')}${ca.start_seconds !== undefined && ca.start_seconds !== null ? ` (${ca.start_seconds}s-${ca.end_seconds}s)` : ''}</span>: "${escapeHtml(ca.quote)}"
                  </div>
                `).join('')}
              </div>
            ` : ''}
            <div style="margin-top: 14px;">${statusBadge}</div>
          </div>
        `;
      }).join("");
    } catch (err) {
      beliefsTimeline.innerHTML = `<div class="m3-card" style="color:var(--md-sys-color-error);">Error loading beliefs: ${err.message}</div>`;
    }
  }

  // Media Ingestion
  const mediaTableBody = document.getElementById("mediaTableBody");
  const runSyntheticIngestionBtn = document.getElementById("runSyntheticIngestionBtn");
  const ingestionOutput = document.getElementById("ingestionOutput");
  const vlogPathInput = document.getElementById("vlogPathInput");
  const saveVlogPathBtn = document.getElementById("saveVlogPathBtn");

  if (saveVlogPathBtn) {
    saveVlogPathBtn.addEventListener("click", () => {
      const p = vlogPathInput.value.trim();
      if (p) {
        alert(`Vlog path configured: ${p}\nThe pipeline is ready to ingest recordings once placed in this location.`);
      } else {
        alert("Please enter a valid directory path.");
      }
    });
  }

  async function loadMediaInventory() {
    try {
      const res = await fetch("/api/media");
      const media = await res.json();

      if (media.length === 0) {
        mediaTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding:28px; color:var(--md-sys-color-on-surface-variant);">No media files ingested yet. Run synthetic test below.</td></tr>`;
        return;
      }

      mediaTableBody.innerHTML = media.map(m => `
        <tr>
          <td><strong>${escapeHtml(m.file_path.split('/').pop())}</strong></td>
          <td>${escapeHtml(m.mime_type || 'unknown')}</td>
          <td>${m.duration_seconds ? m.duration_seconds.toFixed(1) + 's' : 'N/A'}</td>
          <td><span class="m3-badge m3-badge-verified">${escapeHtml(m.status)}</span></td>
          <td style="font-family:var(--md-sys-typescale-mono); font-size:11px; color:var(--md-sys-color-secondary);">${m.file_hash.substring(0, 16)}...</td>
        </tr>
      `).join("");
    } catch (err) {
      console.error(err);
    }
  }

  runSyntheticIngestionBtn.addEventListener("click", async () => {
    ingestionOutput.innerText = "Running synthetic media ingestion test...";
    try {
      const res = await fetch("/api/ingestion/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_synthetic_fixture: true })
      });
      const data = await res.json();
      ingestionOutput.innerText = JSON.stringify(data, null, 2);
      loadMediaInventory();
      updateSidebarStats();
    } catch (err) {
      ingestionOutput.innerText = `Error: ${err.message}`;
    }
  });

  // Voice Synthesizer Studio
  const ttsTextInput = document.getElementById("ttsTextInput");
  const ttsEmotionSelect = document.getElementById("ttsEmotionSelect");
  const ttsRateRange = document.getElementById("ttsRateRange");
  const rateVal = document.getElementById("rateVal");
  const synthesizeBtn = document.getElementById("synthesizeBtn");
  const ttsAudioPlayer = document.getElementById("ttsAudioPlayer");

  if (ttsAudioPlayer) {
    ttsAudioPlayer.addEventListener("play", () => {
      audioController.stop();
    });
  }

  ttsRateRange.addEventListener("input", () => {
    rateVal.innerText = `${ttsRateRange.value}x`;
  });

  let lastSynthesizedAudioFilename = null;

  synthesizeBtn.addEventListener("click", async () => {
    const text = ttsTextInput.value.trim();
    if (!text) return;

    synthesizeBtn.innerHTML = `<span class="material-symbols-outlined spin-icon" style="font-size:18px;">sync</span> <span>Synthesizing...</span>`;
    try {
      const res = await fetch("/api/tts/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          emotion: ttsEmotionSelect.value,
          speech_rate: parseFloat(ttsRateRange.value)
        })
      });
      const data = await res.json();
      synthesizeBtn.innerHTML = `<span class="material-symbols-outlined" style="font-size:18px;">graphic_eq</span> <span>Synthesize Expressive Audio</span>`;

      if (data.audio_url) {
        lastSynthesizedAudioFilename = data.filename;
        ttsAudioPlayer.src = data.audio_url;
        ttsAudioPlayer.style.display = "block";
        audioController.stop();
        ttsAudioPlayer.play();
      }
    } catch (err) {
      synthesizeBtn.innerHTML = `<span class="material-symbols-outlined" style="font-size:18px;">graphic_eq</span> <span>Synthesize Expressive Audio</span>`;
      alert(`Synthesis failed: ${err.message}`);
    }
  });

  // Avatar Studio
  const renderAvatarBtn = document.getElementById("renderAvatarBtn");
  const avatarPlaceholder = document.getElementById("avatarPlaceholder");
  const avatarVideoPlayer = document.getElementById("avatarVideoPlayer");
  const avatarStatusBox = document.getElementById("avatarStatusBox");

  renderAvatarBtn.addEventListener("click", async () => {
    if (!lastSynthesizedAudioFilename) {
      synthesizeBtn.click();
      avatarStatusBox.innerText = "Synthesizing driving audio first...";
      await new Promise(r => setTimeout(r, 1500));
    }

    if (!lastSynthesizedAudioFilename) {
      avatarStatusBox.innerText = "Please synthesize audio first.";
      return;
    }

    avatarStatusBox.innerText = "Enqueuing video avatar render job...";
    const formData = new FormData();
    formData.append("audio_filename", lastSynthesizedAudioFilename);

    try {
      const res = await fetch("/api/avatar/render", {
        method: "POST",
        body: formData
      });
      const job = await res.json();
      pollAvatarJob(job.job_id);
    } catch (err) {
      avatarStatusBox.innerText = `Render failed: ${err.message}`;
    }
  });

  function pollAvatarJob(jobId) {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/avatar/status/${jobId}`);
        const job = await res.json();
        avatarStatusBox.innerText = `Status: ${job.status} (Progress: ${Math.round(job.progress * 100)}%)`;

        if (job.status === "completed") {
          clearInterval(interval);
          avatarStatusBox.innerText = "Render complete! Loading MP4 video preview.";
          avatarPlaceholder.style.display = "none";
          avatarVideoPlayer.src = `/api/avatar/video/${job.output_video_path.split('/').pop()}`;
          avatarVideoPlayer.style.display = "block";
          avatarVideoPlayer.play();
        } else if (job.status === "failed") {
          clearInterval(interval);
          avatarStatusBox.innerText = `Avatar render failed: ${job.error_message}`;
        }
      } catch (err) {
        clearInterval(interval);
        avatarStatusBox.innerText = `Status check error: ${err.message}`;
      }
    }, 400);
  }

  // Backup Manager
  const createBackupBtn = document.getElementById("createBackupBtn");
  const backupOutput = document.getElementById("backupOutput");

  createBackupBtn.addEventListener("click", async () => {
    backupOutput.innerText = "Creating snapshot archive...";
    try {
      const res = await fetch("/api/backup/create", { method: "POST" });
      const data = await res.json();
      backupOutput.innerText = `Archive created:\n${JSON.stringify(data, null, 2)}`;
    } catch (err) {
      backupOutput.innerText = `Backup error: ${err.message}`;
    }
  });

  // Sidebar Stats
  async function updateSidebarStats() {
    try {
      const res = await fetch("/api/archive/stats");
      const stats = await res.json();
      const countMem = document.getElementById("statMemoriesCount");
      const countVer = document.getElementById("statVerifiedCount");
      if (countMem) countMem.innerText = stats.total_memories;
      if (countVer) countVer.innerText = stats.verified_memories;
    } catch (e) {}
  }

  // Helpers
  function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function escapeAttr(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  updateSidebarStats();
});
