import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="VocaBuds All-in-One")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")

# ----------------- BACKEND API -----------------
class TranslationRequest(BaseModel):
    text: str
    from_lang: str
    to_lang: str

@app.post("/api/translate")
def translate(req: TranslationRequest):
    system_prompt = (
        "You are the real-time AI translation core of VocaBuds earbuds. "
        "Translate the input accurately into the requested target language using its native script. "
        "Output ONLY the translated text. No explanations, no quotes, no conversational filler."
    )
    user_prompt = f"Translate from {req.from_lang} to {req.to_lang}:\n{req.text}"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.2}
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=45)
        res.raise_for_status()
        data = res.json()
        translated = data.get("message", {}).get("content", "").strip()
        if not translated:
            raise HTTPException(status_code=500, detail="Empty response from Gemma 3")
        return {"translated": translated, "model": OLLAMA_MODEL}
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama not detected at localhost:11434. Run 'ollama serve' in terminal."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- FRONTEND UI (HTML + CSS + JS) -----------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VocaBuds — Live Voice Translation (Gemma 3)</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: #070709;
      color: #f3f3f5;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 9999px;
      padding: 0.35rem 0.9rem;
      font-size: 0.75rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: #a1a1aa;
      margin-bottom: 1.5rem;
    }
    .badge-dot {
      width: 6px;
      height: 6px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 8px #10b981;
    }
    h1 {
      font-size: 2.8rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      text-align: center;
      background: linear-gradient(180deg, #ffffff 0%, #a1a1aa 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    p.subtitle {
      color: #71717a;
      text-align: center;
      max-width: 520px;
      font-size: 0.95rem;
      margin-bottom: 2.5rem;
      line-height: 1.5;
    }
    .app-card {
      width: 100%;
      max-width: 820px;
      background: rgba(18, 18, 22, 0.75);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 20px;
      padding: 2rem;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
    }
    .lang-controls {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      gap: 1rem;
      align-items: center;
      margin-bottom: 1.8rem;
    }
    .select-wrapper {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }
    .select-label {
      font-size: 0.75rem;
      color: #71717a;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    select {
      background: #131317;
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #fff;
      padding: 0.75rem 1rem;
      border-radius: 10px;
      font-size: 0.9rem;
      outline: none;
      cursor: pointer;
    }
    .swap-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #a1a1aa;
      width: 40px;
      height: 40px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      margin-top: 1.2rem;
      transition: all 0.2s;
    }
    .swap-btn:hover { background: rgba(255, 255, 255, 0.12); color: #fff; }
    
    .mic-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1rem;
      margin: 1.5rem 0 2rem 0;
    }
    .mic-btn {
      width: 80px;
      height: 80px;
      border-radius: 50%;
      background: #ffffff;
      color: #000;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 30px rgba(255, 255, 255, 0.25);
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .mic-btn:hover { transform: scale(1.06); }
    .mic-btn.active {
      background: #ef4444;
      color: #fff;
      animation: pulse 1.6s infinite;
      box-shadow: 0 0 35px rgba(239, 68, 68, 0.6);
    }
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.08); }
      100% { transform: scale(1); }
    }
    .status-text {
      font-size: 0.85rem;
      color: #a1a1aa;
      font-family: monospace;
    }
    
    .waves {
      display: flex;
      gap: 3px;
      height: 24px;
      align-items: center;
    }
    .wave-bar {
      width: 3px;
      height: 4px;
      background: #52525b;
      border-radius: 2px;
      transition: height 0.15s;
    }
    .waves.active .wave-bar {
      background: #ffffff;
      animation: soundwave 1s ease-in-out infinite alternate;
    }
    .waves.active .wave-bar:nth-child(2) { animation-delay: 0.1s; }
    .waves.active .wave-bar:nth-child(3) { animation-delay: 0.2s; }
    .waves.active .wave-bar:nth-child(4) { animation-delay: 0.3s; }
    .waves.active .wave-bar:nth-child(5) { animation-delay: 0.15s; }
    @keyframes soundwave {
      0% { height: 4px; }
      100% { height: 24px; }
    }

    .output-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.2rem;
    }
    .box {
      background: #0d0d11;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 1.2rem;
      min-height: 150px;
      display: flex;
      flex-direction: column;
    }
    .box-header {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #71717a;
      margin-bottom: 0.75rem;
      display: flex;
      justify-content: space-between;
    }
    .box-content {
      flex: 1;
      font-size: 1.05rem;
      line-height: 1.5;
      color: #e4e4e7;
      word-break: break-word;
    }
    .box-content.empty { color: #52525b; font-style: italic; }
    
    .actions {
      display: flex;
      gap: 0.75rem;
      margin-top: 1.2rem;
    }
    .btn-secondary {
      flex: 1;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #fff;
      padding: 0.75rem;
      border-radius: 10px;
      font-size: 0.85rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: background 0.2s;
    }
    .btn-secondary:hover { background: rgba(255, 255, 255, 0.12); }
    .btn-secondary:disabled { opacity: 0.4; cursor: not-allowed; }

    @media (max-width: 640px) {
      .lang-controls { grid-template-columns: 1fr; }
      .swap-btn { margin: 0 auto; }
      .output-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

  <div class="badge">
    <div class="badge-dot"></div>
    Engine: Ollama Gemma 3
  </div>

  <h1>VocaBuds</h1>
  <p class="subtitle">Next-generation real-time neural translation earbuds interface powered by on-device Gemma 3 LLM.</p>

  <div class="app-card">
    <div class="lang-controls">
      <div class="select-wrapper">
        <label class="select-label">Input Language</label>
        <select id="sourceLang"></select>
      </div>
      <button class="swap-btn" id="swapBtn" title="Swap languages">⇄</button>
      <div class="select-wrapper">
        <label class="select-label">Target Translation</label>
        <select id="targetLang"></select>
      </div>
    </div>

    <div class="mic-container">
      <div class="waves" id="waves">
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
      </div>
      <button class="mic-btn" id="micBtn" title="Toggle Recording">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" x2="12" y1="19" y2="22"/>
        </svg>
      </button>
      <div class="status-text" id="status">Ready — Click Mic to Speak</div>
    </div>

    <div class="output-grid">
      <div class="box">
        <div class="box-header">Original Speech</div>
        <div class="box-content empty" id="originalText">Spoken words will appear here...</div>
      </div>
      <div class="box">
        <div class="box-header">
          <span>Gemma 3 Translation</span>
          <span id="targetScriptBadge"></span>
        </div>
        <div class="box-content empty" id="translatedText">Translation will appear here...</div>
      </div>
    </div>

    <div class="actions">
      <button class="btn-secondary" id="speakBtn" disabled>
        🔊 Play Audio (TTS)
      </button>
      <button class="btn-secondary" id="clearBtn">
        🗑 Clear
      </button>
    </div>
  </div>

  <script>
    const LANGUAGES = [
      { code: "en", name: "English", speech: "en-IN" },
      { code: "hi", name: "Hindi (हिन्दी)", speech: "hi-IN" },
      { code: "bho", name: "Bhojpuri (भोजपुरी)", speech: "hi-IN" },
      { code: "mai", name: "Maithili (मैथिली)", speech: "hi-IN" },
      { code: "bn", name: "Bengali (বাংলা)", speech: "bn-IN" },
      { code: "mr", name: "Marathi (मराठी)", speech: "mr-IN" },
      { code: "gu", name: "Gujarati (ગુજરાતી)", speech: "gu-IN" },
      { code: "pa", name: "Punjabi (ਪੰਜਾਬੀ)", speech: "pa-IN" },
      { code: "ta", name: "Tamil (தமிழ்)", speech: "ta-IN" },
      { code: "te", name: "Telugu (తెలుగు)", speech: "te-IN" },
      { code: "kn", name: "Kannada (ಕನ್ನಡ)", speech: "kn-IN" },
      { code: "ml", name: "Malayalam (മലയാളം)", speech: "ml-IN" },
      { code: "es", name: "Spanish (Español)", speech: "es-ES" },
      { code: "fr", name: "French (Français)", speech: "fr-FR" },
      { code: "de", name: "German (Deutsch)", speech: "de-DE" },
      { code: "ja", name: "Japanese (日本語)", speech: "ja-JP" },
      { code: "ko", name: "Korean (한국어)", speech: "ko-KR" },
      { code: "zh", name: "Chinese (中文)", speech: "zh-CN" },
      { code: "ru", name: "Russian (Русский)", speech: "ru-RU" },
      { code: "ar", name: "Arabic (العربية)", speech: "ar-SA" }
    ];

    const sourceSelect = document.getElementById("sourceLang");
    const targetSelect = document.getElementById("targetLang");
    const micBtn = document.getElementById("micBtn");
    const waves = document.getElementById("waves");
    const statusText = document.getElementById("status");
    const originalDiv = document.getElementById("originalText");
    const translatedDiv = document.getElementById("translatedText");
    const speakBtn = document.getElementById("speakBtn");
    const clearBtn = document.getElementById("clearBtn");
    const swapBtn = document.getElementById("swapBtn");

    LANGUAGES.forEach(l => {
      sourceSelect.add(new Option(l.name, l.code));
      targetSelect.add(new Option(l.name, l.code));
    });
    sourceSelect.value = "en";
    targetSelect.value = "hi";

    let isRecording = false;
    let recognition = null;
    let currentSpeech = "";

    // Web Speech Recognition setup
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.onstart = () => {
        isRecording = true;
        micBtn.classList.add("active");
        waves.classList.add("active");
        statusText.textContent = "Listening to speech...";
      };

      recognition.onresult = (event) => {
        let interim = "";
        let final = "";
        for (let i = 0; i < event.results.length; ++i) {
          if (event.results[i].isFinal) final += event.results[i][0].transcript + " ";
          else interim += event.results[i][0].transcript;
        }
        currentSpeech = (final + interim).trim();
        originalDiv.textContent = currentSpeech || "Listening...";
        originalDiv.classList.remove("empty");
      };

      recognition.onerror = (e) => {
        statusText.textContent = "Mic error: " + e.error;
        stopRecording();
      };

      recognition.onend = () => {
        if (isRecording) stopRecording();
      };
    } else {
      statusText.textContent = "Speech recognition not supported in this browser (use Chrome/Edge)";
    }

    micBtn.addEventListener("click", () => {
      if (!recognition) return alert("Speech recognition requires Chrome/Edge browser.");
      if (!isRecording) {
        currentSpeech = "";
        originalDiv.textContent = "Listening...";
        originalDiv.classList.remove("empty");
        const selLang = LANGUAGES.find(l => l.code === sourceSelect.value);
        recognition.lang = selLang ? selLang.speech : "en-IN";
        try { recognition.start(); } catch(err) { console.warn(err); }
      } else {
        stopRecording();
        if (currentSpeech.trim()) {
          requestTranslation(currentSpeech.trim());
        }
      }
    });

    function stopRecording() {
      isRecording = false;
      micBtn.classList.remove("active");
      waves.classList.remove("active");
      try { recognition.stop(); } catch(err) {}
    }

    async function requestTranslation(text) {
      statusText.textContent = "Translating with Gemma 3...";
      translatedDiv.textContent = "Processing with Ollama Gemma 3...";
      translatedDiv.classList.remove("empty");

      const sName = LANGUAGES.find(l => l.code === sourceSelect.value)?.name || sourceSelect.value;
      const tName = LANGUAGES.find(l => l.code === targetSelect.value)?.name || targetSelect.value;

      try {
        const res = await fetch("/api/translate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, from_lang: sName, to_lang: tName })
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Server error");
        }
        const data = await res.json();
        translatedDiv.textContent = data.translated;
        statusText.textContent = "Translation Complete (" + data.model + ")";
        speakBtn.disabled = false;
        playAudio(data.translated);
      } catch (err) {
        translatedDiv.textContent = "Error: " + err.message;
        statusText.textContent = "Translation Failed";
      }
    }

    function playAudio(text) {
      if (!window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      const selLang = LANGUAGES.find(l => l.code === targetSelect.value);
      u.lang = selLang ? selLang.speech : "hi-IN";
      u.onstart = () => { statusText.textContent = "Speaking translation..."; };
      u.onend = () => { statusText.textContent = "Ready"; };
      window.speechSynthesis.speak(u);
    }

    speakBtn.addEventListener("click", () => {
      const text = translatedDiv.textContent;
      if (text && !translatedDiv.classList.contains("empty")) playAudio(text);
    });

    clearBtn.addEventListener("click", () => {
      originalDiv.textContent = "Spoken words will appear here...";
      originalDiv.classList.add("empty");
      translatedDiv.textContent = "Translation will appear here...";
      translatedDiv.classList.add("empty");
      speakBtn.disabled = true;
      statusText.textContent = "Ready";
    });

    swapBtn.addEventListener("click", () => {
      const temp = sourceSelect.value;
      sourceSelect.value = targetSelect.value;
      targetSelect.value = temp;
    });
  </script>
</body>
</html>"""

if __name__ == "__main__":
    print("Starting VocaBuds on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
