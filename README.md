<img width="1299" height="424" alt="cd (1)" src="https://github.com/user-attachments/assets/b25fff4d-043d-4f38-9985-f832ae0d0f6e" />

## Recall.ai - API for desktop recording

If you’re looking for a hosted desktop recording API, consider checking out [Recall.ai](https://www.recall.ai/product/desktop-recording-sdk/?utm_source=github&utm_medium=sponsorship&utm_campaign=sohzm-cheating-daddy), an API that records Zoom, Google Meet, Microsoft Teams, in-person meetings, and more.

This project is sponsored by Recall.ai.

---

> [!NOTE]  
> Use latest MacOS and Windows version, older versions have limited support

> [!NOTE]  
> During testing it wont answer if you ask something, you need to simulate interviewer asking question, which it will answer

A real-time AI assistant that provides contextual help during video calls, interviews, presentations, and meetings using screen capture and audio analysis.

## Features

- **Live AI Assistance**: Real-time help powered by Google Gemini 2.0 Flash Live
- **Screen & Audio Capture**: Analyzes what you see and hear for contextual responses
- **Multiple Profiles**: Interview, Sales Call, Business Meeting, Presentation, Negotiation
- **Transparent Overlay**: Always-on-top window that can be positioned anywhere
- **Click-through Mode**: Make window transparent to clicks when needed
- **Cross-platform**: Works on macOS, Windows, and Linux (kinda, dont use, just for testing rn)

## Setup

1. **Get a Gemini API Key**: Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. **Install Dependencies**: `npm install`
3. **Run the App**: `npm start`

### GigaAM v3 + Codex CLI mode

This mode replaces the Gemini API key with local Russian speech recognition and an authenticated Codex CLI session:

1. Install `ffmpeg`, Python 3.10 or newer (3.10–3.13 recommended for current PyTorch builds), and the
   [Codex CLI](https://developers.openai.com/codex/cli/). Run `codex login` once.
2. Create a dedicated Python environment and install the official GigaAM package:

    ```bash
    python3.11 -m venv ~/.local/share/cheating-daddy/gigaam-venv
    git clone https://github.com/salute-developers/GigaAM.git /tmp/GigaAM
    ~/.local/share/cheating-daddy/gigaam-venv/bin/pip install -e '/tmp/GigaAM[torch]'
    ```

3. Select **GigaAM + Codex CLI** on the home screen. Set **Python executable** to the virtual environment's
   Python (for example `~/.local/share/cheating-daddy/gigaam-venv/bin/python`). The default Codex model is
   `gpt-5.6-sol` with low reasoning effort and the standard service tier.
4. To provide a fixed reference document, place a UTF-8 `lab.md` file in the configured **Working directory**.
   It is loaded once when the session starts; a small local section search supplies only the relevant excerpts to each request.

The default `v3_e2e_rnnt` model adds punctuation and text normalization. Its weights are downloaded on the
first start. Audio and transcription stay local; only the resulting text (and manually submitted screenshots)
is passed to Codex CLI. A persistent Codex app-server keeps the session warm and streams answer deltas into the UI;
its conversation thread is ephemeral and uses a read-only sandbox. User MCP servers, shell commands, web search, apps,
and plugins are disabled, so reference material must be supplied through `lab.md`. MathJax-viewer formula links in the
document are decoded to LaTeX locally.

## Usage

1. Enter your Gemini API key in the main window
2. Choose your profile and language in settings
3. Click "Start Session" to begin
4. Position the window using keyboard shortcuts
5. The AI will provide real-time assistance based on your screen and what interview asks

## Keyboard Shortcuts

- **Window Movement**: `Ctrl/Cmd + Arrow Keys` - Move window
- **Click-through**: `Ctrl/Cmd + M` - Toggle mouse events
- **Close/Back**: `Ctrl/Cmd + \` - Close window or go back
- **Send Message**: `Enter` - Send text to AI

## Audio Capture

- **macOS**: [SystemAudioDump](https://github.com/Mohammed-Yasin-Mulla/Sound) for system audio
- **Windows**: Loopback audio capture
- **Linux**: PipeWire/PulseAudio monitor capture through `parec`. If the default monitor is silent, set the active sink monitor in
  **Customize → Audio Input → Linux System Audio Monitor** (find names with `pactl list short sources`).

## Requirements

- Electron-compatible OS (macOS, Windows, Linux)
- Gemini API key, or a configured GigaAM + Codex CLI environment
- Screen recording permissions
- Microphone/audio permissions
