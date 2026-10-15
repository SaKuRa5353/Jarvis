"""
Все константы проекта в одном месте.
Меняй здесь — не трогая код модулей.
"""

from pathlib import Path

# ──────────────────────── АУДИО ────────────────────────

SAMPLE_RATE = 16000
DEVICE = 4                          # USB Audio CODEC

# ──────────────────────── МОДЕЛИ ────────────────────────

WAKE_MODEL = "models/hey_jarvis_v0.1.onnx"
VAD_MODEL = "models/silero_vad.onnx"
VOSK_MODEL = "models/vosk"

# ──────────────────────── WAKE WORD ────────────────────────

WAKE_THRESHOLD = 0.42
WAKE_COOLDOWN = 2.0
WAKE_BLOCK_SIZE = 1280              # 80 мс при 16 kHz

# ──────────────────────── VAD ────────────────────────

VAD_FRAME = 512                     # 32 мс при 16 kHz
VAD_SPEECH_THRESHOLD = 0.5
VAD_SILENCE_THRESHOLD = 0.4
VAD_MIN_SILENCE_SEC = 0.8
VAD_MAX_RECORD_SEC = 6
VAD_MIN_SPEECH_SEC = 0.3

# ──────────────────────── TTS ────────────────────────

TTS_WAV = Path("/tmp/jarvis_tts.wav")

# ──────────────────────── LLM ────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen2.5:3b"
LLM_MAX_TOKENS = 30
LLM_TEMPERATURE = 0.3
HISTORY_FILE = Path("data/history.json")
MAX_HISTORY_MESSAGES = 6
LLM_SYSTEM_PROMPT = (
    "Ты — Джарвис, голосовой ассистент. "
    "Отвечай ОДНИМ коротким предложением, максимум 5 слов. "
    "Без приветствий, без пояснений, сразу по делу. "
    "По-русски."
)
