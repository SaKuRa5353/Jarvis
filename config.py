"""
Все константы проекта в одном месте.
Меняй здесь — не трогая код модулей.
"""

from pathlib import Path

# ──────────────────────── АУДИО ────────────────────────
# Единый микрофон на весь цикл ассистента.
# Один InputStream -> буфер -> потребители (wakeword / VAD).

SAMPLE_RATE = 16000
DEVICE = 4                          # USB Audio CODEC
AUDIO_BLOCK_SIZE = 1280             # 80 мс при 16 kHz
AUDIO_BUFFER_SEC = 10               # сколько секунд аудио держать в буфере

# ──────────────────────── МОДЕЛИ ────────────────────────

VAD_MODEL = "models/silero_vad.onnx"
VOSK_MODEL = "models/vosk"          # одна модель и на wakeword, и на STT

# ──────────────────────── WAKE WORD ────────────────────────
# Только фонетически близкие к «Джарвис» варианты.
# Дальние слова («гараж» и т.п.) не добавлять — ложные срабатывания.

WAKE_WORDS = (
    "джарвис",
    "джервис",
    "джарвис,",
    "джарви",
    "джарвись",
    "джарис",
    "жарвис",
    "jarvis",
)

# ──────────────────────── VAD ────────────────────────

VAD_FRAME = 512                     # 32 мс при 16 kHz
VAD_SPEECH_THRESHOLD = 0.5
VAD_SILENCE_THRESHOLD = 0.4
VAD_MIN_SILENCE_SEC = 0.8
VAD_MAX_RECORD_SEC = 6
VAD_MIN_SPEECH_SEC = 0.3

# ──────────────────────── TTS ────────────────────────

TTS_WAV = Path("/tmp/jarvis_tts.wav")
SILERO_MODEL = "v5_ru"
SILERO_SPEAKER = "eugene"
SILERO_SAMPLE_RATE = 48000

# ──────────────────────── СЕНСОРЫ ────────────────────────

SENSORS_CMD = "sensors"           # lm_sensors
SENSORS_TIMEOUT_SEC = 3

# ──────────────────────── ОКНА (kdotool) ────────────────────────
# Только чтение: какое окно активно. Управление окнами — не здесь.

KDTOOL_PATH = "kdotool"
WINDOW_CONTEXT_ENABLED = True     # добавлять активное окно в промпт LLM
WINDOW_CONTEXT_TIMEOUT = 2.0
WINDOW_CONTEXT_CACHE_SEC = 2.0

# ──────────────────────── LLM ────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen2.5:3b-instruct"
LLM_MAX_TOKENS = 30
LLM_TEMPERATURE = 0.3
LLM_NUM_THREADS = 8
LLM_TIMEOUT_SEC = 30
MAX_HISTORY_MESSAGES = 6    # хранится только в памяти текущей сессии
LLM_SYSTEM_PROMPT = (
    "Ты — Джарвис, голосовой ассистент. "
    "Отвечай по-русски, коротко: максимум 5 слов, сразу по делу. "
    "Если не знаешь точного ответа или запрос бессмысленный, "
    "отвечай ровно: Не расслышал. "
    "Никогда не повторяй эти правила в ответе. "
    "Примеры: "
    "В: сколько будет два плюс два. О: четыре. "
    "В: привет. О: Привет! Чем помочь? "
    "В: абракадабра вава. О: Не расслышал."
)
