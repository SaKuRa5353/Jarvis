"""
Все константы проекта в одном месте.
Меняй здесь — не трогая код модулей.
"""

import os
from pathlib import Path

# .env рядом с проектом: JARVIS_DEVICE, JARVIS_LOG_LEVEL, JARVIS_LOG_DIR.
# Файл в .gitignore — значения у каждого свои. Создать: python main.py --setup
try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv есть в requirements.txt, но вдруг
    def load_dotenv(*_a, **_kw) -> bool:
        return False

load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Целое из окружения; при мусоре — default с предупреждением."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        print(f"[config] {name}={raw!r} — не число, беру {default}")
        return default

# ──────────────────────── АУДИО ────────────────────────
# Единый микрофон на весь цикл ассистента.
# Один InputStream -> буфер -> потребители (wakeword / VAD).

SAMPLE_RATE = 16000
# Захардкод был здесь. Теперь: .env -> JARVIS_DEVICE, fallback 4.
# Перенастроить микрофон:  python main.py --setup
DEVICE = _env_int("JARVIS_DEVICE", 4)
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

# ──────────────────────── WAKEWORD VAD-GATE ────────────────────────
# В тишине работает только дешёвый Silero VAD; тяжёлый Vosk включается
# на речь. Pre-roll — сколько тишины до начала речи отдать распознавателю,
# чтобы не обрезать первое слово.

WW_PREROLL_SEC = 0.4
WW_MAX_UTTERANCE_SEC = 8.0          # длиннее — принудительный FinalResult

# ──────────────────────── TTS ────────────────────────

SILERO_MODEL = "v5_ru"
SILERO_SPEAKER = "eugene"
SILERO_SAMPLE_RATE = 48000

# Кэш синтеза: hash(текст) → WAV в data/tts_cache. Повторный ответ
# играет мгновенно, без синтеза. Ограничен числом файлов (старые удаляются).
TTS_CACHE_DIR = Path("data/tts_cache")
TTS_CACHE_MAX_FILES = 200

# Стриминг ответа LLM: с какого объёма готового текста начинать озвучку
# по предложениям, не дожидаясь конца генерации. Ответы короче порога
# озвучиваются целиком после (мусорные при этом заменяются готовой фразой).
TTS_STREAM_MIN_CHARS = 30

# ──────────────────────── СЕНСОРЫ ────────────────────────

SENSORS_CMD = "sensors"           # lm_sensors
SENSORS_TIMEOUT_SEC = 3

# ──────────────────────── ОКНА (kdotool) ────────────────────────
# Только чтение: какое окно активно. Управление окнами — не здесь.

KDTOOL_PATH = "kdotool"
WINDOW_CONTEXT_ENABLED = True     # добавлять активное окно в промпт LLM
WINDOW_CONTEXT_TIMEOUT = 2.0
WINDOW_CONTEXT_CACHE_SEC = 2.0

# ──────────────────────── DIALOG ────────────────────────
# После первого обращения (wake word + команда) ассистент остаётся
# «на связи» и слушает следующие команды без повторного «Джарвис».
# Если тишина длится дольше этого времени — режим диалога завершается.
CONVERSATION_TIMEOUT_SEC = 15.0

# ──────────────────────── LLM ────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen3-jarvis:latest"
# Бюджет текстового ответа. ВНИМАНИЕ: до quick wins эта константа была
# объявлена, но НЕ использовалась — llm.py всегда брал LLM_TOOLS_MAX_TOKENS.
# Теперь лимит = LLM_TOOLS_MAX_TOKENS при включённых tools, иначе этот.
LLM_MAX_TOKENS = 30               # обычный текстовый ответ
LLM_TOOLS_MAX_TOKENS = 128        # tool_call требует больше токенов
LLM_TEMPERATURE = 0.3
LLM_NUM_THREADS = 8
LLM_TIMEOUT_SEC = 30
MAX_HISTORY_MESSAGES = 6    # хранится только в памяти текущей сессии
LLM_TOOLS_HINT = (
    "Если просят что-то открыть, запустить, найти или изменить — "
    "вызови подходящую функцию (tool). Для приветствий, ответов на "
    "вопросы и болтовни функция не нужна — просто ответь текстом."
)
# Thinking/reasoning у Qwen3. Модель qwen3-jarvis собрана с thinking.default=true,
# поэтому Ollama отдаёт текст в поле reasoning/thinking, а content приходит ПУСТЫМ.
# Текстом промпта это не отключается — только полем `think` в запросе.
#   think  -> extra_body={'think': False}   (основной способ)
#   effort -> reasoning_effort='none'
#   none   -> запрос не меняется
# Что работает на вашей Ollama — покажет  python tests/check_llm.py
LLM_THINKING = False
LLM_THINK_MODE = "think"

# Чем заменять пустой/оборванный ответ LLM (ключ готовой фразы).
LLM_EMPTY_FALLBACK = "not_listened"

LLM_SYSTEM_PROMPT = (
    "Ты — Джарвис, голосовой ассистент. "
    "Отвечай по-русски коротко: 1-5 слов, сразу по делу. "
    "Отвечай на все понятные вопросы и команды. "
    "Если запрос бессмысленный или состоит из случайных звуков — "
    "ответь ровно: Не расслышал. "
    "Примеры хороших ответов: "
    "«сколько будет два плюс два» → «четыре»; "
    "«привет» → «Привет! Чем помочь?»; "
    "«какая погода» → «Сейчас проверю погоду»; "
    "«расскажи анекдот» → «Коллега, я робот, у меня нет чувства юмора»; "
    "«дбвщта» → «Не расслышал»."
)


# ──────────────────────── ЛОГИРОВАНИЕ ────────────────────────
# Файл: logs/jarvis.log. Ротация по РАЗМЕРУ (предсказуемо на диске),
# уровень файла JARVIS_FILE_LOG_LEVEL, консоль JARVIS_LOG_LEVEL.

LOG_DIR = Path(os.getenv("JARVIS_LOG_DIR", "logs"))
LOG_FILE_NAME = "jarvis.log"
LOG_MAX_BYTES = 5 * 1024 * 1024     # 5 МБ на файл
LOG_BACKUP_COUNT = 5                # хранить 5 архивов
LOG_LEVEL = os.getenv("JARVIS_LOG_LEVEL", "INFO").upper()
LOG_FILE_LEVEL = os.getenv("JARVIS_FILE_LOG_LEVEL", "INFO").upper()

# ──────────────────────── ЗВУКИ ПОДТВЕРЖДЕНИЯ ────────────────────────
# voice   — готовые речевые фразы из src/phrases/ (текущее поведение)
# earcon  — короткий «динь» 180 мс: быстрее на ~0.5 с и меньше эха в микрофон
# off     — тишина
# Создать earcon-WAV:  python tests/generate_earcons.py

CONFIRM_SOUND = os.getenv("JARVIS_CONFIRM_SOUND", "voice").lower()
EARCON_TAIL_SEC = 0.03              # пауза после короткого звука (эхо-хвост)

# ──────────────────────── ИСТОРИЯ ДИАЛОГА ────────────────────────
# Заготовка для персиста (QW4). Сам персист включается отдельно.

HISTORY_DIR = Path(os.getenv("JARVIS_HISTORY_DIR", "data/history"))
HISTORY_ENABLED = os.getenv("JARVIS_HISTORY_ENABLED", "0") == "1"
HISTORY_RESTORE_TURNS = 6
