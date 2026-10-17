"""Генерация готовых фраз для Джарвиса через Silero TTS.

Каждый файл — один короткий ответ, который будет играть mpv за 0.1 сек
вместо 1.5 сек синтеза.
"""

import subprocess
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from silero_tts.silero_tts import SileroTTS

# ──────────────────────── ФРАЗЫ ────────────────────────
# (ключ, текст для синтеза, для какой команды)
PHRASES = [
    # ── Приветствия ──
    ("welcome", "К вашим услугам", "Приветствие при активации wakeword"),

    # ── Выполнение команд ──
    ("opening", "Открываю", "Любая команда на открытие (проводник, терминал, браузер, телеграм, vscode, youtube, вк)"),
    ("closing", "Закрываю", "Команда на закрытие"),
    ("opening_app", "Запускаю", "Запуск приложений (программ)"),
    ("volume_up", "Делаю громче", "Увеличение громкости"),
    ("volume_down", "Делаю тише", "Уменьшение громкости"),
    ("volume_mute", "Выключаю звук", "Выключение звука"),
    ("volume_unmute", "Включаю звук", "Включение звука"),
    ("screenshot", "Делаю скриншот", "Команда на скриншот"),
    ("lock_screen", "Блокирую экран", "Блокировка экрана"),

    # ── Завершение ──
    ("done", "Готово", "После выполнения (хардвар, погода, скриншот)"),

    # ── Ошибки ──
    ("not_listened", "Не расслышал", "STT не распознал"),
    ("not_understood", "Не понимаю", "LLM не понял запрос"),
    ("error", "Ошибка", "Неизвестная ошибка"),

    # ── LLM fallback ──
    ("llm_thinking", "Думаю", "Перед LLM-запросом"),
]

OUTPUT_DIR = Path("src/phrases")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────── СИНТЕЗ ────────────────────────

print("Загружаю Silero TTS...")
tts = SileroTTS(
    model_id="v5_ru",
    language="ru",
    speaker="eugene",
    sample_rate=48000,
    device="cpu",
    put_accent=True,
    put_yo=True,
)

print(f"Генерация {len(PHRASES)} фраз в {OUTPUT_DIR}...")

for key, text, note in PHRASES:
    wav = OUTPUT_DIR / f"{key}.wav"
    tts.tts(text, str(wav))
    size_kb = wav.stat().st_size // 1024
    print(f"  [{key:20s}] {text:35s} -> {wav} ({size_kb} KB)")

print(f"\nГотово. {len(PHRASES)} фраз в {OUTPUT_DIR}")

# Запуск mpv для первого файла
subprocess.run(["mpv", "--no-terminal", str(OUTPUT_DIR / "welcome.wav")])
