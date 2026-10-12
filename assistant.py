"""
Jarvis — голосовой ассистент.

Цикл: wakeword → запись с VAD → STT → план → TTS → выполнение.
"""

import asyncio
import os
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import sounddevice as sd
import onnxruntime as ort
import speech_recognition as sr
import edge_tts
from openwakeword.model import Model


# ──────────────────────── ПОДАВЛЕНИЕ ШУМА ────────────────────────

# Скрываем warning про CUDAExecutionProvider (у нас CPU)
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")


# ──────────────────────── КОНСТАНТЫ ────────────────────────

SAMPLE_RATE = 16000
DEVICE = 4                          # USB Audio CODEC
WAKE_MODEL = "models/hey_jarvis_v0.1.onnx"
VAD_MODEL = "models/silero_vad.onnx"

WAKE_THRESHOLD = 0.35
WAKE_COOLDOWN = 2.0

VAD_FRAME = 512                     # 32 мс при 16 kHz
VAD_SPEECH_THRESHOLD = 0.5          # порог "это речь"
VAD_SILENCE_THRESHOLD = 0.4         # порог "это тишина" (было 0.35)
VAD_MIN_SILENCE_SEC = 0.8           # сколько тишины до конца фразы
VAD_MAX_RECORD_SEC = 6              # было 10
VAD_MIN_SPEECH_SEC = 0.3            # отсекаем короткие шумы

TTS_VOICE = "ru-RU-DmitryNeural"
TTS_FILE = Path("/tmp/jarvis_tts.mp3")


# ──────────────────────── МОДЕЛИ ────────────────────────

wake_model = Model(wakeword_model_paths=[WAKE_MODEL])
vad_session = ort.InferenceSession(VAD_MODEL, providers=["CPUExecutionProvider"])


# ──────────────────────── WAKE WORD ────────────────────────

_wake_triggered = False
_wake_last_time = 0.0


def _wake_callback(indata, frames, time_info, status):
    global _wake_triggered, _wake_last_time
    if status:
        print(f"[wake status] {status}")
    audio = (indata[:, 0] * 32767).astype(np.int16)
    prediction = wake_model.predict(audio)
    score = prediction.get("hey_jarvis_v0.1", 0.0)
    now = time.time()
    if score > WAKE_THRESHOLD and (now - _wake_last_time) > WAKE_COOLDOWN:
        _wake_last_time = now
        _wake_triggered = True


def wait_for_wakeword():
    """Блокирующе ждёт, пока не услышит 'Hey Jarvis'."""
    global _wake_triggered
    _wake_triggered = False
    print("💤 Жду 'Hey Jarvis'...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=1280,
        device=DEVICE,
        callback=_wake_callback,
    ):
        while not _wake_triggered:
            sd.sleep(100)

    print("🎯 Услышал wakeword!")


# ──────────────────────── VAD (Silero) ────────────────────────

_vad_h = np.zeros((2, 1, 64), dtype=np.float32)
_vad_c = np.zeros((2, 1, 64), dtype=np.float32)


def _vad_reset():
    global _vad_h, _vad_c
    _vad_h = np.zeros((2, 1, 64), dtype=np.float32)
    _vad_c = np.zeros((2, 1, 64), dtype=np.float32)


def _vad_prob(frame_int16: np.ndarray) -> float:
    """Возвращает вероятность речи для кадра 512 сэмплов int16."""
    global _vad_h, _vad_c
    x = frame_int16.astype(np.float32) / 32768.0
    x = x.reshape(1, -1)
    out, _vad_h, _vad_c = vad_session.run(
        None,
        {
            "input": x,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            "h": _vad_h,
            "c": _vad_c,
        },
    )
    return float(out[0][0])


def record_until_silence() -> np.ndarray:
    """Записывает аудио с микрофона, пока не наступит тишина."""
    _vad_reset()

    buffer = []
    speech_started = False
    silence_frames = 0
    min_silence_frames = int(VAD_MIN_SILENCE_SEC * SAMPLE_RATE / VAD_FRAME)
    max_frames = int(VAD_MAX_RECORD_SEC * SAMPLE_RATE / VAD_FRAME)
    min_speech_frames = int(VAD_MIN_SPEECH_SEC * SAMPLE_RATE / VAD_FRAME)
    speech_frames = 0

    print("🎤 Слушаю команду...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=VAD_FRAME,
        device=DEVICE,
    ) as stream:
        for _ in range(max_frames):
            frame, _ = stream.read(VAD_FRAME)
            frame = frame[:, 0].copy()
            prob = _vad_prob(frame)

            if prob > VAD_SPEECH_THRESHOLD:
                speech_started = True
                silence_frames = 0
                speech_frames += 1
                buffer.append(frame)
            elif speech_started:
                silence_frames += 1
                buffer.append(frame)
                if silence_frames >= min_silence_frames:
                    break

    if speech_frames < min_speech_frames:
        print("⚠️ Слишком короткая речь")
        return np.array([], dtype=np.int16)

    audio = np.concatenate(buffer)
    print(f"✅ Записано {len(audio) / SAMPLE_RATE:.1f} сек")
    return audio


# ──────────────────────── STT ────────────────────────

_recognizer = sr.Recognizer()


def recognize(audio_int16: np.ndarray) -> str | None:
    if len(audio_int16) == 0:
        return None
    audio_data = sr.AudioData(audio_int16.tobytes(), SAMPLE_RATE, 2)
    print("🧠 Распознаю...")
    try:
        text = _recognizer.recognize_google(audio_data, language="ru-RU")
        print(f"📝 Ты: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Не разобрал")
        return None
    except sr.RequestError as e:
        print(f"❌ Ошибка STT: {e}")
        return None


# ──────────────────────── TTS ────────────────────────

async def _tts_save(text: str, path: Path, retries: int = 3):
    """Генерирует mp3 с retry — edge-tts иногда падает на коротких фразах."""
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, TTS_VOICE)
            await communicate.save(str(path))
            if path.exists() and path.stat().st_size > 1000:
                return
            print(f"[TTS retry {attempt + 1}] файл пустой")
        except Exception as e:
            print(f"[TTS retry {attempt + 1}] {e}")
        await asyncio.sleep(0.5)
    raise RuntimeError("TTS failed after retries")


def speak(text: str):
    print(f"🔊 Джарвис: {text}")
    try:
        asyncio.run(_tts_save(text, TTS_FILE))
        size = TTS_FILE.stat().st_size
        print(f"[mpv] играю {TTS_FILE} ({size} байт)")
        result = subprocess.run(
            ["mpv", "--no-terminal", str(TTS_FILE)],
            check=False,
        )
        print(f"[mpv] exit code: {result.returncode}")
    except Exception as e:
        print(f"[TTS error] {e}")


# ──────────────────────── РОУТЕР (план без выполнения) ────────────────────────

def plan_command(text: str):
    """
    Возвращает (response_text, action_callable) или (None, None).
    Ничего НЕ выполняет — только планирует.
    """
    t = text.lower()

    # Вырезаем обращение "джарвис"
    t = t.replace("джарвис", "").strip()

    if any(w in t for w in ["проводник", "файлы", "документы", ]):
        return (
            "Открываю проводник",
            lambda: subprocess.Popen(
                ["dolphin"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ),
        )

    if any(w in t for w in [
        "vscode", "вс код", "вс-код", "vs code", "visual studio",
        "вс студия", "код",
    ]):
        return (
            "Запускаю VS Code",
            lambda: subprocess.Popen(
                ["code"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ),
        )

    if any(w in t for w in ["браузер", "brave", "брейв"]):
        return (
            "Открываю браузер",
            lambda: subprocess.Popen(
                ["brave"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ),
        )

    return None, None


# ──────────────────────── ГЛАВНЫЙ ЦИКЛ ────────────────────────

def main():
    print("=" * 50)
    print("         ДЖАРВИС ЗАПУЩЕН")
    print("=" * 50)

    while True:
        try:
            wait_for_wakeword()
            audio = record_until_silence()
            text = recognize(audio)

            if text is None:
                # короткая запись или не разобрал — просто возвращаемся
                continue

            response, action = plan_command(text)

            if response is None:
                # Игнорируем короткие непонятные фразы, отвечаем только на осмысленные
                if len(text.split()) >= 3:
                    response = "Не знаю такой команды"
                    speak(response)
                continue

            # Сначала говорим — потом делаем
            speak(response)
            if action is not None:
                try:
                    action()
                except Exception as e:
                    print(f"[action error] {e}")

        except KeyboardInterrupt:
            print("\n👋 Выход")
            break
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()