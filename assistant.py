"""
Jarvis — голосовой ассистент.

Цикл: wakeword → запись с VAD → STT → план → TTS (Piper) → выполнение.
"""

import subprocess
import time
import warnings
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import onnxruntime as ort
import speech_recognition as sr
from openwakeword.model import Model
from piper import PiperVoice


# ──────────────────────── ПОДАВЛЕНИЕ ШУМА ────────────────────────

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")


# ──────────────────────── КОНСТАНТЫ ────────────────────────

SAMPLE_RATE = 16000
DEVICE = 4                          # USB Audio CODEC
WAKE_MODEL = "models/hey_jarvis_v0.1.onnx"
VAD_MODEL = "models/silero_vad.onnx"
PIPER_MODEL = "models/piper/ru_RU-ruslan-medium.onnx"

WAKE_THRESHOLD = 0.42
WAKE_COOLDOWN = 2.0

VAD_FRAME = 512
VAD_SPEECH_THRESHOLD = 0.5
VAD_SILENCE_THRESHOLD = 0.4
VAD_MIN_SILENCE_SEC = 0.8
VAD_MAX_RECORD_SEC = 6
VAD_MIN_SPEECH_SEC = 0.3

TTS_WAV = Path("/tmp/jarvis_tts.wav")


# ──────────────────────── МОДЕЛИ ────────────────────────

wake_model = Model(wakeword_model_paths=[WAKE_MODEL])
vad_session = ort.InferenceSession(VAD_MODEL, providers=["CPUExecutionProvider"])

_piper_voice = None


def _get_piper_voice() -> PiperVoice:
    """Ленивая загрузка Piper — только при первом обращении."""
    global _piper_voice
    if _piper_voice is None:
        print("[TTS] Загружаю Piper...")
        _piper_voice = PiperVoice.load(PIPER_MODEL)
    return _piper_voice


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


# ──────────────────────── TTS (Piper, локальный) ────────────────────────

def speak(text: str):
    print(f"🔊 Джарвис: {text}")
    try:
        voice = _get_piper_voice()
        with wave.open(str(TTS_WAV), "wb") as wav_file:
            # Параметры узнаём из первого чанка
            first_chunk = None
            chunks = []
            for chunk in voice.synthesize(text):
                if first_chunk is None:
                    first_chunk = chunk
                chunks.append(chunk.audio_int16_bytes)

            if first_chunk is None:
                print("[TTS] нет аудио")
                return

            wav_file.setnchannels(first_chunk.sample_channels)
            wav_file.setsampwidth(first_chunk.sample_width)
            wav_file.setframerate(first_chunk.sample_rate)
            wav_file.writeframes(b"".join(chunks))

        subprocess.run(
            ["mpv", "--no-terminal", "--really-quiet", str(TTS_WAV)],
            check=False,
        )
    except Exception as e:
        print(f"[TTS error] {e}")


# ──────────────────────── РОУТЕР ────────────────────────

def plan_command(text: str):
    """Возвращает (response_text, action_callable) или (None, None). Ничего не выполняет."""
    t = text.lower()
    t = t.replace("джарвис", "").strip()

    if any(w in t for w in ["проводник", "файлы", "документы", "папк"]):
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
            "Запускаю редактор кода",
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
                continue

            # Отсекаем ложные срабатывания: распознали только обращение
            normalized = text.lower().strip().replace(",", "").replace(".", "")
            if len(normalized.split()) <= 1 and "джарвис" in normalized:
                print("[ложное срабатывание — игнор]")
                continue

            response, action = plan_command(text)

            if response is None:
                if len(text.split()) >= 3:
                    speak("Не знаю такой команды")
                continue

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