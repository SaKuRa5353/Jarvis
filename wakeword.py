import time
import sounddevice as sd
import numpy as np
from openwakeword.model import Model

oww = Model(wakeword_model_paths=["models/hey_jarvis_v0.1.onnx"])

SAMPLE_RATE = 16000
CHUNK = 1280
DEVICE = 4
THRESHOLD = 0.35
COOLDOWN = 2.0

print("Слушаю... Скажи 'Hey Jarvis' (Ctrl+C для выхода)")

last_trigger = 0.0

def audio_callback(indata, frames, time_info, status):
    global last_trigger
    if status:
        print(f"[status] {status}")
    audio = (indata[:, 0] * 32767).astype(np.int16)
    prediction = oww.predict(audio)
    score = prediction.get("hey_jarvis_v0.1", 0.0)

    now = time.time()
    if score > THRESHOLD and (now - last_trigger) > COOLDOWN:
        last_trigger = now
        print(f"🎯 ДЖАРВИС УСЛЫШАН (score={score:.2f})")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    blocksize=CHUNK,
    device=DEVICE,
    callback=audio_callback,
):
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nВыход.")