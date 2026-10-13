import wave
import subprocess
from piper import PiperVoice

voice = PiperVoice.load("models/piper/ru_RU-ruslan-medium.onnx")
chunks = list(voice.synthesize("Привет, я Джарвис"))

first = chunks[0]
with wave.open("/tmp/check.wav", "wb") as wav_file:
    wav_file.setnchannels(first.sample_channels)
    wav_file.setsampwidth(first.sample_width)
    wav_file.setframerate(first.sample_rate)
    wav_file.writeframes(b"".join(c.audio_int16_bytes for c in chunks))

print("Файл создан, играю...")
subprocess.run(["mpv", "--no-terminal", "/tmp/check.wav"])
