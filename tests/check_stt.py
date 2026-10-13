import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone(device_index=4)

with mic as source:
    r.adjust_for_ambient_noise(source, duration=1)
    print("Говори 5 секунд...")
    audio = r.listen(source, timeout=5, phrase_time_limit=5)

print("Распознаю...")
try:
    text = r.recognize_google(audio, language="ru-RU")
    print(f"Ты сказал: {text}")
except sr.UnknownValueError:
    print("Не разобрал")
except sr.RequestError as e:
    print(f"Ошибка сервиса: {e}")

