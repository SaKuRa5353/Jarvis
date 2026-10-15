# 🤖 Джарвис — голосовой ассистент на русском

Голосовой ассистент с русским wakeword «Джарвис», работающий **полностью офлайн**:
распознавание речи, синтез, wakeword и LLM — всё локально, без API-ключей и облаков.

## ✨ Возможности

- 🎤 **Wakeword «Джарвис»** — офлайн, через [Vosk](https://alphacephei.com/vosk/)
- 🧠 **LLM** — [Qwen2.5:3b](https://ollama.com/) через Ollama (локально, приватно)
- 🗣️ **TTS** — [Silero](https://github.com/snakers4/silero-models) (голос Eugene, естественный русский)
- 🎙️ **STT** — Google Speech Recognition (ru-RU)
- 🎚️ **VAD** — [Silero VAD](https://github.com/snakers4/silero-vad)
- ⚡ **Скорость** — ~1-2 сек на ответ LLM (8 потоков CPU)
- 🛑 **Прерывание по wakeword** — скажи «Джарвис» во время ответа, и он остановится

### Команды

| Категория | Команды |
|-----------|---------|
| **Приложения** | «открой проводник», «запусти vscode», «открой терминал», «открой браузер», «открой телеграм» |
| **Система** | «который час», «какая сегодня дата», «заблокируй экран» |
| **Звук** | «сделай тише», «сделай громче», «выключи звук», «включи звук» |
| **Скриншот** | «сделай скриншот» |
| **Веб** | «открой вк», «открой youtube», «открой почту», «открой расписание» |
| **Поиск** | «загугли X», «найди в гугле X» |
| **Погода** | «какая погода», «погода на завтра», «погода в Москве» |
| **Инфо** | «дай характеристику железа» |
| **LLM** | любые вопросы: «сколько будет 2+2», «в каком году была ВОВ» |

## 📋 Требования

### Обязательные

- **ОС:** Linux (разработано на CachyOS / Arch, должно работать на любом дистрибутиве)
- **Python:** 3.12+
- **ОЗУ:** минимум 8 GB (для Qwen2.5:3b)
- **Микрофон:** любой USB или встроенный
- **Наушники** — рекомендуются (чтобы Vosk не слышал эхо TTS)
- **Ollama** — для локального LLM
- **mpv** — для воспроизведения TTS
- **PortAudio** — для `sounddevice`

### Опциональные (для конкретных команд)

| Что | Для каких команд | Можно заменить на |
|-----|------------------|-------------------|
| **Brave** | «открой браузер» | Любой браузер (поменяй в `src/commands/apps.py`) |
| **Dolphin** | «открой проводник» | `nemo`, `thunar`, `nautilus` |
| **Konsole** | «открой терминал» | `alacritty`, `kitty`, `gnome-terminal` |
| **Spectacle** | «сделай скриншот» | `grim` + `slurp` (Wayland), `scrot` (X11) |
| **pactl** | «сделай тише/громче», mute | — (есть в большинстве систем) |
| **loginctl** | «заблокируй экран» | `qdbus org.kde.screensaver` (KDE) |

## 🚀 Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/SaKuRa5353/Jarvis.git
cd Jarvis
2. Создать виртуальное окружение
bash
python3.12 -m venv .venv
source .venv/bin/activate
3. Установить зависимости
bash
pip install -r requirements.txt
4. Установить Ollama и скачать модель
bash
# Установка Ollama (Arch / CachyOS)
sudo pacman -S ollama
sudo systemctl enable --now ollama

# Скачать Qwen2.5:3b
ollama pull qwen2.5:3b
5. Настроить Ollama для CPU (опционально, но рекомендуется)
bash
sudo mkdir -p /etc/systemd/system/ollama.service.d

echo '[Service]' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
echo 'Environment="OLLAMA_NUM_THREADS=8"' | sudo tee -a /etc/systemd/system/ollama.service.d/override.conf
echo 'Environment="OLLAMA_KEEP_ALIVE=30m"' | sudo tee -a /etc/systemd/system/ollama.service.d/override.conf

sudo systemctl daemon-reload
sudo systemctl restart ollama
6. Скачать модели для wakeword
bash
mkdir -p models/vosk
cd models/vosk
curl -L -O https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip
mv vosk-model-small-ru-0.22/* .
rmdir vosk-model-small-ru-0.22
rm vosk-model-small-ru-0.22.zip
cd ../..
Silero TTS и VAD скачиваются автоматически при первом запуске.

7. Настроить микрофон
Проверь, что микрофон работает:

bash
python tests/test_mic.py
Если amplitude < 500 — подними усиление в pavucontrol → Input Devices.

Найди индекс своего микрофона:

bash
python -c "import sounddevice; print(sounddevice.query_devices())"
Запиши индекс в config.py:

python
DEVICE = 4   # ← твой индекс микрофона
8. Запустить
bash
python main.py
Скажи «Джарвис» (пауза 1 сек) → «привет».

🏗️ Структура проекта
text
Jarvis/
├── main.py                    # точка входа
├── config.py                  # все константы
├── src/
│   ├── assistant.py           # главный цикл
│   ├── wakeword.py            # Vosk wakeword «Джарвис»
│   ├── vad.py                 # Silero VAD
│   ├── stt.py                 # Google STT
│   ├── tts.py                 # Silero TTS (Eugene)
│   ├── llm.py                 # Ollama + Qwen2.5
│   ├── logger.py              # логирование
│   └── commands/
│       ├── __init__.py        # собирает все команды
│       ├── apps.py            # приложения
│       ├── system.py          # система (время, звук, скриншот)
│       ├── web.py             # браузер, поиск
│       ├── weather.py         # погода
│       ├── hardware.py        # характеристики железа
│       └── utils.py           # общие утилиты
├── models/                    # модели (не в git)
├── data/                      # история диалога (не в git)
├── logs/                      # логи (не в git)
└── tests/                     # тестовые скрипты
🎯 Как добавить свою команду
Открой подходящий файл в src/commands/ (например, apps.py)

Добавь блок if в функцию try_handle:

python
def try_handle(text: str):
    if any(w in text for w in ["мой триггер"]):
        return (
            "Голосовой ответ",
            lambda: subprocess.Popen(["моя", "команда"]),
        )
    return None
Готово. Команда подключится автоматически.

🙏 Благодарности
Vosk — офлайн распознавание

Silero — офлайн TTS

Ollama — локальный LLM

Qwen — модель

Silero VAD — детекция речи

SpeechRecognition — STT

sounddevice — аудио

📄 Лицензия
MIT — см. LICENSE.

text
