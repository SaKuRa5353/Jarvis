# 🤖 Джарвис — голосовой ассистент на русском

Голосовой ассистент с русским wakeword «Джарвис», работающий **полностью офлайн**:
распознавание речи, синтез, wakeword и LLM — всё локально, без API-ключей и облаков.

## ✨ Возможности

- 🎤 **Wakeword «Джарвис»** — офлайн, через [Vosk](https://alphacephei.com/vosk/)
- 🧠 **LLM** — [Qwen2.5:3b](https://ollama.com/) через Ollama (локально, приватно)
- 🗣️ **TTS** — [Piper](https://github.com/rhasspy/piper) (голос Ruslan, естественный русский)
- 🎙️ **STT** — Google Speech Recognition (ru-RU)
- 🎚️ **VAD** — [Silero VAD](https://github.com/snakers4/silero-vad)
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
| **Telegram** | «открой телеграм» | Другое приложение или веб-версия |
| **xdg-open** | Все веб-ссылки, поиск | Встроен в большинство дистрибутивов |

> **Не KDE?** Замени команды в `src/commands/apps.py` и `src/commands/system.py`:
> - `dolphin` → `nemo`, `thunar`, `nautilus`
> - `konsole` → `alacritty`, `kitty`, `gnome-terminal`
> - `spectacle` → `grim` + `slurp` (Wayland) или `scrot` (X11)

## 🚀 Установка

### 1. Клонируй репозиторий

```bash
git clone https://github.com/ТВОЙ_НИК/jarvis.git
cd jarvis
```

### 2. Создай виртуальное окружение

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Установи системные зависимости

```bash
# Arch / CachyOS
sudo pacman -S portaudio mpv ffmpeg
```

### 3.1. Установи опциональные приложения

```bash
# Arch / CachyOS
sudo pacman -S brave-browser dolphin konsole spectacle telegram-desktop
```

### 4. Установи Ollama и модель

```bash
sudo pacman -S ollama
sudo systemctl enable --now ollama
ollama pull qwen2.5:3b
```

### 5. Скачай модели

```bash
mkdir -p models/vosk models/piper

# Vosk
cd models/vosk
curl -L -O https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip
mv vosk-model-small-ru-0.22/* .
rmdir vosk-model-small-ru-0.22
rm vosk-model-small-ru-0.22.zip
cd ../..

# Piper
cd models/piper
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx.json
cd ../..
```

### 6. Настрой микрофон

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

Открой `config.py` и укажи свой `DEVICE`.

### 7. Запуск

```bash
python main.py
```

Скажи **«Джарвис»** → пауза → команду.

## 📁 Структура проекта

```
jarvis/
├── main.py
├── config.py
├── src/
│   ├── assistant.py
│   ├── wakeword.py
│   ├── vad.py
│   ├── stt.py
│   ├── tts.py
│   ├── llm.py
│   ├── logger.py
│   └── commands/
│       ├── apps.py
│       ├── system.py
│       ├── web.py
│       ├── hardware.py
│       └── utils.py
├── models/        (не в git)
├── data/          (не в git)
└── logs/          (не в git)
```

## 📝 Лицензия

MIT — делай что хочешь.

## 🙏 Благодарности

- [Vosk](https://alphacephei.com/vosk/) — офлайн распознавание
- [Piper](https://github.com/rhasspy/piper) — офлайн TTS
- [Ollama](https://ollama.com/) — локальный LLM
- [Silero VAD](https://github.com/snakers4/silero-vad) — детекция речи