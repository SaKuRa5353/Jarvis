# 🤖 Джарвис — голосовой ассистент на русском

Голосовой ассистент с русским wakeword «Джарвис», работающий **полностью офлайн**:
распознавание речи, синтез, wakeword и LLM — всё локально, без API-ключей и облаков.

## ✨ Возможности

- 🎤 **Wakeword «Джарвис»** — офлайн, через [Vosk](https://alphacephei.com/vosk/)
- 🧠 **LLM** — [Qwen2.5:3b-instruct](https://ollama.com/) через Ollama (локально, приватно)
- 🗣️ **TTS** — [Silero](https://github.com/snakers4/silero-models) (голос Eugene, естественный русский)
- ⚡ **Готовые фразы** — частые ответы записаны WAV и играются за 0.1 сек вместо синтеза
- 🎙️ **STT** — Vosk (ru-RU, офлайн, без интернета)
- 🎚️ **VAD** — [Silero VAD](https://github.com/snakers4/silero-vad)
- 🪟 **Контекст окна** — LLM знает, в каком окне ты работаешь (kdotool, только чтение)
- ⚡ **Скорость** — ~1-2 сек на ответ LLM (8 потоков CPU)
- 🛑 **Прерывание по wakeword** — скажи «Джарвис» во время ответа, и он остановится
- 🔗 **Команда вместе с обращением** — «Джарвис открой терминал» срабатывает сразу, без ожидания

### Команды

| Категория | Команды |
|-----------|---------|
| **Приложения** | «открой проводник», «запусти vscode», «открой терминал», «открой браузер», «открой телеграм» |
| **Система** | «который час», «какая сегодня дата», «заблокируй экран» |
| **Датчики** | «температура процессора», «перегрев компьютера», «сколько градусов у процессора» |
| **Звук** | «сделай тише», «сделай громче», «громкость выше», «выключи звук», «включи звук» |
| **Скриншот** | «сделай скриншот» |
| **Веб** | «открой вк», «открой youtube», «открой почту», «открой расписание», «открой дипсик» |
| **Поиск** | «загугли X», «найди в гугле X», «найди в интернете X» |
| **Погода** | «какая погода», «погода на завтра», «погода в Москве» |
| **Инфо** | «дай характеристику железа» |
| **LLM** | любые вопросы: «сколько будет 2+2», «в каком году была ВОВ» |

## 📋 Требования

### Обязательные

- **ОС:** Linux (разработано на CachyOS / Arch, должно работать на любом дистрибутиве)
- **Python:** 3.12+
- **ОЗУ:** минимум 8 GB (для qwen2.5:3b-instruct)
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
| **lm_sensors** | «температура процессора» | — (нужен вывод `sensors`) |
| **kdotool** | контекст активного окна для LLM | отключи `WINDOW_CONTEXT_ENABLED` в `config.py` |

Погода требует доступа в интернет (`wttr.in`); остальные команды работают офлайн.

## 🚀 Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/SaKuRa5353/Jarvis.git
cd Jarvis
```

### 2. Создать виртуальное окружение

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Установить Ollama и скачать модель

```bash
# Установка Ollama (Arch / CachyOS)
sudo pacman -S ollama
sudo systemctl enable --now ollama

# Скачать Qwen2.5:3b-instruct
ollama pull qwen2.5:3b-instruct
```

### 5. Настроить Ollama для CPU (опционально, но рекомендуется)

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d

echo '[Service]' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
echo 'Environment="OLLAMA_NUM_THREADS=8"' | sudo tee -a /etc/systemd/system/ollama.service.d/override.conf
echo 'Environment="OLLAMA_KEEP_ALIVE=30m"' | sudo tee -a /etc/systemd/system/ollama.service.d/override.conf

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 6. Скачать модели

```bash
# Vosk — одна модель и на wakeword, и на распознавание (88 MB)
mkdir -p models/vosk
cd models/vosk
curl -L -O https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip
mv vosk-model-small-ru-0.22/* .
rmdir vosk-model-small-ru-0.22
rm vosk-model-small-ru-0.22.zip
cd ../..

# Silero VAD (1.8 MB)
curl -L -o models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

Silero TTS скачивается автоматически при первом запуске.

### 7. Установить kdotool (для контекста активного окна)

```bash
# Arch / CachyOS — из репозиториев или AUR
sudo pacman -S kdotool        # либо: yay -S kdotool-git

# либо скачай готовый бинарник со страницы релизов
# https://github.com/jinliu/kdotool/releases
install -Dm755 ./kdotool ~/.local/bin/kdotool
```

Необязательно: без kdotool ассистент работает, но LLM не будет знать, в каком окне ты работаешь.
Отключить совсем: `WINDOW_CONTEXT_ENABLED = False` в `config.py`.

### 8. Настроить микрофон

Проверь, что микрофон работает:

```bash
python tests/check_mic.py
```

Если `Max amplitude < 500` — подними усиление в `pavucontrol` → Input Devices.

Найди индекс своего микрофона:

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

Запиши индекс в `config.py`:

```python
DEVICE = 4   # ← твой индекс микрофона
```

### 9. Сгенерировать готовые фразы (опционально)

Короткие ответы («К вашим услугам», «Открываю») лежат в `src/phrases/` готовыми WAV —
они играются мгновенно, без синтеза. Если файлов нет или нужен другой голос:

```bash
python tests/generate_phrases.py
```

### 10. Запустить

```bash
python main.py
```

Скажи «Джарвис» (пауза 1 сек) → «привет».

## 🏗️ Структура проекта

```text
Jarvis/
├── main.py                    # точка входа
├── config.py                  # все константы
├── src/
│   ├── assistant.py           # главный цикл + прерывание по wakeword
│   ├── audio.py               # единый микрофон: один поток на всех
│   ├── window_context.py      # чтение активного окна (kdotool)
│   ├── wakeword.py            # Vosk wakeword «Джарвис»
│   ├── vad.py                 # Silero VAD — запись команды
│   ├── stt.py                 # Vosk STT (офлайн)
│   ├── vosk_model.py          # общая загрузка модели Vosk (singleton)
│   ├── tts.py                 # Silero TTS (Eugene) + mpv
│   ├── phrases/               # готовые WAV коротких ответов (без синтеза)
│   ├── llm.py                 # Ollama + Qwen2.5
│   ├── logger.py              # логирование
│   └── commands/
│       ├── __init__.py        # собирает все команды
│       ├── apps.py            # приложения
│       ├── system.py          # система (время, звук, скриншот)
│       ├── sensors.py         # температура CPU (lm_sensors)
│       ├── web.py             # браузер, поиск
│       ├── weather.py         # погода
│       ├── hardware.py        # характеристики железа
│       └── utils.py           # общие утилиты
├── models/                    # модели (не в git)
├── data/                      # история диалога (не в git)
└── tests/                     # проверочные скрипты (check_*.py)
```

## 🔧 Как это работает

```text
микрофон (один RawInputStream + фоновый читатель)
   │
   ├──► wakeword ──► «Джарвис»? ──► фраза-приветствие (готовый WAV)
   │                     │
   │                     └─ «Джарвис открой код» → команда уже здесь, VAD не нужен
   │
   └──► VAD ──► запись команды ──► Vosk STT ──► план команд
                                                    │
                        правило сработало ◄─────────┴─────────► нет: LLM (Ollama)
                              │                                      │
                    готовый WAV + действие              контекст активного окна
```

Ключевые решения:

- **Один микрофон.** `src/audio.py` держит единственный `RawInputStream`, фоновый поток
  раскладывает блоки по потребителям. Два `RawInputStream` на одном устройстве в PipeWire
  не живут — второй открывался с тишиной на входе.
- **Хвост wakeword.** `WakewordDetector.wait()` возвращает всю распознанную фразу.
  Если команда произнесена вместе с обращением, она берётся оттуда сразу: минус ~4 секунды
  на VAD и повторное распознавание.
- **Готовые фразы.** Частые ответы записаны WAV-файлами (`src/phrases/`) и играются за
  0.1 сек вместо ~1.5 сек на синтез.
- **Прерывание по wakeword.** Скажи «Джарвис» во время ответа — TTS или LLM-запрос
  останавливаются, цикл начинается заново.
- **Контекст окна.** Активное окно (kdotool, только чтение) добавляется в системный промпт
  текущего запроса и не сохраняется в историю.

## 🧪 Проверочные скрипты

```bash
python tests/check_mic.py              # микрофон: индекс и уровень
python tests/check_tts.py --no-play    # Silero TTS и наличие фраз
python tests/check_commands.py         # правила распознавания команд
python tests/check_phrases_search.py   # готовые фразы и паттерны поиска
python tests/check_wakewords.py        # варианты «Джарвис» в Vosk
python tests/check_sensors.py          # температура CPU
python tests/check_window.py           # активное окно через kdotool
python tests/check_stt.py              # качество распознавания
```

## 🎯 Как добавить свою команду

Открой подходящий файл в `src/commands/` (например, `apps.py`) и добавь блок в `try_handle`:

```python
def try_handle(text: str):
    if any(w in text for w in ["мой триггер"]):
        return (
            "Голосовой ответ",                    # что сказать
            lambda: run(["моя", "команда"]),      # что сделать (или None)
            "done",                               # ключ готовой фразы (или None)
        )
    return None
```

Готово — команда подключится автоматически, `src/commands/__init__.py` опрашивает все модули
по очереди. Первый, кто вернул не `None`, и есть обработчик.

## 🙏 Благодарности

- [Vosk](https://alphacephei.com/vosk/) — офлайн распознавание речи
- [Silero](https://github.com/snakers4/silero-models) — офлайн TTS и VAD
- [Ollama](https://ollama.com/) — локальный LLM
- [Qwen](https://qwenlm.github.io/) — модель
- [sounddevice](https://python-sounddevice.readthedocs.io/) — захват аудио
- [kdotool](https://github.com/jinliu/kdotool) — активное окно в KDE/Wayland

## 📄 Лицензия

MIT — см. [LICENSE](LICENSE).
