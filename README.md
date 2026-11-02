# 🤖 Джарвис — офлайн-голосовой ассистент

Голосовой ассистент с русским wakeword «Джарвис»: распознавание речи, синтез и LLM —
без API-ключей и облаков. Почти всё работает офлайн; исключение — команда погоды (wttr.in).

## ✨ Что умеет

- 🎤 **Wakeword «Джарвис»** — детектируется через Vosk (офлайн) + Silero VAD
- 🧠 **LLM** — `qwen2.5:3b` через [Ollama](https://ollama.com/), локально, без API-ключей
- 🗣️ **TTS** — [Silero](https://github.com/snakers4/silero-models), голос Eugene, офлайн
- 🎙️ **STT** — Vosk (ru-RU), офлайн
- 🎚️ **VAD** — [Silero VAD](https://github.com/snakers4/silero-vad), пороговый детектор
- ⚡ **Готовые фразы** — частые ответы записаны WAV и играются за ~0.1 сек
- 🗃️ **Кэш TTS** — hash(текст) → WAV, повторный ответ играется без синтеза
- 🌊 **Стриминг ответа** — длинные ответы озвучиваются по предложениям во время генерации LLM
- 🛑 **Прерывание по wakeword** — скажи «Джарвис» во время ответа, и он остановится
- 🔗 **Команда вместе с обращением** — «Джарвис открой терминал» срабатывает сразу, без ожидания
- 🪟 **Контекст окна** — LLM знает, в каком окне ты работаешь (kdotool, только чтение)
- 🔧 **Fuzzy-коррекция** — исправляет типичные ошибки STT через взвешенный Дамерау-Левенштейн
- 🤖 **Intent-классификатор** — Байесовский классификатор защищает от ложных действий
- 🔑 **Tool calling** — LLM может явно вызвать инструменты (открыть сайт, посчитать)

### Команды

Полный список интентов — в разделе [🗣️ Список команд](#-список-команд) (таблица
генерируется из `src/intent.py`, поле `CATALOG`).

| Категория | Команды |
|-----------|---------|
| **Приложения** | «открой проводник», «запусти vscode», «открой терминал», «открой браузер», «открой телеграм» |
| **Система** | «который час», «какая сегодня дата», «заблокируй экран», «сделай скриншот» |
| **Датчики** | «температура процессора», «перегрев компьютера», «сколько градусов у процессора» |
| **Звук** | «сделай тише», «сделай громче», «громкость выше», «выключи звук», «включи звук» |
| **Веб** | «открой вк», «открой ютуб», «открой почту», «открой расписание», «открой дипсик» |
| **Поиск** | «загугли X», «найди в гугле X», «найди в интернете X» |
| **Погода** | «какая погода», «прогноз на завтра», «погода в Москве» |
| **Инфо** | «характеристики железа», «что за компьютер» |
| **LLM** | любые вопросы: «сколько будет 2+2», «в каком году была ВОВ» |

## 📋 Требования

### Обязательные

- **ОС:** Linux (разработано на CachyOS / Arch, должно работать на любом дистрибутиве)
- **Python:** 3.12+
- **ОЗУ:** минимум 4 ГБ (8 ГБ рекомендуется для LLM и моделей)
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
source .venv/bin/activate.fish   # в bash/zsh: source .venv/bin/activate
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

ollama pull qwen2.5:3b
```

> По умолчанию используется `qwen2.5:3b`. Чтобы переключить модель без правки кода,
> создай `.env` в корне проекта: `JARVIS_LLM_MODEL=qwen3-jarvis:latest`

### 5. Настроить Ollama для CPU (рекомендуется)

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

Мастер настройки показывает пригодные устройства ввода, записывает выбор в `.env`
и предлагает тестовую запись:

```bash
python main.py --setup
```

Вручную — добавь в `.env` (файл в `.gitignore`, в репозиторий не попадёт):

```bash
JARVIS_DEVICE=4
```

Проверка уровня сигнала: `python tests/check_mic.py`. Если `Max amplitude < 500` —
подними усиление в `pavucontrol` → Input Devices.

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

## 🔧 Как это работает

```text
микрофон (один RawInputStream + фоновый читатель)
   │
   ├──► wakeword ──► «Джарвис»? ──► фраза-приветствие (готовый WAV)
   │                     │
   │                     └─ «Джарвис открой код» → команда уже здесь, VAD не нужен
   │
   └──► VAD + Vosk STT ──► fuzzy (WDL + фонетика) ──► intent (Naive Bayes)
                                                            │
                       действие разрешено ◄─────────────────┴──► нет: LLM + tool calling
                             │                                        │
                   готовый WAV + действие                контекст окна + ответ/инструмент
```

### Режим диалога (15 сек)

После обработки команды ассистент не засыпает сразу, а ждёт следующую реплику
в течение `CONVERSATION_TIMEOUT_SEC` (15 сек, настраивается в `config.py`).
Пользователь может продолжать разговор без повторного «Джарвис»:

```text
«Джарвис» → «К вашим услугам» → «какая погода» → «Погода: 3°, ясно…»
  → [диалог 15 сек] → «а ветер?» → «Ветер 2 м/с…» → [диалог 15 сек]
```

Если в диалоге сказать «Джарвис» — режим прерывается и начинается новый цикл
с приветствием. Если молчать — через 15 секунд ассистент снова ждёт обращения.

### Ключевые решения

- **Один микрофон.** `src/audio.py` держит единственный `RawInputStream`, фоновый поток
  раскладывает блоки по потребителям. Два `RawInputStream` на одном устройстве в PipeWire
  не живут — второй открывался с тишиной на входе.
- **Хвост wakeword.** `WakewordDetector.wait()` возвращает всю распознанную фразу.
  Если команда произнесена вместе с обращением, она берётся оттуда сразу: минус ~4 секунды
  на VAD и повторное распознавание.
- **Фразы не тормозят ответ.** «К вашим услугам» и «Секунду» играют параллельно
  обработке: LLM-запрос стартует немедленно, а не после конца фразы. Если модель
  успела ответить во время приветствия — фаза «Секунду» пропускается.
- **Готовые фразы.** Частые ответы записаны WAV-файлами (`src/phrases/`) и играются за
  0.1 сек вместо ~1.5 сек на синтез.
- **Кэш синтеза.** Каждый синтезированный ответ ложится в кэш `data/tts_cache`
  (hash текста → WAV, не больше 200 файлов): повторный ответ играет мгновенно.
- **Стриминг ответа.** Ответы длиннее ~30 символов озвучиваются по предложениям,
  пока LLM ещё генерирует: первое аудио звучит за секунды, а не после всего ответа.
  Короткие ответы идут целиком через обычный синтез — так мусорные ответы
  («Не расслышал») по-прежнему заменяются готовой фразой.
- **Прерывание по wakeword.** Скажи «Джарвис» во время ответа — TTS или LLM-запрос
  останавливаются, цикл начинается заново.
- **Контекст окна.** Активное окно (kdotool, только чтение) добавляется в системный промпт
  текущего запроса и не сохраняется в историю.

## 🧪 Проверочные скрипты

| Скрипт | Что проверяет |
|--------|--------------|
| `python tests/check_mic.py` | микрофон: индекс и уровень громкости |
| `python tests/check_tts.py --no-play` | Silero TTS и наличие WAV-фраз |
| `python tests/check_commands.py` | правила распознавания команд |
| `python tests/check_fuzzy.py` | нечёткое исправление STT-ошибок |
| `python tests/check_phrases_search.py` | готовые фразы и паттерны поиска |
| `python tests/check_wakewords.py` | варианты «Джарвис» в Vosk |
| `python tests/check_sensors.py` | температура CPU |
| `python tests/check_window.py` | активное окно через kdotool |
| `python tests/check_stt.py` | качество распознавания речи (нужен микрофон) |
| `python tests/check_llm.py` | доступность Ollama и модели |

## 🧪 Unit-тесты (pytest)

```bash
pip install -r requirements-dev.txt   # pytest, numpy, openai, python-dotenv
python -m pytest
```

Тесты, требующие микрофон или железо, помечены `@pytest.mark.hw` и в CI пропускаются:

```bash
python -m pytest -m "not hw"   # только юнит-тесты, без устройств и сети
```

Покрывают: математическое ядро (`src/mathcore/`), fuzzy-коррекцию, правила команд
(`plan_command`), классификатор интентов, готовые фразы, метрики.
CI: `.github/workflows/tests.yml` (push + PR).

## 📂 Структура проекта

```
src/
  assistant.py       главный цикл: wake → запись → план → ответ
  wakeword.py        детектор обращения через Vosk + Silero VAD
  wakeword_dtw.py    экспериментальный DTW-детектор по эталонам (пока вне цикла assistant.py)
  vad.py             SpeechRecorder: запись речи до тишины
  stt.py             VoskSTT — офлайн распознавание
  tts.py             Silero TTS + mpv + кэш + стриминг
  fuzzy.py           нечёткое исправление STT-ошибок (WDL + фонетика)
  intent.py          Байесовский классификатор намерений
  llm.py             клиент Ollama + история диалога
  llm_tools.py       tool calling: open_site, open_app, calculate, …
  actions.py         низкоуровневые действия (xdg-open, subprocess)
  audio.py           единый микрофонный поток → BroadcastBuffer
  metrics.py         замер времени вызовов
  setup.py           интерактивный мастер настройки микрофона (--setup)
  vosk_model.py      общая загрузка модели Vosk (singleton)
  logger.py          ротация логов в logs/jarvis.log и вывод в консоль
  window_context.py  активное окно через kdotool
  silero_vad.py      ONNX-сессия Silero VAD
  commands/
    __init__.py      plan_command: интент → правила → LLM
    apps.py          запуск приложений
    system.py        время, дата, громкость, скриншот, блокировка
    sensors.py       температура CPU (lm_sensors)
    web.py           быстрые ссылки + поиск Google
    weather.py       погода через wttr.in
    hardware.py      характеристики железа (lscpu, free, /sys/class/drm)
  mathcore/          математическое ядро (чистые функции, без зависимостей)
    distance.py      взвешенный Дамерау-Левенштейн, similarity, is_metric
    phonetics.py     фонетическая нормализация русского текста
    weights.py       таблица весов редакционных операций
    bayes.py         наивный Байес, char n-граммы, margin, decide
    viterbi.py       HMM + алгоритм Витерби, гистерезис
    features.py      MFCC, мел-фильтрбанк, DCT-II, DTW
  phrases/           готовые WAV-ответы (играются мгновенно)

models/
  vosk/              модель распознавания (88 МБ)
  silero_vad.onnx    Silero VAD (1.8 МБ)

config.py            все настройки проекта в одном месте
main.py              точка входа: --setup, --devices
```

## 🗣️ Список команд

Таблица соответствует каталогу `CATALOG` в `src/intent.py`. Столбец «опасная»
означает, что для выполнения нужен более высокий порог уверенности.

| Интент | Примеры фраз | Опасная |
|--------|--------------|:-------:|
| `apps.explorer` | «проводник», «файлы», «документы» | ✓ |
| `apps.editor` | «vscode», «вс код», «вс-код» | ✓ |
| `apps.browser` | «браузер», «brave», «брейв» | ✓ |
| `apps.terminal` | «терминал», «консоль», «konsole» | ✓ |
| `apps.telegram` | «телеграм», «telegram», «тг» | ✓ |
| `web.vk` | «вк», «вконтакте», «вэ ка», «века» | ✓ |
| `web.youtube` | «ютуб», «youtube», «ютубе» | ✓ |
| `web.music` | «музыка», «музыку» | ✓ |
| `web.moodle` | «мудл» | ✓ |
| `web.schedule` | «расписание» | ✓ |
| `web.deepseek` | «дипсик», «дипсика», «дип сик» | ✓ |
| `web.mail` | «почта», «почту» | ✓ |
| `web.search` | «загугли X», «найди в гугле X» | ✓ |
| `system.time` | «который час», «сколько времени» | — |
| `system.date` | «какое сегодня число», «какая дата» | — |
| `system.lock` | «заблокируй», «блокировка экрана» | ✓ |
| `system.volume_up` | «громкость выше», «громче», «погромче» | ✓ |
| `system.volume_down` | «громкость тише», «убавь», «тише» | ✓ |
| `system.volume_mute` | «выключи звук», «без звука» | ✓ |
| `system.volume_unmute` | «включи звук», «верни звук» | ✓ |
| `system.screenshot` | «скриншот», «снимок экрана», «скрин» | ✓ |
| `sensor.cpu` | «температура процессора», «перегрев» | — |
| `weather.current` | «какая погода», «погода в городе» | — |
| `weather.forecast` | «прогноз на завтра» | — |
| `hardware.info` | «характеристики», «характеристика железа» | — |

Любая другая фраза уходит в LLM (`qwen2.5:3b` через Ollama).

## 🔧 Настройка: .env и config.py

Пользовательские значения — в `.env` (не попадает в git):

```bash
JARVIS_DEVICE=4                  # индекс микрофона
JARVIS_LLM_MODEL=qwen2.5:3b     # модель Ollama
JARVIS_LOG_LEVEL=INFO            # DEBUG — подробный лог
JARVIS_CONFIRM_SOUND=voice       # voice | earcon | off
```

Остальные параметры — в `config.py`: VAD-пороги, пути к моделям, системный промпт,
пороги классификатора интентов и флаги математического ядра.

## 🎯 Как добавить свою команду

1. Открой подходящий файл в `src/commands/` (например, `apps.py`) и добавь ветку в `try_handle`:

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

2. При включённом `INTENT_ENABLED = True` зарегистрируй новый `Intent` в `CATALOG`
   (`src/intent.py`): укажи `patterns`, флаг `dangerous` (`True` для запуска процессов,
   ссылок и смены настроек) и ключ обработчика `handler`. Иначе действия с побочным
   эффектом будут заблокированы защитой от ложных срабатываний.

## 🧮 Математическое ядро

`src/mathcore/` — слой чистых математических функций: он не импортирует ничего
из остального `src/`, не обращается к сети, микрофону и LLM. Покрытие — **99%** строк.

| Модуль | Что делает | Потребитель |
|--------|------------|-------------|
| `distance.py` | взвешенный Дамерау-Левенштейн, `similarity()`, проверка аксиом метрики | `fuzzy.py` |
| `phonetics.py` | оглушение, редукция, ассимиляция; `combined_distance()` | `fuzzy.py` |
| `weights.py` | таблица весов: б→п 0.2, о→а 0.25, й→и 0.1 и другие пары | `distance.py` |
| `bayes.py` | char n-граммы, tf/idf, cosine, `NaiveBayes`, `margin()`, `decide()` | `intent.py` |
| `viterbi.py` | HMM с бета-эмиссиями, Витерби в лог-пространстве, гистерезис | `vad.py` (`VAD_DECODER="viterbi"`, выкл. по умолчанию) |
| `features.py` | окно Ханна, мел-фильтрбанк, MFCC, дельты, DTW | `wakeword_dtw.py` (заготовка, выкл. по умолчанию) |

Подробности: [docs/THESIS_TZ.md](docs/THESIS_TZ.md) (математическая постановка),
[docs/T6_MFCC_DTW.md](docs/T6_MFCC_DTW.md) (акустика).

## 🙏 Благодарности

- [Vosk](https://alphacephei.com/vosk/) — офлайн распознавание речи
- [Silero](https://github.com/snakers4/silero-models) — офлайн TTS и VAD
- [Ollama](https://ollama.com/) — локальный LLM
- [Qwen](https://qwenlm.github.io/) — модель
- [sounddevice](https://python-sounddevice.readthedocs.io/) — захват аудио
- [kdotool](https://github.com/jinliu/kdotool) — активное окно в KDE/Wayland

## 📄 Лицензия

MIT — см. [LICENSE](LICENSE).
