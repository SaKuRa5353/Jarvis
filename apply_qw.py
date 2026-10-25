#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Применяет Quick Wins 1, 2, 3, 5, 6 к проекту Джарвис.

Запуск из КОРНЯ проекта:
    python apply_qw.py --check     # только проверить, что все якоря находятся
    python apply_qw.py             # применить + py_compile
    python apply_qw.py --revert    # откатить (бэкапы *.qwbak, новые файлы удалить)

Каждая правка ищется по точному якорю. Если якорь не найден или встречается
несколько раз — правка ПРОПУСКАЕТСЯ с сообщением. Молча ничего не ломается.

  QW6 логи      LOG_* в config, RotatingFileHandler 5MB x 5 вместо
                TimedRotatingFileHandler(utc=True) — ротация съезжала на
                часовой пояс, а backupCount=2 противоречил docstring про 7 дней
  QW5 метрики   отчёт при любом выходе (atexit + KeyboardInterrupt в main.py),
                log_report(reset=) чтобы не печатать дважды
  QW1 --setup   python main.py --setup: список микрофонов -> .env (JARVIS_DEVICE),
                тест записи 3 с; DEVICE больше не захардкожен
  QW3 токены    LLM_MAX_TOKENS подключён (был объявлен и НЕ использовался:
                llm.py всегда брал LLM_TOOLS_MAX_TOKENS), + warning при
                finish_reason == "length"
  QW2 earcons   CONFIRM_SOUND=voice|earcon|off, генератор tests/generate_earcons.py
                (по умолчанию voice — текущее поведение не меняется)
  .gitignore    убран test_*.py (иначе будущие pytest-тесты не попадут в git)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRY = "--check" in sys.argv
REVERT = "--revert" in sys.argv

R: list[tuple[str, str, str]] = []
RX: list[tuple[str, str, str]] = []   # (файл, regex, замена) — для плавающих хвостов


def rep(path: str, old: str, new: str) -> None:
    R.append((path, old, new))


def rex(path: str, pattern: str, new: str) -> None:
    RX.append((path, pattern, new))


# DEVICE = 4  + произвольный хвост-комментарий в конце строки
rex("config.py",
    r"DEVICE = 4[^\n]*",
    '# Захардкод был здесь. Теперь: .env -> JARVIS_DEVICE, fallback 4.\n'
    '# Перенастроить микрофон:  python main.py --setup\n'
    'DEVICE = _env_int("JARVIS_DEVICE", 4)')


# ═══════════════════════════════ config.py ═══════════════════════════════
rep("config.py",
    "from pathlib import Path\n",
    'import os\n'
    'from pathlib import Path\n'
    '\n'
    '# .env рядом с проектом: JARVIS_DEVICE, JARVIS_LOG_LEVEL, JARVIS_LOG_DIR.\n'
    '# Файл в .gitignore — значения у каждого свои. Создать: python main.py --setup\n'
    'try:\n'
    '    from dotenv import load_dotenv\n'
    'except ImportError:  # python-dotenv есть в requirements.txt, но вдруг\n'
    '    def load_dotenv(*_a, **_kw) -> bool:\n'
    '        return False\n'
    '\n'
    'load_dotenv()\n'
    '\n'
    '\n'
    'def _env_int(name: str, default: int) -> int:\n'
    '    """Целое из окружения; при мусоре — default с предупреждением."""\n'
    '    raw = os.getenv(name)\n'
    '    if raw is None or not raw.strip():\n'
    '        return default\n'
    '    try:\n'
    '        return int(raw.strip())\n'
    '    except ValueError:\n'
    '        print(f"[config] {name}={raw!r} — не число, беру {default}")\n'
    '        return default\n')


CONFIG_TAIL = '''

# ──────────────────────── ЛОГИРОВАНИЕ ────────────────────────
# Файл: logs/jarvis.log. Ротация по РАЗМЕРУ (предсказуемо на диске),
# уровень файла JARVIS_FILE_LOG_LEVEL, консоль JARVIS_LOG_LEVEL.

LOG_DIR = Path(os.getenv("JARVIS_LOG_DIR", "logs"))
LOG_FILE_NAME = "jarvis.log"
LOG_MAX_BYTES = 5 * 1024 * 1024     # 5 МБ на файл
LOG_BACKUP_COUNT = 5                # хранить 5 архивов
LOG_LEVEL = os.getenv("JARVIS_LOG_LEVEL", "INFO").upper()
LOG_FILE_LEVEL = os.getenv("JARVIS_FILE_LOG_LEVEL", "INFO").upper()

# ──────────────────────── ЗВУКИ ПОДТВЕРЖДЕНИЯ ────────────────────────
# voice   — готовые речевые фразы из src/phrases/ (текущее поведение)
# earcon  — короткий «динь» 180 мс: быстрее на ~0.5 с и меньше эха в микрофон
# off     — тишина
# Создать earcon-WAV:  python tests/generate_earcons.py

CONFIRM_SOUND = os.getenv("JARVIS_CONFIRM_SOUND", "voice").lower()
EARCON_TAIL_SEC = 0.03              # пауза после короткого звука (эхо-хвост)

# ──────────────────────── ИСТОРИЯ ДИАЛОГА ────────────────────────
# Заготовка для персиста (QW4). Сам персист включается отдельно.

HISTORY_DIR = Path(os.getenv("JARVIS_HISTORY_DIR", "data/history"))
HISTORY_ENABLED = os.getenv("JARVIS_HISTORY_ENABLED", "0") == "1"
HISTORY_RESTORE_TURNS = 6
'''

rep("config.py",
    'Не расслышал»."\n)\n',
    'Не расслышал»."\n)\n' + CONFIG_TAIL)

rep("config.py",
    "LLM_MAX_TOKENS = 30",
    '# Бюджет текстового ответа. ВНИМАНИЕ: до quick wins эта константа была\n'
    '# объявлена, но НЕ использовалась — llm.py всегда брал LLM_TOOLS_MAX_TOKENS.\n'
    '# Теперь лимит = LLM_TOOLS_MAX_TOKENS при включённых tools, иначе этот.\n'
    'LLM_MAX_TOKENS = 30')

# ═════════════════════════════ src/logger.py ═════════════════════════════
rep("src/logger.py",
    "import logging\nimport sys\nfrom logging.handlers import TimedRotatingFileHandler\nfrom pathlib import Path\n",
    "import logging\nimport os\nimport sys\nfrom logging.handlers import RotatingFileHandler, TimedRotatingFileHandler\n")

rep("src/logger.py",
    'LOG_DIR = Path("logs")\nLOG_FILE = LOG_DIR / "jarvis.log"\n',
    "from config import (\n"
    "    LOG_BACKUP_COUNT,\n"
    "    LOG_DIR,\n"
    "    LOG_FILE_NAME,\n"
    "    LOG_MAX_BYTES,\n"
    ")\n"
    "\n"
    "LOG_FILE = LOG_DIR / LOG_FILE_NAME\n")

rep("src/logger.py",
    'def setup_logging(level: int = logging.INFO):\n'
    '    """Настраивает root logger. Вызывать один раз при старте."""\n'
    "    LOG_DIR.mkdir(parents=True, exist_ok=True)\n",
    "def setup_logging(level: int | str | None = None):\n"
    '    """Настраивает root logger. Вызывать один раз при старте.\n'
    "\n"
    "    level=None → JARVIS_LOG_LEVEL из окружения (по умолчанию INFO).\n"
    '    Путь к файлу лога печатается в консоль, чтобы его не искать.\n'
    '    """\n'
    "    if level is None:\n"
    '        level = os.getenv("JARVIS_LOG_LEVEL", "INFO").upper()\n'
    "    if isinstance(level, str):\n"
    "        level = getattr(logging, level.upper(), logging.INFO)\n"
    "\n"
    "    LOG_DIR.mkdir(parents=True, exist_ok=True)\n")

rep("src/logger.py",
    "    file_handler = TimedRotatingFileHandler(\n"
    "        LOG_FILE,\n"
    '        when="midnight",        # каждый день в полночь\n'
    "        backupCount=2,          # хранить 2 архива\n"
    '        encoding="utf-8",\n'
    "        utc=True,\n"
    "    )\n"
    "    file_handler.setLevel(logging.INFO)    # INFO вместо DEBUG\n",
    "    # Ротация по размеру: jarvis.log.1 ... jarvis.log.N\n"
    '    # (Раньше было TimedRotatingFileHandler(when="midnight", utc=True) —\n'
    "    #  из-за utc=True граница суток съезжала на часовой пояс, а backupCount=2\n"
    '    #  не соответствовал docstring про «7 дней».)\n'
    "    file_handler = RotatingFileHandler(\n"
    "        LOG_FILE,\n"
    "        maxBytes=LOG_MAX_BYTES,\n"
    "        backupCount=LOG_BACKUP_COUNT,\n"
    '        encoding="utf-8",\n'
    "    )\n"
    '    file_handler.setLevel(os.getenv("JARVIS_FILE_LOG_LEVEL", "INFO").upper())\n')

rep("src/logger.py",
    "    root.addHandler(file_handler)\n",
    "    root.addHandler(file_handler)\n"
    '    print(f"[logs] {LOG_FILE.resolve()}  "\n'
    '          f"(уровень файла {logging.getLevelName(file_handler.level)}, "\n'
    '          f"{LOG_MAX_BYTES // (1024 * 1024)} МБ x {LOG_BACKUP_COUNT})")\n')

rep("src/logger.py",
    "        # ── Приглушаем шумные библиотеки ──",
    "    # ── Приглушаем шумные библиотеки ──")

# ═══════════════════════════════ main.py ═══════════════════════════════
MAIN_NEW = '''"""Точка входа Джарвиса.

    python main.py              запуск ассистента
    python main.py --setup      мастер настройки микрофона (пишет .env)
    python main.py --devices    просто список устройств ввода

JARVIS_LOG_LEVEL=DEBUG python main.py — подробный лог в консоль.
"""

import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

from src.logger import setup_logging

setup_logging()


def _main() -> int:
    # Тяжёлые импорты (torch, silero, vosk) — ТОЛЬКО после разбора аргументов,
    # иначе мастер настройки открывается несколько секунд.
    if "--setup" in sys.argv or "--devices" in sys.argv:
        from src.setup import run_setup
        return run_setup(list_only="--devices" in sys.argv)

    import atexit

    from src.assistant import run
    from src.metrics import metrics

    def _final_report():
        try:
            metrics.log_report(reset=False)
        except Exception:
            pass

    # Отчёт метрик при любом выходе, включая SIGTERM от systemd.
    atexit.register(_final_report)

    try:
        run()
    except KeyboardInterrupt:
        # Раньше Ctrl-C ловился только внутри главного цикла: прерывание на
        # старте (нет Ollama, занят DEVICE) падало с необработанным traceback.
        metrics.log_report()
        print("\\n\U0001F44B Выход")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
'''

rep("main.py",
    '"""Точка входа Джарвиса."""\n'
    "\n"
    "import warnings\n"
    "\n"
    'warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")\n'
    "\n"
    "from src.logger import setup_logging\n"
    "\n"
    "setup_logging()\n"
    "\n"
    "from src.assistant import run\n"
    "\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    run()",
    MAIN_NEW.rstrip("\n"))

# ════════════════════════════ src/assistant.py ════════════════════════════
rep("src/assistant.py",
    "import threading\nimport time\nimport traceback\n",
    "import atexit\nimport threading\nimport time\nimport traceback\n")

rep("src/assistant.py",
    '    logger.info("         ДЖАРВИС ЗАПУЩЕН")\n'
    '    logger.info("=" * 50)\n',
    '    logger.info("         ДЖАРВИС ЗАПУЩЕН")\n'
    '    logger.info("=" * 50)\n'
    "\n"
    "    # Отчёт метрик при любом выходе из процесса (в т.ч. SIGTERM, SystemExit).\n"
    "    # reset=False: KeyboardInterrupt в main.py уже мог напечатать отчёт,\n"
    "    # а повторная печать очищенного реестра ничего не даёт.\n"
    "    atexit.register(lambda: metrics.log_report(reset=False))\n")

# ═════════════════════════════ src/metrics.py ═════════════════════════════
rep("src/metrics.py",
    "    def log_report(self, level: int = logging.INFO) -> None:\n"
    '        """Логгирует отчёт по метрикам."""\n'
    "        logger.log(level, self.get_report())\n",
    "    def log_report(self, level: int = logging.INFO, reset: bool = True) -> None:\n"
    '        """Логгирует отчёт по метрикам.\n'
    "\n"
    "        reset=True (по умолчанию) очищает реестр после печати, чтобы отчёт\n"
    '        не дублировался при втором вызове (например, atexit после Ctrl-C).\n'
    '        """\n'
    "        if not reset and not self._metrics:\n"
    "            return  # отчёт уже напечатан и реестр очишен — не дублируем\n"
    "        logger.log(level, self.get_report())\n"
    "        if reset:\n"
    "            self.reset()\n")

rep("src/metrics.py",
    """            f"  {'метрика':32} | вызовы | ср.время | мин | макс | всего\"""",
    """            f"  {'метрика':44} | вызовы | ср.время | мин | макс | всего\"""")

rep("src/metrics.py",
    """            f"  {name:32} | {entry.call_count:6d} | \"""",
    """            f"  {name:44} | {entry.call_count:6d} | \"""")

# ═══════════════════════════════ src/llm.py ═══════════════════════════════
rep("src/llm.py",
    "from config import (\n    LLM_MODEL,\n",
    "from config import (\n    LLM_MAX_TOKENS,\n    LLM_MODEL,\n")

rep("src/llm.py",
    "        request = dict(\n"
    "            model=self.model,\n"
    "            messages=messages,\n"
    "            temperature=LLM_TEMPERATURE,\n"
    "            max_tokens=LLM_TOOLS_MAX_TOKENS,\n"
    "            tools=TOOLS,\n"
    '            tool_choice="auto",\n'
    '            extra_body={"options": {"num_thread": LLM_NUM_THREADS}},\n'
    "        )\n",
    "        # tools передаются всегда → фактический лимит LLM_TOOLS_MAX_TOKENS.\n"
    "        # LLM_MAX_TOKENS применяется, только если tools отключат.\n"
    "        # (До quick wins LLM_MAX_TOKENS не использовался вовсе.)\n"
    "        max_tokens = LLM_TOOLS_MAX_TOKENS if TOOLS else LLM_MAX_TOKENS\n"
    "\n"
    "        request = dict(\n"
    "            model=self.model,\n"
    "            messages=messages,\n"
    "            temperature=LLM_TEMPERATURE,\n"
    "            max_tokens=max_tokens,\n"
    "            tools=TOOLS,\n"
    '            tool_choice="auto",\n'
    '            extra_body={"options": {"num_thread": LLM_NUM_THREADS}},\n'
    "        )\n"
    "        finish_reason: str | None = None\n")

rep("src/llm.py",
    "            if on_delta is None:\n"
    "                msg = self.client.chat.completions.create(**request).choices[0].message\n"
    '                tool_calls = getattr(msg, "tool_calls", None)\n'
    "                raw = msg.content\n"
    "            else:\n"
    "                raw, tool_calls = self._ask_streaming(on_delta=on_delta, **request)\n",
    "            if on_delta is None:\n"
    "                choice = self.client.chat.completions.create(**request).choices[0]\n"
    "                msg = choice.message\n"
    '                tool_calls = getattr(msg, "tool_calls", None)\n'
    "                raw = msg.content\n"
    '                finish_reason = getattr(choice, "finish_reason", None)\n'
    "            else:\n"
    "                raw, tool_calls, finish_reason = self._ask_streaming(\n"
    "                    on_delta=on_delta, **request\n"
    "                )\n")

rep("src/llm.py",
    "        answer = raw.strip()\n"
    "\n"
    "        # Не сохраняем мусорные ответы",
    "        answer = raw.strip()\n"
    "\n"
    "        # Ответ, обрезанный по max_tokens, озвучивать как полноценный нельзя:\n"
    '        # половина фразы звучит как ошибка распознавания.\n'
    '        if finish_reason == "length":\n'
    "            logger.warning(\n"
    '                "[LLM] ответ оборван по max_tokens=%d: %r",\n'
    '                request.get("max_tokens", 0), answer[-60:],\n'
    "            )\n"
    "\n"
    "        # Не сохраняем мусорные ответы")

# Один блок вместо трёх пересекающихся (иначе второй якорь исчезает после первой правки)
rep("src/llm.py",
    "    def _ask_streaming(self, on_delta, **request) -> tuple[str | None, list | None]:\n",
    "    def _ask_streaming(self, on_delta, **request) -> tuple[str | None, list | None, str | None]:\n")

rep("src/llm.py",
    "        parts: list[str] = []\n"
    "        calls: dict[int, dict] = {}\n"
    "\n"
    "        for chunk in self.client.chat.completions.create(**request, stream=True):\n"
    "            if not chunk.choices:\n"
    "                continue\n"
    "            delta = chunk.choices[0].delta\n"
    "            if delta is None:\n"
    "                continue\n",
    "        parts: list[str] = []\n"
    "        calls: dict[int, dict] = {}\n"
    "        finish: str | None = None\n"
    "\n"
    "        for chunk in self.client.chat.completions.create(**request, stream=True):\n"
    "            if not chunk.choices:\n"
    "                continue\n"
    "            choice = chunk.choices[0]\n"
    "            delta = choice.delta\n"
    '            if getattr(choice, "finish_reason", None):\n'
    "                finish = choice.finish_reason\n"
    "            if delta is None:\n"
    "                continue\n")

rep("src/llm.py",
    "            return None, [call]\n"
    '        return "".join(parts), None\n',
    "            return None, [call], finish\n"
    '        return "".join(parts), None, finish\n')

# ═══════════════════════════════ src/tts.py ═══════════════════════════════
rep("src/tts.py",
    "from config import (\n    SILERO_MODEL,\n",
    "from config import (\n    CONFIRM_SOUND,\n    EARCON_TAIL_SEC,\n    SILERO_MODEL,\n")

rep("src/tts.py",
    "from src.phrases import get_phrase_path, PHRASES\n",
    "from src.phrases import PHRASES, get_phrase_path, resolve_phrase_key\n")

rep("src/tts.py",
    "# ──────────────────────── ВОСПРОИЗВЕДЕНИЕ ────────────────────────\n",
    "def _echo_tail(path: str | Path) -> float:\n"
    '    """Пауза после воспроизведения, чтобы эхо не попало в микрофон.\n'
    "\n"
    "    Для коротких earcon (~17 КБ) хвост не нужен — экономия ~0.2 с на каждом\n"
    '    подтверждении; для секундных речевых фраз оставляем прежние 200 мс.\n'
    '    """\n'
    "    try:\n"
    "        size = Path(path).stat().st_size\n"
    "    except OSError:\n"
    "        return 0.2\n"
    "    return EARCON_TAIL_SEC if size < 30_000 else 0.2\n"
    "\n"
    "\n"
    "# ──────────────────────── ВОСПРОИЗВЕДЕНИЕ ────────────────────────\n")

rep("src/tts.py",
    "        time.sleep(0.2)  # «хвост» эха в комнате/звуковом буфере\n",
    "        time.sleep(_echo_tail(path))  # «хвост» эха в комнате/звуковом буфере\n")

rep("src/tts.py",
    'def speak_phrase(key: str) -> bool:\n'
    '    """Проигрывает готовую фразу из src/phrases/. True если файл найден."""\n'
    "    path = get_phrase_path(key)\n"
    "    if path is None or not path.exists():\n"
    "        return False\n",
    'def speak_phrase(key: str) -> bool:\n'
    '    """Проигрывает готовую фразу из src/phrases/. True если файл найден.\n'
    "\n"
    '    При CONFIRM_SOUND="earcon" подтверждающие фразы (welcome, llm_thinking,\n'
    '    done) заменяются коротким звуком; при "off" не звучат вовсе.\n'
    "    Остальные ключи (not_listened, error, volume_*) играют как раньше.\n"
    '    """\n'
    "    key = resolve_phrase_key(key)\n"
    "    if key is None:\n"
    "        return False\n"
    "    path = get_phrase_path(key)\n"
    "    if path is None or not path.exists():\n"
    "        return False\n")

# ═════════════════════════ src/phrases/__init__.py ═════════════════════════
rep("src/phrases/__init__.py",
    '    # ── LLM ──\n'
    '    "llm_thinking": ("llm_thinking.wav", "Думаю"),\n'
    "}\n",
    '    # ── LLM ──\n'
    '    "llm_thinking": ("llm_thinking.wav", "Думаю"),\n'
    "\n"
    '    # ── Earcons: короткие «динь» вместо речевых фраз ──\n'
    '    # Включаются CONFIRM_SOUND="earcon" в config.py (или JARVIS_CONFIRM_SOUND).\n'
    "    # Создать файлы: python tests/generate_earcons.py\n"
    '    "earcon_welcome": ("earcon_welcome.wav", "[динь]"),\n'
    '    "earcon_thinking": ("earcon_thinking.wav", "[динь-думает]"),\n'
    '    "earcon_done": ("earcon_done.wav", "[динь-готово]"),\n'
    "}\n"
    "\n"
    '# Какие фразы считаются «подтверждением» и заменяются earcon.\n'
    "# not_listened / error / volume_* — содержательные, их не трогаем.\n"
    "CONFIRM_KEYS = {\n"
    '    "welcome": "earcon_welcome",\n'
    '    "llm_thinking": "earcon_thinking",\n'
    '    "done": "earcon_done",\n'
    "}\n")

rep("src/phrases/__init__.py",
    'def get_phrase_path(key: str) -> Path | None:\n'
    '    """Возвращает путь к WAV-файлу или None."""\n'
    "    entry = PHRASES.get(key)\n"
    "    if entry is None:\n"
    "        return None\n"
    "    return PHRASES_DIR / entry[0]",
    'def get_phrase_path(key: str) -> Path | None:\n'
    '    """Возвращает путь к WAV-файлу или None."""\n'
    "    entry = PHRASES.get(key)\n"
    "    if entry is None:\n"
    "        return None\n"
    "    return PHRASES_DIR / entry[0]\n"
    "\n"
    "\n"
    "def resolve_phrase_key(key: str) -> str | None:\n"
    '    """Ключ фразы с учётом CONFIRM_SOUND; None — если звук выключен.\n'
    "\n"
    "    Если earcon-файла ещё нет (не сгенерирован), возвращает исходный ключ:\n"
    '    ассистент не онемеет из-за отсутствующего WAV.\n'
    '    """\n'
    "    from config import CONFIRM_SOUND  # локально: config не должен зависеть от нас\n"
    "\n"
    '    mode = (CONFIRM_SOUND or "voice").lower()\n'
    '    if mode == "off" and key in CONFIRM_KEYS:\n'
    "        return None\n"
    '    if mode == "earcon" and key in CONFIRM_KEYS:\n'
    "        earcon = CONFIRM_KEYS[key]\n"
    "        if (PHRASES_DIR / PHRASES[earcon][0]).exists():\n"
    "            return earcon\n"
    "    return key")

# ══════════════════════════════ .gitignore ══════════════════════════════
rep(".gitignore", "test_*.py\n", "")

# ══════════════════════════ tests/check_stt.py ══════════════════════════
rep("tests/check_stt.py",
    "from src.stt import VoskSTT\n",
    "from config import DEVICE, SAMPLE_RATE  # noqa: E402\nfrom src.stt import VoskSTT\n")

rep("tests/check_stt.py",
    "DURATION = 5\nSR = 16000\nDEVICE = 4\n",
    "DURATION = 5\nSR = SAMPLE_RATE  # из config, а не литералом\n")

# ------------------------------------------------------------ новые файлы --
NEW: dict[str, str] = {}

NEW["src/setup.py"] = '''"""Мастер настройки микрофона: python main.py --setup

Показывает только пригодные устройства ввода (реальные карты, без ALSA-виртуалов
и HDMI без входа), пишет выбор в .env как JARVIS_DEVICE и предлагает тестовую
запись 3 секунды с пиковой амплитудой.

Зачем: DEVICE был захардкожен как 4 в config.py и отдельным литералом
в tests/check_stt.py — на другой машине проект не заводился.
"""

from __future__ import annotations

from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
ENV_KEY = "JARVIS_DEVICE"

# ALSA-виртуалы и агрегаторы: их в списке sounddevice больше десятка, толку нет.
_SKIP_NAMES = {
    "sysdefault", "front", "surround40", "surround51", "surround71",
    "hdmi", "lavrate", "samplerate", "speexrate", "speex", "upmix",
    "vdownmix", "dmix", "pipewire", "pulse", "default",
}


def list_input_devices() -> list[tuple[int, str, int]]:
    """(индекс, имя, число входов) — только устройства с микрофонным входом."""
    import sounddevice as sd

    out: list[tuple[int, str, int]] = []
    for i, d in enumerate(sd.query_devices()):
        if int(d.get("max_input_channels", 0)) <= 0:
            continue
        name = str(d.get("name", ""))
        base = name.split(":")[0].strip().lower()
        if base in _SKIP_NAMES or name.strip().lower() in _SKIP_NAMES:
            continue
        out.append((i, name, int(d["max_input_channels"])))
    return out


def _print_devices(devs: list[tuple[int, str, int]]) -> None:
    print("\\nПригодные устройства ввода:\\n")
    for idx, name, ch in devs:
        print(f"  [{idx:>2}] {name}  ({ch} in)")
    print()


def _write_env(device: int) -> None:
    lines: list[str] = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if not ln.strip().startswith(f"{ENV_KEY}=")]
    lines.append(f"{ENV_KEY}={device}")
    ENV_FILE.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    print(f"\\n\u2705 {ENV_KEY}={device} записан в {ENV_FILE}")
    print("   (.env уже в .gitignore — в репозиторий не попадёт)")


def _record_test(device: int) -> bool:
    print("\\n\U0001F3A4 Скажите что-нибудь 3 секунды...")
    try:
        import numpy as np
        import sounddevice as sd

        from config import SAMPLE_RATE

        rec = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                     channels=1, dtype="int16", device=device)
        sd.wait()
        amp = int(np.abs(rec).max())
    except Exception as e:
        print(f"   \u26A0\uFE0F Запись не удалась: {e}")
        return False

    print(f"   Max amplitude: {amp}")
    if amp < 500:
        print("   \u274C Слишком тихо. pavucontrol \u2192 Input Devices \u2192 поднять усиление.")
        return False
    print("   \u2705 Микрофон слышит вас.")
    return True


def run_setup(list_only: bool = False) -> int:
    try:
        import sounddevice as sd  # noqa: F401
    except Exception as e:
        print(f"\u274C sounddevice недоступен: {e}")
        print("   pip install sounddevice   (нужен системный PortAudio)")
        return 1

    devs = list_input_devices()
    if not devs:
        print("\u274C Устройств ввода не найдено. Проверьте микрофон и PortAudio.")
        return 1

    _print_devices(devs)
    if list_only:
        return 0

    try:
        from config import DEVICE as current
        print(f"Сейчас в конфиге: DEVICE={current}\\n")
    except Exception:
        current = None

    valid = {str(i) for i, _, _ in devs}
    raw = input("Номер микрофона (Enter = оставить текущий): ").strip()
    if not raw:
        if current is None:
            print("Текущего значения нет — укажите номер.")
            return 1
        print(f"Оставляем DEVICE={current}")
        chosen = int(current)
    elif raw not in valid:
        print(f"\u274C {raw!r} не в списке {sorted(valid, key=int)}")
        return 1
    else:
        chosen = int(raw)
        _write_env(chosen)

    if input("\\nПроверить запись 3 секунды? [Y/n] ").strip().lower() in ("", "y", "\u0434"):
        _record_test(chosen)

    print("\\nГотово. Запуск: python main.py")
    return 0
'''

NEW["tests/generate_earcons.py"] = '''"""Генерация коротких звуков подтверждения (earcons) в src/phrases/.

    python tests/generate_earcons.py

Зачем: earcon 180 мс вместо речевой фразы ~0.6-1.1 с = минус ~0.5 с на каждом
цикле и заметно меньше эха в микрофон (реже самосрабатывание wakeword на
собственный TTS). Синтез они НЕ экономят — готовые WAV-фразы уже есть.

Без сторонних зависимостей: wave + math + struct.
Файлы попадают в git (.gitignore содержит !src/phrases/*.wav).
"""

from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SILERO_SAMPLE_RATE  # noqa: E402
from src.phrases import PHRASES_DIR  # noqa: E402

# имя файла -> (частоты по сегментам, длительность мс, громкость 0..1)
TONES: dict[str, tuple[tuple[float, ...], int, float]] = {
    "earcon_welcome.wav": ((880.0,), 180, 0.35),
    "earcon_thinking.wav": ((660.0, 990.0), 220, 0.30),
    "earcon_done.wav": ((990.0, 1320.0), 200, 0.35),
}


def make_tone(freqs: tuple[float, ...], ms: int, gain: float, sr: int) -> bytes:
    n = int(sr * ms / 1000)
    seg = max(1, n // len(freqs))
    fade = int(sr * 0.010)  # 10 мс на фронты — иначе щёлкает
    frames = bytearray()
    for i in range(n):
        f = freqs[min(i // seg, len(freqs) - 1)]
        env = 1.0
        if i < fade:
            env = i / fade
        elif i > n - fade:
            env = max(0.0, (n - i) / fade)
        val = int(gain * env * 32767 * math.sin(2 * math.pi * f * i / sr))
        frames += struct.pack("<h", max(-32768, min(32767, val)))
    return bytes(frames)


def main() -> int:
    PHRASES_DIR.mkdir(parents=True, exist_ok=True)
    sr = int(SILERO_SAMPLE_RATE)
    for name, (freqs, ms, gain) in TONES.items():
        path = PHRASES_DIR / name
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(make_tone(freqs, ms, gain, sr))
        print(f"  \u2705 {path}  {path.stat().st_size / 1024:.1f} КБ  {ms} мс  {freqs}")
    print('\\nВключить: CONFIRM_SOUND = "earcon" в config.py')
    print("         (или JARVIS_CONFIRM_SOUND=earcon в .env)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

# ------------------------------------------------------------------ движок --
TARGETS = sorted({f for f, _, _ in R} | {f for f, _, _ in RX} | set(NEW))


def apply() -> int:
    errors = 0
    for rel in TARGETS:
        p = ROOT / rel
        is_new = rel in NEW
        exists = p.exists()

        if not exists and not is_new:
            print(f"\u26A0 {rel}: файл не найден — правки пропущены")
            errors += 1
            continue

        orig = p.read_text(encoding="utf-8") if exists else ""
        text = orig
        done = skipped = already = 0

        for f, pat, new in RX:
            if f != rel:
                continue
            found = re.findall(pat, text)
            if len(found) == 1:
                if not DRY:
                    text = re.sub(pat, lambda _m: new, text, count=1)
                done += 1
            elif not found and new in text:
                already += 1
            else:
                print(f"\u26A0 {rel}: regex {pat!r} нашёл {len(found)} совпадений (нужно 1)")
                skipped += 1
                errors += 1

        for f, old, new in R:
            if f != rel:
                continue
            n = text.count(old)
            head = old.strip().splitlines()[0][:58] if old.strip() else "(пусто)"
            if n == 1:
                if not DRY:
                    text = text.replace(old, new, 1)
                done += 1
            elif n == 0 and new and new in text:
                already += 1          # правка уже применена ранее
            elif n == 0 and not new and old.strip() and old.strip() not in text:
                already += 1          # удаление строки: её уже нет
            elif n == 0:
                print(f"\u26A0 {rel}: якорь НЕ НАЙДЕН -> {head!r}")
                skipped += 1
                errors += 1
            else:
                print(f"\u26A0 {rel}: якорь найден {n} раза (нужен 1) -> {head!r}")
                skipped += 1
                errors += 1

        if is_new:
            print(f"  + {rel}: новый файл ({len(NEW[rel].splitlines())} строк)")
            if not DRY:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(NEW[rel], encoding="utf-8")

        if text != orig:
            if not DRY:
                shutil.copy2(p, p.with_name(p.name + ".qwbak"))
                p.write_text(text, encoding="utf-8")
            print(f"  \u2713 {rel}: применено {done}, уже было {already}, пропущено {skipped}")
        elif done == 0 and not is_new:
            print(f"  \u00B7 {rel}: без изменений (уже применено {already}, пропущено {skipped})")

    print()
    print("DRY RUN — ничего не записано" if DRY else "Записано. Бэкапы: *.qwbak")
    return 1 if errors else 0


def revert() -> int:
    n = 0
    for bak in sorted(ROOT.rglob("*.qwbak")):
        target = bak.with_name(bak.name[: -len(".qwbak")])
        shutil.move(str(bak), str(target))
        print(f"  \u21A9 {target.relative_to(ROOT)}")
        n += 1
    for rel in NEW:
        p = ROOT / rel
        if p.exists():
            p.unlink()
            print(f"  \u2717 {rel} (новый файл удалён)")
            n += 1
    print(f"Откачено: {n}")
    return 0


def verify() -> int:
    files = [ROOT / r for r in TARGETS
             if (ROOT / r).suffix == ".py" and (ROOT / r).exists()]
    bad = 0
    for f in files:
        r = subprocess.run([sys.executable, "-m", "py_compile", str(f)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            bad += 1
            print(f"\u274C {f.relative_to(ROOT)}: {r.stderr.strip()[:400]}")
    print(f"\npy_compile: {len(files) - bad}/{len(files)} OK")
    return 1 if bad else 0


if __name__ == "__main__":
    if REVERT:
        raise SystemExit(revert())
    code = apply()
    if not DRY:
        code |= verify()
    raise SystemExit(code)
