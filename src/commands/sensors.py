"""Команды датчиков: температура CPU через lm_sensors.

Работает без LLM — триггер ловится в try_handle, значение читается из
`sensors` за ~0.1 сек. LLM здесь не нужна: число либо есть, либо его нет.
"""

import re
import subprocess

from src.logger import get_logger

logger = get_logger("commands.sensors")


# ──────────────────────── НАСТРОЙКИ ────────────────────────

SENSORS_CMD = "sensors"
_TIMEOUT = 3

# Строки, по которым ищем температуру CPU (Intel → AMD → fallback)
_CPU_KEYS = ("Package id", "Tctl", "Tdie", "CPU Christoph")

# Что считаем «железячным» контекстом, чтобы не путать с погодой
_CPU_WORDS = (
    "процессор", "проц", "cpu", "цпу",
    "видеокарт", "gpu", "гпу",
    "комп", "ноут", "систем", "материн",
    "желез", "конденсатор", "радиатор", "кулер", "вентилятор",
)

# Что указывает на улицу — тогда «температура» не про CPU
_OUTDOOR_WORDS = (
    "улиц", "воздух", "погод", "окн", "снаруж", "за бортом",
    "градусник", "мете", "синоп",
)

_TEMP_WORDS = ("температур", "нагрев", "перегрев", "градус", "жарит", "горяч")


# ──────────────────────── ЧТЕНИЕ ДАТЧИКОВ ────────────────────────

_TEMP_RE = re.compile(r"\+?(-?\d+(?:[.,]\d+)?)\s*°?C")


def _read_sensors() -> str | None:
    """Возвращает сырой вывод sensors или None, если команда недоступна."""
    try:
        result = subprocess.run(
            [SENSORS_CMD],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
    except FileNotFoundError:
        logger.error("sensors не установлен (пакет lm_sensors)")
        return None
    except subprocess.TimeoutExpired:
        logger.error("sensors завис (>%.1f сек)", _TIMEOUT)
        return None

    # sensors пишет предупреждения в stderr — их игнорируем,
    # useful данные всегда в stdout.
    return result.stdout


def _parse_cpu_temp(output: str) -> int | None:
    """Вытасывает температуру CPU из вывода sensors."""
    # 1. Ищем известный CPU-датчик
    for line in output.splitlines():
        if any(key in line for key in _CPU_KEYS):
            match = _TEMP_RE.search(line)
            if match:
                return int(round(float(match.group(1).replace(",", "."))))

    # 2. Fallback: блок coretemp — берём максимум по ядрам
    in_coretemp = False
    core_temps = []
    for line in output.splitlines():
        stripped = line.strip()
        if "coretemp" in stripped:
            in_coretemp = True
            continue
        if in_coretemp:
            if not stripped or stripped.endswith(":"):
                if core_temps:
                    break
                if stripped and "temp" not in stripped:
                    in_coretemp = False
                    continue
            match = _TEMP_RE.search(stripped)
            if match:
                core_temps.append(float(match.group(1).replace(",", ".")))

    if core_temps:
        return int(round(max(core_temps)))

    return None


def get_cpu_temp() -> int | None:
    """Температура CPU в градусах Цельсия, либо None."""
    output = _read_sensors()
    if output is None:
        return None
    temp = _parse_cpu_temp(output)
    if temp is None:
        logger.warning("Не нашёл датчик CPU в выводе sensors")
    return temp


# ──────────────────────── ФОРМИРОВАНИЕ ОТВЕТА ────────────────────────

def _describe_cpu_temp(temp: int) -> str:
    """Температура + короткая оценка. Пороги типичны для мобильных Intel."""
    base = f"Температура процессора {temp} градусов"

    if temp >= 95:
        return f"{base}. Критично, глуши систему"
    if temp >= 85:
        return f"{base}. Горячо, проверь охлаждение"
    if temp >= 70:
        return f"{base}. Нормально под нагрузкой"
    return f"{base}, всё в норме"


# ──────────────────────── HANDLER ────────────────────────

def _is_cpu_temperature(text: str) -> bool:
    """Похожа ли фраза на вопрос о температуре железа, а не воздуха."""
    if not any(w in text for w in _TEMP_WORDS):
        return False
    if any(w in text for w in _OUTDOOR_WORDS):
        return False
    return any(w in text for w in _CPU_WORDS)


def try_handle(text: str):
    if not _is_cpu_temperature(text):
        return None

    logger.info("Читаю температуру CPU...")
    temp = get_cpu_temp()

    if temp is None:
        return "Не могу прочитать датчики температуры", None, None

    return _describe_cpu_temp(temp), None, None
