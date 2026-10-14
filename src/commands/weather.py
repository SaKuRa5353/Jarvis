"""Команда: погода через wttr.in (бесплатно, без API-ключей)."""

import json
import re
import subprocess

from src.logger import get_logger

logger = get_logger("commands.weather")


# ──────────────────────── КОНСТАНТЫ ────────────────────────

_WTTR_URL = "https://wttr.in/{city}?format=j1&lang=ru&m"
_TIMEOUT = 10
DEFAULT_CITY = "Петрозаводск"


# Словарь переводов английских описаний погоды
_WEATHER_TRANSLATIONS = {
    "sunny": "солнечно",
    "clear": "ясно",
    "partly cloudy": "переменная облачность",
    "cloudy": "облачно",
    "overcast": "пасмурно",
    "mist": "туман",
    "fog": "туман",
    "freezing fog": "ледяной туман",
    "patchy rain nearby": "местами небольшой дождь",
    "patchy rain possible": "возможен небольшой дождь",
    "light rain": "небольшой дождь",
    "light rain shower": "небольшой ливень",
    "moderate rain": "умеренный дождь",
    "moderate rain at times": "временами умеренный дождь",
    "heavy rain": "сильный дождь",
    "heavy rain at times": "временами сильный дождь",
    "light drizzle": "слабая морось",
    "freezing drizzle": "ледяная морось",
    "light freezing rain": "небольшой ледяной дождь",
    "moderate or heavy freezing rain": "умеренный или сильный ледяной дождь",
    "light snow": "небольшой снег",
    "moderate snow": "умеренный снег",
    "heavy snow": "сильный снег",
    "patchy snow possible": "возможен небольшой снег",
    "blowing snow": "метель",
    "blizzard": "метель",
    "sleet": "мокрый снег",
    "light sleet": "небольшой мокрый снег",
    "moderate or heavy sleet": "умеренный или сильный мокрый снег",
    "thundery outbreaks possible": "возможны грозы",
    "patchy light rain with thunder": "местами небольшой дождь с грозой",
    "moderate or heavy rain with thunder": "умеренный или сильный дождь с грозой",
    "torrential rain shower": "ливень",
    "light rain with thunder": "небольшой дождь с грозой",
}


# ──────────────────────── ЗАПРОС ────────────────────────

def _fetch_weather(city: str) -> dict | None:
    url = _WTTR_URL.format(city=city)
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", str(_TIMEOUT), url],
            capture_output=True, text=True, timeout=_TIMEOUT + 2, check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("wttr.in вернул пусто для города '%s'", city)
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.error("Ошибка запроса погоды: %s", e)
        return None


# ──────────────────────── ПАРСИНГ ────────────────────────

def _get_city(data: dict, fallback: str) -> str:
    try:
        return data["nearest_area"][0]["areaName"][0]["value"]
    except (KeyError, IndexError):
        return fallback


def _translate_description(desc: str) -> str:
    """Переводит английское описание в русское через словарь."""
    if not desc:
        return ""
    key = desc.strip().lower()
    # Точное совпадение
    if key in _WEATHER_TRANSLATIONS:
        return _WEATHER_TRANSLATIONS[key]
    # Частичное совпадение
    for eng, rus in _WEATHER_TRANSLATIONS.items():
        if eng in key:
            return rus
    # Не нашли — возвращаем как есть
    return desc.strip()


def _get_description(condition: dict) -> str:
    """Берёт русское описание из lang_ru, если есть. Иначе переводит."""
    try:
        ru = condition.get("lang_ru")
        if ru and ru[0].get("value"):
            val = ru[0]["value"].strip()
            if val:
                return val
    except (KeyError, IndexError, TypeError):
        pass

    # Fallback — перевод английского
    try:
        eng = condition["weatherDesc"][0]["value"].strip()
        return _translate_description(eng)
    except (KeyError, IndexError):
        return ""


def _format_wind(kmph: str) -> str:
    try:
        ms = round(int(kmph) / 3.6)
        return f"{ms} м/с"
    except (ValueError, TypeError):
        return ""


def _format_current(data: dict, fallback_city: str) -> str:
    try:
        cur = data["current_condition"][0]
    except (KeyError, IndexError):
        return "Не удалось получить данные о погоде"

    city = _get_city(data, fallback_city)
    temp = cur.get("temp_C", "?")
    feels = cur.get("FeelsLikeC", temp)
    desc = _get_description(cur)
    wind = _format_wind(cur.get("windspeedKmph", ""))
    humidity = cur.get("humidity", "")

    parts = [f"В {city} сейчас {temp} градусов"]
    if desc:
        parts.append(desc.lower())
    if feels and feels != temp:
        parts.append(f"ощущается как {feels}")
    if wind:
        parts.append(f"ветер {wind}")
    if humidity:
        parts.append(f"влажность {humidity} процентов")

    return ", ".join(parts) + "."


def _format_tomorrow(data: dict, fallback_city: str) -> str:
    try:
        tomorrow = data["weather"][1]
    except (KeyError, IndexError):
        return "Не удалось получить прогноз на завтра"

    city = _get_city(data, fallback_city)
    min_t = tomorrow.get("mintempC", "?")
    max_t = tomorrow.get("maxtempC", "?")

    desc = ""
    try:
        hours = tomorrow.get("hourly", [])
        if len(hours) > 4:
            desc = _get_description(hours[4])
    except (KeyError, IndexError):
        pass

    parts = [f"Завтра в {city} от {min_t} до {max_t} градусов"]
    if desc:
        parts.append(desc.lower())

    return ", ".join(parts) + "."


# ──────────────────────── HANDLER ────────────────────────

_TRIGGERS = ["погод", "температур", "дожд", "солнечн", "ветер", "прогноз"]


def _extract_city(text: str) -> str:
    """Извлекает город из фразы 'погода в Москве' → 'Москва'."""
    # Ищем "в <Город>" после слова погода/прогноз
    match = re.search(
        r"(?:погод\w*|прогноз\w*)\s+в\s+([а-яёa-z\-]+)",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().capitalize()

    # Общий паттерн "в <Город>"
    match = re.search(r"\bв\s+([А-ЯЁ][а-яё\-]+)", text)
    if match:
        return match.group(1)

    return ""


def try_handle(text: str):
    if not any(w in text for w in _TRIGGERS):
        return None

    is_tomorrow = any(w in text for w in ["завтра", "прогноз"])
    city = _extract_city(text) or DEFAULT_CITY

    logger.info("Запрашиваю погоду (город: %s, завтра: %s)...", city, is_tomorrow)

    data = _fetch_weather(city)
    if data is None:
        return "Не могу получить данные о погоде. Проверь интернет.", None

    if is_tomorrow:
        return _format_tomorrow(data, city), None
    return _format_current(data, city), None