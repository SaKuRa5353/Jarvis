"""Каталог интентов и классификация намерений через математическое ядро.

Интенты описывают готовые действия ассистента (открыть приложение, изменить
громкость, заблокировать экран, уточнить погоду). Классификатор —
наивный байесовский, обученный на триггерных паттернах каталога.

Решение принимается правилом:

    dangerous = intent.dangerous
    tau, delta = (INTENT_TAU_DANGEROUS, INTENT_DELTA_DANGEROUS) if dangerous \
        else (INTENT_TAU, INTENT_DELTA)
    action_allowed = decide(proba, tau, delta)
    fallback = not action_allowed

Если действие не разрешено (низкая уверенность или недостаточная маржа) —
возвращается ``fallback = True``: вызывается прежняя цепочка правил
``src/commands``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import config
from src.logger import get_logger
from src.mathcore.bayes import NGRAM_BOUNDARY, NaiveBayes, decide, margin

logger = get_logger(__name__)


@dataclass(frozen=True)
class Intent:
    """Готовое действие ассистента.

    Attributes:
        name:        идентификатор интента, например ``"web.vk"`` или ``"system.time"``.
        patterns:    триггерные подстроки, использованные при обучении.
        dangerous:   ``True`` — действие имеет побочный эффект (открывает,
                    блокирует, меняет состояние); требует более строгих порогов.
        handler:     модуль, исполняющий действие (ключ из ``src/commands``).
    """

    name: str
    patterns: tuple[str, ...]
    dangerous: bool
    handler: str | None


@dataclass(frozen=True)
class IntentResult:
    """Результат классификации для одной фразы.

    Attributes:
        intent:      имя кандидата или ``None``, если командного соответствия нет.
        prob:        апостериорная вероятность топ-1.
        margin:      разрыв ``P(top1) - P(top2)``.
        action_allowed: разрешено ли выполнение действия (правила ``decide``).
        fallback:    ``True``, если действие не разрешено — нужен прежний путь.
    """

    intent: str | None
    prob: float
    margin: float
    action_allowed: bool
    fallback: bool


# Каталог интентов, собранный из фактического кода src/commands/*.py.
# handler — ключ модуля из src/commands (apps/system/sensors/web/weather/hardware).
CATALOG: tuple[Intent, ...] = (
    # ── Приложения (apps.py) ──
    Intent("apps.explorer", ("проводник", "файлы", "документы", "папк"), True, "apps"),
    Intent(
        "apps.editor",
        ("vscode", "вс код", "вс-код", "vs code", "visual studio", "вс студия", "код"),
        True,
        "apps",
    ),
    Intent("apps.browser", ("браузер", "brave", "брейв"), True, "apps"),
    Intent("apps.terminal", ("терминал", "консоль", "konsole"), True, "apps"),
    Intent("apps.telegram", ("телеграм", "telegram", "тг", "телега"), True, "apps"),
    # ── Веб: открытие ссылок, поиск ──
    Intent("web.vk", ("вк", "вконтакте", "вэ ка", "века", "vk", "в вк", "в контакте"), True, "web"),
    Intent("web.youtube", ("ютуб", "youtube", "ютубе"), True, "web"),
    Intent("web.music", ("музыка", "музыку"), True, "web"),
    Intent("web.moodle", ("мудл",), True, "web"),
    Intent("web.schedule", ("расписание",), True, "web"),
    Intent(
        "web.deepseek",
        ("дипсик", "дипсика", "дип сик", "дикси", "deepseek", "deep sick", "депсик"),
        True,
        "web",
    ),
    Intent("web.mail", ("почта", "почту"), True, "web"),
    Intent(
        "web.search",
        (
            "загугли",
            "за гугли",
            "за гугле",
            "погугли",
            "загуглить",
            "загули",
            "за гуглить",
            "найди",
            "поищи",
            "ищи",
            "найти",
            "искать",
            "интернет",
            "интернете",
            "сети",
        ),
        True,
        "web",
    ),
    # ── Система: время, дата, блокировка, громкость, скриншот ──
    Intent(
        "system.time",
        (
            "который час",
            "который сейчас час",
            "сколько времени",
            "сколько сейчас времени",
            "какой час",
            "какой сейчас час",
            "время сейчас",
            "сколько время",
        ),
        False,
        "system",
    ),
    Intent(
        "system.date",
        (
            "какое сегодня число",
            "какая дата",
            "какое число",
            "какое сегодня",
            "какой сегодня день",
            "какая сегодня дата",
        ),
        False,
        "system",
    ),
    Intent(
        "system.lock", ("заблокируй", "блокировка экрана", "заблокировать экран"), True, "system"
    ),
    Intent(
        "system.volume_up",
        (
            "громкость выше",
            "сделай погромче",
            "громче",
            "погромче",
            "прибавь",
            "добавь",
            "громкость больше",
            "звук выше",
            "звук больше",
        ),
        True,
        "system",
    ),
    Intent(
        "system.volume_down",
        ("громкость тише", "убавь", "тише", "потише", "громкость ниже", "звук ниже"),
        True,
        "system",
    ),
    Intent("system.volume_mute", ("выключи звук", "выключи зву", "без звука"), True, "system"),
    Intent("system.volume_unmute", ("включи звук", "верни звук"), True, "system"),
    Intent("system.screenshot", ("скриншот", "снимок экрана", "скрин"), True, "system"),
    # ── Датчики: температура CPU ──
    Intent(
        "sensor.cpu",
        (
            "температур",
            "нагрев",
            "перегрев",
            "градус",
            "жарит",
            "горяч",
            "процессор",
            "проц",
            "cpu",
            "цпу",
            "видеокарт",
            "gpu",
            "гпу",
            "комп",
            "ноут",
            "систем",
            "материн",
            "желез",
            "железо",
            "конденсатор",
            "радиатор",
            "кулер",
            "вентилятор",
        ),
        False,
        "sensors",
    ),
    # ── Погода ──
    Intent(
        "weather.current",
        (
            "погод",
            "погода",
            "погоду",
            "температур",
            "дожд",
            "солнечн",
            "ветер",
        ),
        False,
        "weather",
    ),
    Intent("weather.forecast", ("завтра", "прогноз"), False, "weather"),
    # ── Железо: характеристики компьютера ──
    Intent(
        "hardware.info",
        (
            "характеристик",
            "характеристика",
            "желез",
            "железо",
            "конфигураци",
            "что у меня за компьютер",
            "что за компьютер",
            "какое у меня железо",
            "что у меня внутри",
            "расскажи про компьютер",
            "параметры компьютера",
        ),
        False,
        "hardware",
    ),
)


def fit_catalog(catalog: Sequence[Intent] = CATALOG) -> NaiveBayes:
    """Обучает классификатор на паттернах каталога."""
    samples: list[tuple[str, str]] = []
    for intent in catalog:
        for pattern in intent.patterns:
            samples.append((pattern, intent.name))
    model = NaiveBayes(
        alpha=1.0,
        n_min=config.INTENT_NGRAM_MIN,
        n_max=config.INTENT_NGRAM_MAX,
        boundary=NGRAM_BOUNDARY,
    )
    return model.fit(samples)


@lru_cache(maxsize=4)
def _model(n_min: int, n_max: int) -> NaiveBayes:
    """Явно инициализирует модель для пары границ n-грамм: M = fit(CATALOG)."""
    samples = [(pattern, item.name) for item in CATALOG for pattern in item.patterns]
    return NaiveBayes(n_min=n_min, n_max=n_max).fit(samples)


def supports(text: str, intent: Intent) -> bool:
    """Проверяет лексическую область команды: поддержка равна 0 или 1.

    Для коротких команд все слова должны покрываться шаблоном и служебными
    словами. Поиск и информационные запросы допускают свободные аргументы.
    Это не позволяет относительной вероятности Байеса заменить проверку того,
    что реплика вообще относится к командам.
    """
    words = re.findall(r"[\w-]+", text.lower())
    normalized = " ".join(words)
    if intent.name == "sensor.cpu":
        return (
            any(
                w in normalized
                for w in ("температур", "нагрев", "перегрев", "градус", "жарит", "горяч")
            )
            and any(w in normalized for w in intent.patterns[6:])
            and not any(
                w in normalized
                for w in (
                    "улиц",
                    "воздух",
                    "погод",
                    "окн",
                    "снаруж",
                    "за бортом",
                    "градусник",
                    "мете",
                    "синоп",
                )
            )
        )
    if intent.name.startswith("weather."):
        triggered = any(
            w in normalized for w in ("погод", "температур", "дожд", "солнечн", "ветер", "прогноз")
        )
        forecast = "завтра" in words or "прогноз" in normalized
        return triggered and forecast == (intent.name == "weather.forecast")
    if intent.name == "web.search":
        return (
            re.search(
                r"(?:найди в (?:гугле|интернете)|загугли|за гуг(?:ли|ле|лить)|загули|погугли|загуглить|поищи)\s+\S|найди\s+.+в гугле",
                normalized,
            )
            is not None
        )
    if intent.name == "hardware.info":
        return any(pattern in normalized for pattern in intent.patterns)
    wrappers = {
        "открой",
        "открыть",
        "запусти",
        "включи",
        "иди",
        "на",
        "перейди",
        "пожалуйста",
        "джарвис",
        "сделай",
        "мне",
        "звук",
        "громкость",
        "экран",
    }
    for pattern in intent.patterns:
        # Короткие ключи не должны превращать «включи» в команду «вк».
        suffix = r"\w*" if len(pattern) > 3 else ""
        match = re.search(r"(?<!\w)" + re.escape(pattern) + suffix + r"(?!\w)", normalized)
        if match is None:
            continue
        if pattern in {"века", "вэ ка"} and not any(
            w in words for w in ("открой", "открыть", "запусти", "включи", "перейди")
        ):
            continue
        remainder = normalized[: match.start()] + " " + normalized[match.end() :]
        if all(word in wrappers for word in remainder.split()):
            return True
    return False


def classify(text: str) -> IntentResult:
    """Оценивает P(c|q, поддержка) и применяет пороги вероятности и маржи.

    P(c|q, поддержка) ∝ P(c|q) * supports(q, c). При отсутствии поддержки
    используется равномерное распределение, маржа которого равна нулю.
    """
    if not config.INTENT_ENABLED:
        return IntentResult(intent=None, prob=0.0, margin=0.0, action_allowed=False, fallback=True)
    proba = _model(config.INTENT_NGRAM_MIN, config.INTENT_NGRAM_MAX).predict_proba(text)
    supported = {item.name for item in CATALOG if supports(text, item)}
    total = sum(prob for name, prob in proba.items() if name in supported)
    proba = {
        name: (prob / total if name in supported else 0.0) if total else 1.0 / len(CATALOG)
        for name, prob in proba.items()
    }
    intent_name = max(proba, key=lambda k: proba[k]) if supported else None
    prob = proba[intent_name] if intent_name is not None else 0.0
    best_margin = margin(proba)
    intent_obj = next((i for i in CATALOG if i.name == intent_name), None)
    dangerous = bool(intent_obj.dangerous) if intent_obj is not None else False
    tau, delta = (
        (config.INTENT_TAU_DANGEROUS, config.INTENT_DELTA_DANGEROUS)
        if dangerous
        else (config.INTENT_TAU, config.INTENT_DELTA)
    )
    action_allowed = decide(proba, tau, delta)
    return IntentResult(
        intent=intent_name,
        prob=prob,
        margin=best_margin,
        action_allowed=action_allowed,
        fallback=not action_allowed,
    )
