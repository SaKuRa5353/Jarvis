# Baseline — зафиксированное состояние проекта «Джарвис»

Дата фиксации: 2026-10-28
Задача: T0 (Phase 5). Файл создаётся впервые; содержание — результат измерений.

## Команды и их выводы

```fish
git log --oneline -15
git status --short
ls src src/mathcore scripts docs 2>/dev/null
grep -n "def " src/wakeword.py
grep -n "is_match\|_match" src/wakeword.py
grep -n "_int_to_words\|_DIGIT_RE\|_OPER_RE\|_normalize_numbers" src/tts.py
grep -n "calculate\|^TOOLS\|def execute_tool\|safe_eval\|evaluate_expression" src/llm_tools.py
grep -n "THRESHOLD\|WAKE_WORDS\|TTS_CACHE_MAX_FILES\|VAD_\|FUZZY_\|INTENT_\|LLM_THINK" config.py
.venv/bin/python -m pytest -q -m "not hw" 2>&1 | tail -3
.venv/bin/pip list 2>/dev/null | grep -iE "numpy|scipy|matplotlib|pytest|mypy|ruff|coverage"
```

## Таблица

| Вопрос | Ответ | Доказательство |
|---|---|---|
| Существует ли `src/mathcore/`? | да | `ls src/mathcore` → `__init__.py`, `distance.py`, `phonetics.py`, `weights.py` |
| Есть ли `is_match()` как публичный метод `WakewordDetector`? | да | `src/wakeword.py:48: def is_match(self, text: str) -> bool:` |
| Есть ли `_int_to_words()` в `src/tts.py`? | да | `src/tts.py:180: def _int_to_words(n: int) -> str:` |
| Есть ли `calculate` в `TOOLS` или `execute_tool` в `src/llm_tools.py`? | да | `src/llm_tools.py:212: "name": "calculate",`; `src/llm_tools.py:230: def execute_tool(...)`; безопасный AST-движок `_eval_math` + `ast.parse` (`src/llm_tools.py:35`, `src/llm_tools.py:90`) |
| Есть ли pytest-тесты (`tests/test_*.py`)? | да | `tests/` содержит `test_*.py`; `python -m pytest -q -m "not hw"` → **101 passed** |
| Установлены ли pytest / mypy / ruff / matplotlib / scipy? | pytest 9.1.1, mypy 2.4.1, ruff 0.17.3, scipy 1.18.1 — установлены; matplotlib — не установлен | `pip list` → `pytest`, `mypy`, `ruff`, `scipy`, `numpy`, `coverage`, `pytest-cov` найдены; `matplotlib` отсутствует |
| Зелёный ли `check_commands.py`? | да | `python tests/check_commands.py` → `VSYO ZELYONOYE`, exit 0 |
| Зелёный ли `check_fuzzy.py`? | да | `python tests/check_fuzzy.py` → `VSYO ZELYONOYE`, exit 0 |

## Текущее состояние репозитория

- HEAD: `61a21c7 Phase 4: математический модуль — tool calculate + безопасный AST-вычислитель`
- `src/mathcore/` создан: `__init__.py`, `weights.py`, `distance.py`, `phonetics.py`.
- `docs/` — каталог отсутствует (создан для этого файла).
- `scripts/` — каталог отсутствует.
- Pytest: `101 passed`, `-m "not hw"` (маркер `hw` не использовался в этом прогоне).
- `git status --short` показывает ожидаемые и неожиданные артефакты (см. Шаг 5 — гигиена).

## Критерий приёмки T0

`docs/BASELINE.md` существует, все строки таблицы заполнены, ни один файл
проекта не изменён (кроме самого `docs/BASELINE.md`).

Дополнительно (Шаг 2 этого раунда): `src/mathcore` — 100% покрытие, mypy и ruff
чисты, значения математических функций сверены (ВСЕ ЗНАЧЕНИЯ СОВПАЛИ).
