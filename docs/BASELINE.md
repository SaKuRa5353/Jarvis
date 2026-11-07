# Baseline — проект «Джарвис»

Дата актуализации: 2026-11-07.
Опорный commit: `024714e` (`docs: document mathcore and standardize CI`).
Назначение: T0 и сверка состояния T1–T12.

## Результаты

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
| Существует ли `src/mathcore/`? | да | 8 модулей: distance, phonetics, bayes, viterbi, features, thresholds, arith, weights |
| Есть ли `is_match()` как публичный метод `WakewordDetector`? | да | `src/wakeword.py:48: def is_match(self, text: str) -> bool:` |
| Есть ли безопасный `calculate`? | да | `TOOLS`, `execute_tool`, `src/mathcore/arith.py`, `tests/test_arith.py` |
| Есть ли T9-скрипты? | да | четыре скрипта с `--json` и `--self-test` |
| Проходят ли быстрые pytest? | да | `python -m pytest -q -m "not hw"` — все собранные тесты зелёные |
| Покрытие mathcore? | да | pytest-cov — 98% |
| Проходит ли `make ci`? | да | scoped ruff + mypy + pytest — зелёные |
| Проходит ли dry-run runtime-зависимостей? | да | `.venv/bin/python -m pip install -r requirements.txt --dry-run` |
| Есть ли реальные аудиоданные T9? | нет | найден только `data/tts_cache/*.wav` |

## Текущее состояние репозитория

- Опорный HEAD: `024714e`.
- Реализованы T1–T8; T9 self-test, T10 CI и T11 документация находятся в репозитории.
- `docs/MATH.md` описывает публичный API mathcore.
- Default-модель: `qwen2.5:3b`; решение принято явно из-за CPU и поведения reasoning у qwen3.
- Внешние проверки микрофона, Ollama, KDE и длительных WAV-наборов требуют
	доступного hardware/data set.

## Критерий приёмки T0

`docs/BASELINE.md` существует, все строки таблицы заполнены, ни один файл
проекта не изменён (кроме самого `docs/BASELINE.md`).

Дополнительно: `src/mathcore` — 98% покрытия, scoped ruff и mypy чисты, T9
self-tests зелёные. Полевая точность STT/wakeword и ROC-пороги пока не измерены.
