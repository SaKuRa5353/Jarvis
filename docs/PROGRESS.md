## Состояние проекта после T0–T12

Актуализация: 2026-11-07. Опорная точка: commit `024714e`.

| Задача | Статус | Подтверждение |
|---|---|---|
| T0 | готово | `docs/BASELINE.md` |
| T1 | реализовано | weighted distance и тесты |
| T2 | реализовано | phonetic code и тесты |
| T3 | реализовано | WDL подключён в `src/fuzzy.py` |
| T4 | реализовано | Bayes/intent и tests |
| T5 | частично | HMM подключается при `VAD_DECODER="viterbi"`, default threshold |
| T6 | частично | MFCC/DTW есть, основной wakeword default text |
| T7 | реализовано | ROC, пороги, FAH, Ципф, M/M/1, NLMS, статистика |
| T8 | реализовано | безопасный AST и tool `calculate` |
| T9 | self-test готов | четыре скрипта, но нет реальных WAV-разметок |
| T10 | готово | Makefile, fixtures, CI, ruff, mypy |
| T11 | готово частично | `docs/MATH.md`, README; таблица интентов ручная |
| T12 | в работе | unit/CI зелёные, ручная часть требует hardware/services |

### Подтверждено

- `make ci` проходит: scoped ruff, mypy mathcore и pytest.
- Покрытие `src/mathcore` — 98%.
- Все четыре T9 self-test проходят без микрофона и Ollama.
- `pip install -r requirements.txt --dry-run` проходит в `.venv`.
- `wttr.in` доступен и возвращает JSON для Петрозаводска.
- Арифметический маршрут проверен fake LLM: `sin(pi/2)` → `calculate` → `1`.
- Пустой ответ wttr.in повторяется один раз, затем команда возвращает отказ.
- Default-модель — `qwen2.5:3b`: она соответствует целевому CPU и не включает
  проблемный обязательный reasoning у `qwen3-jarvis`.
- Default-флаги: `FUZZY_ENGINE="wdl"`, `INTENT_ENABLED=True`,
  `VAD_DECODER="threshold"`, `WAKEWORD_MODE="text"`.

### Ограничения

- В репозитории нет размеченных пользовательских WAV: найден только TTS-кэш.
  Синтетические T9-пороги нельзя переносить в `config.py`.
- Нужны реальные STT-команды с golden-текстом, положительные/отрицательные
  записи wakeword и размеченные fuzzy/intent-примеры.
- Остались ручные `check_*.py`, проверки микрофона, Ollama, KDE и итоговая
  фиксация T12 после доступа к hardware/services.

### Команды

```fish
make ci
make bench
make report
.venv/bin/python -m pytest -q -m "not hw" --cov=src/mathcore
```

