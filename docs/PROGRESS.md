### T1 + T2 — `src/mathcore/` (distance + phonetics): старт Phase 5

Статус: готово

Создано:
  - `src/mathcore/__init__.py`
  - `src/mathcore/weights.py`
  - `src/mathcore/distance.py`
  - `src/mathcore/phonetics.py`
  - `tests/test_distance.py`
  - `tests/test_phonetics.py`

Изменено: (новых модулей в рантайме нет — чистые функции; `src/llm_tools.py` не трогался)

Тесты: T1 — 12 тестов (`tests/test_distance.py`); T2 — 10 тестов (`tests/test_phonetics.py`);
итого 22 новых; полный прогон `python -m pytest -q -m "not hw"` → **52 passed**.

Статический анализ: `py_compile` OK; `ruff`/`mypy` в окружении **отсутствуют** (не ставились);
проверка длин строк (> 100) — чисто.

Ключевые решения и отклонения:
  - Взвешенное DL (insert=0.8, del=0.9) по умолчанию асимметричен → публичный API симметризирует
    через `min(d(a,b), d(b,a))`; аксиомы метрики (симметрия, треугольник, неотрицательность,
    нулевое-равенство) проверены на случайных данных — зелёные.
  - `sub_table` по умолчанию = `DEFAULT_SUB`; замена равных символов = 0.
  - Оглушение фонетики доведено до фикса точки (каскад «жжд»).
  - Ожидаемые коды: `phonetic_code("вк") == "фк"`, `phonetic_code("вода") == "вада"`,
    `phonetic_code("громкость") == "грамкасть"`, `phonetic_code("процессора") == "прациссара"`,
    `phonetic_code("открой века") == "открай вика"`, `phonetic_code("джарвис") == "джарвис"`.

Не сделано и почему:
  - T3 (WDL в `src/fuzzy.py`) — следующий шаг (Phase 5).
  - T4–T8 — позже.

Данные, которые нужно собрать на реальном железе: аудио «Джарвис» для E2 (T7).

Команды для проверки пользователем:
  - `python -m pytest -q -m "not hw"`
  - `python -m pytest tests/test_distance.py tests/test_phonetics.py -v`

Критерий приёмки: T1 — все группы тестов distance зелёные; T2 — тесты фонетики зелёные.

