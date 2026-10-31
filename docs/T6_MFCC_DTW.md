# T6: MFCC и DTW для детектора обращения

**Статус**: ✅ Завершён  
**Тесты**: 20 pytest (зелёные), покрытие 96%  
**Интеграция**: режим `WAKEWORD_MODE="text"` сохранён по умолчанию

## Реализация

### 1. Акустические признаки (`src/mathcore/features.py`, 211 строк)

Чистые функции на numpy без побочных эффектов:

- **`hann(n)`** — окно Ханна для сглаживания спектра
- **`hz_to_mel(f)`**, **`mel_to_hz(m)`** — конвертация частота ↔ мелы
- **`mel_filterbank(sr, n_fft, n_mels)`** — треугольные мел-фильтры, матрица (n_mels, n_fft//2+1)
- **`dct_ii(x, n_out)`** — дискретное косинусное преобразование типа II
- **`mfcc(frame, sr, n_fft, n_mels, n_coef)`** — 13 MFCC одного кадра (окно → спектр → мел → логарифм → DCT)
- **`mfcc_sequence(audio, sr, frame, hop)`** — последовательность MFCC: матрица (T, 13)
- **`deltas(coeffs, k)`** — дельта-признаки: производная по времени (не используется в T6, заготовка)
- **`dtw(x, y, window)`** — динамическое выравнивание: нормированное расстояние D/(n+m) и путь

**Параметры по умолчанию**: sr=16 кГц, кадр 512, сдвиг 160, 26 мел-фильтров, 13 коэффициентов.  
**Сложность**: O(n·m) для DTW без окна, O(n·window) с окном Сако-Чибы.

### 2. DTW-детектор (`src/wakeword_dtw.py`, 91 строка)

```python
detector = WakewordDTW(refs_dir=Path("data/wakeword"))
score = detector.score(audio_int16)  # float: меньше → ближе к «Джарвису»
is_wake = detector.is_match(audio_int16)  # bool: score <= threshold
```

**Эталоны**: файлы `data/wakeword/ref_*.npy` (MFCC-последовательности, сохранённые numpy).  
**Запись эталонов**: `python scripts/record_wakeword_refs.py` (требует микрофон, маркер hw).  
**Режим**: `WAKEWORD_MODE="text"` (по умолчанию); для включения DTW: `"dtw"` в `config.py`.

### 3. Конфигурация (`config.py`)

```python
WAKEWORD_MODE = "text"              # "text" | "dtw"
WAKEWORD_DTW_THRESHOLD = 0.35       # максимальное DTW-расстояние
WAKEWORD_DTW_WINDOW = 5             # окно Сако-Чибы
DATA_WAKEWORD_DIR = Path("data/wakeword")
```

## Тесты

### `tests/test_features.py` (15 тестов)

- **Математические свойства**: hann симметричен, мелы обратимы (roundtrip), DCT совпадает с прямой суммой
- **DTW инварианты**: тождество (0 для x=x), симметрия, неравенство треугольника, монотонность пути
- **Окно Сако-Чибы**: узкое окно → больше расстояние; |n-m| > window → inf
- **MFCC разделимость**: тон 440 Гц имеет выше отношение C1..C3/C4..C12, чем белый шум
- **Валидация**: негативные частоты, пустые входы, некорректные размерности → ValueError

### `tests/test_wakeword_dtw.py` (5 тестов)

- Эталон совпадает с самим собой: `score ≈ 0`, `is_match=True`
- Зашумлённый сигнал дальше от эталона, чем исходный
- `add_reference` сохраняет `.npy` и увеличивает `len(_refs)`
- Пустое аудио → `score=inf`, отсутствие эталонов → `FileNotFoundError`
- Режим по умолчанию остался `text` (обратная совместимость)

**Запуск**:
```bash
pytest tests/test_features.py tests/test_wakeword_dtw.py --cov=src.mathcore.features --cov=src.wakeword_dtw
# 20 passed, покрытие 96% (missing: error-paths малых вероятностей)
```

## Использование (опциональное)

### Запись эталонов

```bash
python scripts/record_wakeword_refs.py --count 5 --sec 1.5
# Сохранит 5 файлов ref_000.npy .. ref_004.npy в data/wakeword/
```

### Включение DTW-режима

```python
# config.py
WAKEWORD_MODE = "dtw"

# src/wakeword.py (будущая интеграция — T6 не трогает основной детектор)
if config.WAKEWORD_MODE == "dtw":
    from src.wakeword_dtw import WakewordDTW
    dtw_detector = WakewordDTW()
    # Использовать dtw_detector.is_match(audio) вместо text-based проверки
```

## Ограничения и будущие улучшения

- **Только математическое ядро**: T6 не интегрирует DTW в `WakewordDetector` (это отдельная задача)
- **Без дельта-признаков**: `deltas()` реализована, но не используется; можно конкатенировать Δ-MFCC к статическим
- **Эталоны не входят в репозиторий**: каждый пользователь записывает свой голос
- **Фиксированные параметры**: sr=16 кГц, frame=512 — для совместимости эталонов и оценок

## Проверка задания

- [x] MFCC на чистом numpy (rfft, матричная DCT-II, окно Ханна, мел-фильтры)
- [x] DTW с окном Сако-Чибы: O(n·window) память/время
- [x] Тесты покрывают инварианты (тождество, симметрия, неравенство треугольника)
- [x] Детектор по эталонам: min DTW-расстояние <= порога
- [x] Скрипт записи эталонов (маркер hw, требует микрофон)
- [x] Режим `text` остался по умолчанию (прежний путь работает)
- [x] 20 pytest зелёных, 229 всех тестов без регрессий
- [x] Покрытие 96%, только валидационные error-paths пропущены