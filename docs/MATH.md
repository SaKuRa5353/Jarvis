# Математическое ядро

`src/mathcore/` содержит чистые функции, которые не открывают микрофон, не
обращаются к сети и не вызывают LLM. Runtime-модули используют их через
направление зависимостей `потребитель -> mathcore`.

Проверки выполняются в тестах `tests/test_*.py`. Текущее покрытие mathcore —
98% строк.

## 1. Редакционное расстояние

Модуль: `src/mathcore/distance.py`, веса: `src/mathcore/weights.py`.
Потребитель: `src/fuzzy.py`. Тесты: `tests/test_distance.py`.

Для строк `a` и `b` динамическое программирование считает минимальную стоимость
вставок, удалений, замен и соседних транспозиций:

$$D_{i,j}=\min\begin{cases}
D_{i-1,j}+w_{del},\\
D_{i,j-1}+w_{ins},\\
D_{i-1,j-1}+w_{sub}(a_i,b_j),\\
D_{i-2,j-2}+w_{tr},
\end{cases}$$

Последний переход используется только для соседних перепутанных символов.
Функция `similarity` нормирует результат:

$$sim(a,b)=1-\frac{d(a,b)}{\max(|a|,|b|)}.$$

Публичные функции:

- `directed_distance` — направленная стоимость преобразования `a` в `b`;
- `damerau_levenshtein` — расстояние с настраиваемыми весами и полосой поиска;
- `similarity` — нормированное сходство, для двух пустых строк возвращает `1`;
- `is_metric` — проверяет аксиомы на переданной тройке строк, но не доказывает,
  что алгоритм является метрикой на всех строках.

Без полосы время и память составляют `O(nm)`. Текущая реализация с `band`
пропускает клетки вне полосы, но хранит полную матрицу, поэтому память остаётся
`O(nm)`.

## 2. Фонетическая нормализация

Модуль: `src/mathcore/phonetics.py`. Потребитель: `src/fuzzy.py`.
Тесты: `tests/test_phonetics.py`.

`phonetic_code` последовательно приводит текст к нижнему регистру, заменяет `ё`,
удаляет лишние символы, моделирует оглушение и редукцию, удаляет мягкий и
твёрдый знаки, схлопывает повторы и применяет ассимиляцию шипящих.

Публичные функции:

- `phonetic_code` — устойчивое к типичным ошибкам STT представление текста;
- `phonetic_distance` — `damerau_levenshtein` от фонетических кодов;
- `combined_distance` — взвешенная сумма символьного и фонетического расстояний:

$$d_c(a,b)=\lambda d(a,b)+(1-\lambda)d_{phon}(a,b).$$

Каждый проход по строке имеет линейную сложность `O(n)` без учёта вызова
редакционного расстояния. Для проверки стабильности используется свойство
идемпотентности `phonetic_code(phonetic_code(x)) == phonetic_code(x)`.

## 3. Классификация намерений

Модуль: `src/mathcore/bayes.py`. Потребитель: `src/intent.py`.
Тесты: `tests/test_bayes.py`, `tests/test_intent.py`.

`char_ngrams` строит символьные n-граммы с граничным символом. Для класса `c`
наивный Байес использует сглаживание Лапласа:

$$P(t|c)=\frac{N_{ct}+\alpha}{N_c+\alpha|V|}.$$

Логарифмы вероятностей суммируются, а затем нормируются softmax-преобразованием:

$$P(c|q)=\frac{e^{L_c-\max L}}{\sum_k e^{L_k-\max L}}.$$

Публичные функции:

- `char_ngrams` — n-граммы текста;
- `tf`, `idf`, `tfidf` — вспомогательное TF-IDF представление;
- `cosine` — косинусное сходство разреженных словарей;
- `NaiveBayes.fit` — обучение по парам `(текст, класс)`;
- `NaiveBayes.predict_proba` — распределение вероятностей по классам;
- `NaiveBayes.predict_top2` — два наиболее вероятных класса;
- `margin` — разница между первой и второй вероятностями;
- `decide` — проверка `P(top1) >= tau` и `margin >= delta`.

Построение n-грамм для текста длины `n` и ограничений `n_min..n_max` занимает
`O(n(n_max-n_min+1))`. Предсказание пропорционально числу классов и n-грамм.

## 4. HMM и Viterbi для VAD

Модуль: `src/mathcore/viterbi.py`. Потребитель: `src/vad.py`, если
`VAD_DECODER="viterbi"`. Тесты: `tests/test_viterbi.py`.

Скрытая последовательность состояний оценивается по переходам `A`, начальному
распределению `pi` и Beta-эмиссиям:

$$P(s,x)=\pi_{s_0}\prod_t A_{s_{t-1},s_t}
\prod_t Beta(x_t;\alpha_{s_t},\beta_{s_t}).$$

Viterbi работает в лог-пространстве:

$$V_t(j)=\log b_j(x_t)+\max_i(V_{t-1}(i)+\log A_{ij}).$$

Публичные методы и функции:

- `HMM.fit_transitions` — переходы с лапласовским сглаживанием;
- `HMM.fit_emissions` — Beta-параметры через SciPy MLE или метод моментов;
- `HMM.viterbi` — наиболее вероятный путь и его log-правдоподобие;
- `HMM.forward_loglik` — полное log-правдоподобие через log-sum-exp;
- `hysteresis` — порог с памятью состояния;
- `loop_width_for` — оценка длительности подтверждения по вероятности серии
  ошибочных кадров.

Декодирование занимает `O(T|S|^2)` времени и `O(T|S|)` памяти.
По умолчанию сохранён прежний пороговый VAD.

## 5. MFCC и DTW

Модуль: `src/mathcore/features.py`. Потребитель: `src/wakeword_dtw.py`.
Тесты: `tests/test_features.py`, `tests/test_wakeword_dtw.py`.

Окно Ханна:

$$w[k]=0.5-0.5\cos\left(\frac{2\pi k}{N-1}\right).$$

Переход между частотой и mel-шкалой:

$$m=2595\log_{10}(1+f/700),\qquad
f=700(10^{m/2595}-1).$$

MFCC получает log-энергии mel-фильтров и применяет DCT-II:

$$c_k=\sum_{n=0}^{N-1}x_n\cos\left(\frac{\pi k(2n+1)}{2N}\right).$$

Публичные функции:

- `hann`, `hz_to_mel`, `mel_to_hz` — базовые операции окна и шкалы;
- `mel_filterbank` — треугольный mel-фильтрбанк;
- `dct_ii` — дискретное косинусное преобразование;
- `mfcc`, `mfcc_sequence` — признаки одного кадра и последовательности;
- `deltas` — производные MFCC по времени;
- `dtw` — динамическое выравнивание последовательностей с путём.

DTW использует рекуррентность
`D(i,j)=d(x_i,y_j)+min(D(i-1,j),D(i,j-1),D(i-1,j-1))` и возвращает
`D(n,m)/(n+m)`. Время — `O(nm)`, память текущей реализации — `O(nm)`;
полоса ограничивает вычисляемые переходы, но не выделение полной матрицы.

## 6. Пороги и экспериментальные метрики

Модуль: `src/mathcore/thresholds.py`. Потребители: T9-скрипты измерений.
Тесты: `tests/test_thresholds.py`.

Публичные группы функций:

- `confusion`, `roc_curve`, `auc`, `youden_threshold`, `cost_threshold` —
  матрица ошибок, ROC-AUC и подбор порогов;
- `np_threshold`, `fah`, `alpha_for_fah` — ограничение ложных срабатываний;
- `zipf_hit_rate`, `zipf_cache_size`, `expected_tts_time` — модель TTS-кэша;
- `mm1_utilization`, `mm1_wait` — загрузка и ожидание в M/M/1;
- `liu_layland_bound`, `schedulable` — проверка планируемости задач;
- `optimal_first_chunk` — оценка первого чанка потокового TTS;
- `nlms_update`, `nlms_excess_mse` — адаптивная фильтрация;
- `wilcoxon_paired`, `percentile_ci_median` — статистика связанных выборок.

Например, ложные срабатывания в час считаются как

$$FAH=\alpha\frac{3600}{t_{frame}},$$

а граница Liu–Layland для `n` задач:

$$U(n)=n(2^{1/n}-1).$$

ROC требует сортировки и имеет сложность `O(N log N)`. Большинство скалярных
оценок работает за `O(1)`, а сортировки и переборы кэша — за `O(N)` или
`O(N log N)`.

## 7. Безопасная арифметика

Модуль: `src/mathcore/arith.py`. Потребитель: `src/llm_tools.py` через tool
`calculate`. Тесты: `tests/test_arith.py`, `tests/test_math.py`.

`safe_eval` разбирает выражение через `ast.parse` и разрешает только белый список
числовых констант, операторов и функций из `math`. `eval`, `exec`, атрибуты,
подписки, comprehensions и доступ к файлам запрещены.

Публичные элементы:

- `ArithError` — ошибка политики или вычисления;
- `validate_ast` — проверка AST и лимита узлов;
- `safe_eval` — вычисление с ограничением размера, глубины и результата.

При `N` узлах AST время и память составляют `O(N)`. Дополнительно проверяются
конечность результата и ограничение `|result| <= 10^18`.

## 8. Как запускать проверки

```fish
make ci
make bench
make report
.venv/bin/python -m pytest -q -m "not hw" --cov=src/mathcore
```

Экспериментальные T9-скрипты сохраняют JSON. При отсутствии matplotlib отчёты
используют CSV fallback:

```fish
.venv/bin/python scripts/measure_stt.py --self-test --json /tmp/stt.json
.venv/bin/python scripts/measure_wakeword.py --self-test --json /tmp/wakeword.json
.venv/bin/python scripts/fit_thresholds.py --self-test --json /tmp/thresholds.json
.venv/bin/python scripts/report.py --self-test --json /tmp/report.json --out /tmp/report
```
