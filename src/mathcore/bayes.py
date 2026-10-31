"""Математическое ядро: вероятностные модели и статистические операторы.

Содержит:
    char_ngrams  генерация n-грамм текста с границами слов;
    tf / idf / tfidf  векторное представление терминов;
    cosine  косинусное сходство векторов;
    NaiveBayes  наивный байесовский классификатор с лапласовым сглаживанием;
    margin / decide  правила принятия решения по апостериорным вероятностям.

Вероятностная модель классификатора
-----------------------------------
Фраза ``q`` из ``k`` классов ``C`` оценивается апостериорным правилом

    P(C | q) ∝ P(q | C) * P(C)

Логарифмическая оценка суммирует вклады всех n-грамм фразы.
Лаплазовское сглаживание частот:

    P(t | C) = (N_ct + alpha) / (N_c + alpha * |V|)

где ``N_ct`` — число вхождений термина ``t`` в классе ``C``,
``N_c`` — суммарное число вхождений всех терминов класса, ``|V|`` — размер
обучающего словаря. Логарифмическая форма:

    log P(C | q) = log P(C) + sum_t tf(t) * log P(t | C)

После вычисления по всем классам выполняется нормировка softmax-подобной:

    P(C | q) = exp(log P(C|q) - max_c) / sum_c exp(log P(C|q) - max_c)

    sum_c P(C | q) = 1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

NGRAM_MIN: int = 2
"""Минимальная длина n-граммы (bigram)."""

NGRAM_MAX: int = 4
"""Максимальная длина n-граммы."""

NGRAM_BOUNDARY: str = "␣"
"""Граничный символ, заменяющий пробел и ставящийся слева/справа от слова."""


def char_ngrams(
    text: str,
    n_min: int = NGRAM_MIN,
    n_max: int = NGRAM_MAX,
    boundary: str = NGRAM_BOUNDARY,
) -> list[str]:
    """Возвращает все n-граммы текста для ``n`` из ``[n_min, n_max]``.

    Текст приводится к нижнему регистру, каждый пробел заменяется граничным
    символом, слева и справа от текста добавляются границы. Поэтому для
    однословного слова границы дают первый и последний n-граммы.

    Пример: ``char_ngrams("вк", 2, 2)`` -> ``["␣в", "вк", "к␣"]``.
    """
    if not 1 <= n_min <= n_max:
        raise ValueError("Нужны границы 1 <= n_min <= n_max")
    s = text.lower().replace(" ", boundary)
    s = boundary + s + boundary
    grams: list[str] = []
    for n in range(n_min, n_max + 1):
        for i in range(len(s) - n + 1):
            grams.append(s[i : i + n])
    return grams


def tf(tokens: Sequence[str]) -> dict[str, float]:
    """Частоты терминов в документе: ``tf(t) = N_t`` (подсчёт вхождений).

    Счётчики используются в сумме ``log P(C|q) = log P(C) + Σ tf(t) log P(t|C)``.
    """
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def idf(docs: Sequence[Sequence[str]]) -> dict[str, float]:
    """Обратная документная частота: ``idf(t) = ln(N / df(t))``.

    ``docs`` — последовательность документов, каждый из себя последовательность
    терминов. Термины, не встретившиеся ни в одном документе, не попадают в
    результирующий словарь (их ``idf = 0``).
    """
    n_docs = len(docs)
    df: dict[str, int] = {}
    for doc in docs:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1
    result: dict[str, float] = {}
    for token, doc_freq in df.items():
        result[token] = math.log(n_docs / doc_freq) if doc_freq > 0 else 0.0
    return result


def tfidf(tokens: Sequence[str], idf_map: Mapping[str, float]) -> dict[str, float]:
    """Вектор TF-IDF: произведение ``tf(t) * idf(t)`` по терминам."""
    counts = tf(tokens)
    return {token: value * idf_map.get(token, 0.0) for token, value in counts.items()}


def cosine(u: Mapping[str, float], v: Mapping[str, float]) -> float:
    """Косинусное сходство двух векторов: ``dot / (||u|| * ||v||)``.

    При нулевых векторах возвращается ``0.0``.
    """
    dot = 0.0
    norm_u = 0.0
    norm_v = 0.0
    for key in set(u) | set(v):
        a = u.get(key, 0.0)
        b = v.get(key, 0.0)
        dot += a * b
        norm_u += a * a
        norm_v += b * b
    norm_u = math.sqrt(norm_u)
    norm_v = math.sqrt(norm_v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return dot / (norm_u * norm_v)


class NaiveBayes:
    """Наивный байесовский классификатор с лапласовым сглаживанием.

    Модель обучается по парам ``(text, label)`` — фактически это обучающая
    выборка интентов. Для оценки вероятности класса берётся апостериор с
    лапласовым сглаживанием частот и нормировкой по классам (softmax).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_min: int = NGRAM_MIN,
        n_max: int = NGRAM_MAX,
        boundary: str = NGRAM_BOUNDARY,
    ) -> None:
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha (лапласовское сглаживание) должен быть > 0")
        if not 1 <= n_min <= n_max:
            raise ValueError("Нужны границы 1 <= n_min <= n_max")
        self.alpha = alpha
        self.n_min = n_min
        self.n_max = n_max
        self.boundary = boundary
        self.classes: tuple[str, ...] = ()
        self._token_counts: dict[str, dict[str, float]] = {}
        self._class_totals: dict[str, float] = {}
        self._class_docs: dict[str, int] = {}
        self._doc_count: int = 0
        self._vocab: set[str] = set()
        self._docs_tokens: list[list[str]] = []

    def fit(self, samples: Sequence[tuple[str, str]]) -> NaiveBayes:
        """Заново обучает модель: P(t|c) = (N_ct + alpha)/(N_c + alpha |V|)."""
        self._token_counts.clear()
        self._class_totals.clear()
        self._class_docs.clear()
        self._vocab.clear()
        self._docs_tokens.clear()
        self._doc_count = 0
        for text, label in samples:
            tokens = char_ngrams(text, self.n_min, self.n_max, self.boundary)
            counter = self._token_counts.setdefault(label, {})
            for token in tokens:
                counter[token] = counter.get(token, 0.0) + 1.0
                self._vocab.add(token)
            self._class_totals[label] = self._class_totals.get(label, 0.0) + len(tokens)
            self._class_docs[label] = self._class_docs.get(label, 0) + 1
            self._doc_count += 1
            self._docs_tokens.append(tokens)
        self.classes = tuple(sorted(self._class_docs))
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        """Апостериорные вероятности классов для фразы ``text``."""
        if not self.classes:
            return {}
        tokens = char_ngrams(text, self.n_min, self.n_max, self.boundary)
        counts = tf(tokens)
        vocab_size = max(len(self._vocab), 1)
        n_docs = max(self._doc_count, 1)
        n_classes = len(self.classes)
        log_posteriors: list[tuple[str, float]] = []
        for klass in self.classes:
            prior = self._class_docs.get(klass, 0) / n_docs if n_docs else 1.0 / n_classes
            total = self._class_totals.get(klass, 0.0)
            log_sum = 0.0
            for token, token_count in counts.items():
                token_count_ct = self._token_counts.get(klass, {}).get(token, 0.0)
                prob_token = (token_count_ct + self.alpha) / (total + self.alpha * vocab_size)
                log_sum += token_count * math.log(prob_token)
            log_posteriors.append((klass, math.log(prior) + log_sum))
        max_log = max(log_prob for _, log_prob in log_posteriors)
        exps = [math.exp(log_prob - max_log) for _, log_prob in log_posteriors]
        total_exp = sum(exps)
        return {
            klass: exp_val / total_exp
            for (klass, _), exp_val in zip(log_posteriors, exps, strict=False)
        }

    def predict_top2(self, text: str) -> tuple[str, float, str, float]:
        """Топ-2 класса: ``(имя_1, prob_1, имя_2, prob_2)``."""
        proba = self.predict_proba(text)
        ordered = sorted(proba.items(), key=lambda item: (-item[1], item[0]))
        if not ordered:
            raise ValueError("Классификатор ещё не обучен")
        name1, prob1 = ordered[0]
        if len(ordered) > 1:
            name2, prob2 = ordered[1]
        else:
            name2, prob2 = "", 0.0
        return name1, prob1, name2, prob2


def margin(proba: Mapping[str, float]) -> float:
    """Разрыв между топ-1 и топ-2 апостериорной вероятностью.

    Для вырожденного распределения (один класс) возвращает ``P(top1)``.
    """
    values = sorted(proba.values(), reverse=True)
    if not values:
        return 0.0
    top1 = values[0]
    if len(values) == 1:
        return top1
    return top1 - values[1]


def decide(proba: Mapping[str, float], tau: float, delta: float) -> bool:
    """Решение по правилам: ``P(top1) >= tau`` И ``margin >= delta``."""
    if not proba:
        return False
    top1 = max(proba.values())
    return top1 >= tau and margin(proba) >= delta
