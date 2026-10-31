"""Unit-тесты математического ядра bayes: n-граммы, tf/idf, cosine, NaiveBayes."""

from __future__ import annotations

import math

import pytest

from src.mathcore.bayes import (
    NGRAM_BOUNDARY,
    NaiveBayes,
    char_ngrams,
    cosine,
    decide,
    idf,
    margin,
    tf,
    tfidf,
)


def test_char_ngrams_contains_boundaries():
    grams = char_ngrams("вк", 2, 2)
    assert f"{NGRAM_BOUNDARY}в" in grams
    assert "вк" in grams
    assert f"к{NGRAM_BOUNDARY}" in grams


def test_char_ngrams_lowercases_and_adds_boundaries():
    assert char_ngrams("ВК", 2, 2) == [f"{NGRAM_BOUNDARY}в", "вк", f"к{NGRAM_BOUNDARY}"]


def test_char_ngrams_respects_range():
    grams = char_ngrams("вк", 2, 2)
    assert len(grams) == 3
    assert all(len(g) == 2 for g in grams)


def test_tf_counts():
    assert tf(["a", "b", "a"]) == {"a": 2.0, "b": 1.0}


def test_idf_scores_document_frequency():
    docs = [["a", "b"], ["a"], ["c"]]
    idf_map = idf(docs)
    # df(a) = 2 -> ln(3/2); df(c) = 1 -> ln(3)
    assert math.isclose(idf_map["a"], math.log(1.5), rel_tol=1e-9)
    assert math.isclose(idf_map["c"], math.log(3.0), rel_tol=1e-9)


def test_tfidf_is_tf_times_idf():
    result = tfidf(["a", "a", "b"], {"a": 2.0, "b": 0.0})
    assert result["a"] == 4.0
    assert result["b"] == 0.0


def test_cosine_identity_and_orthogonal():
    v = {"a": 1.0, "b": 2.0}
    assert math.isclose(cosine(v, v), 1.0, rel_tol=1e-9)
    assert cosine({"a": 1.0}, {"b": 2.0}) == 0.0


def test_naive_bayes_proba_sums_to_one_over_50_texts():
    model = NaiveBayes(alpha=1.0).fit(
        [
            ("вк", "web.vk"),
            ("вконтакте", "web.vk"),
            ("который час", "system.time"),
            ("сколько времени", "system.time"),
            ("громкость выше", "system.volume_up"),
        ]
    )
    texts = [
        "вк",
        "вконтакте",
        "который час",
        "сколько времени",
        "громкость выше",
        "температура процессора",
        "погода в петрозаводске",
        "скриншот",
        "выключи звук",
        "ты ждешь",
        "я иду домой",
        "офис",
        "клавиатура",
        "монитор",
        "браузер",
    ]
    texts.extend(f"пример реплики номер {i}" for i in range(50 - len(texts)))
    assert len(set(texts)) == 50
    for text in texts:
        proba = model.predict_proba(text)
        assert math.isclose(sum(proba.values()), 1.0, rel_tol=1e-9)


def test_margin_uniform_zero():
    proba = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    assert margin(proba) == 0.0


def test_margin_degenerate_single_class():
    assert margin({"a": 1.0}) == 1.0


def test_decide_requires_tau_and_margin():
    assert decide({"a": 0.9, "b": 0.1}, tau=0.55, delta=0.15) is True
    assert decide({"a": 0.4, "b": 0.4}, tau=0.55, delta=0.15) is False
    assert decide({"a": 0.9, "b": 0.05}, tau=0.95, delta=0.15) is False


def test_fit_replaces_previous_samples() -> None:
    model = NaiveBayes().fit([("вк", "старый")])
    model.fit([("час", "новый")])
    assert model.classes == ("новый",)
    assert model.predict_top2("час") == ("новый", 1.0, "", 0.0)


def test_predict_top2_and_empty_model() -> None:
    model = NaiveBayes()
    assert model.predict_proba("вк") == {}
    with pytest.raises(ValueError, match="не обучен"):
        model.predict_top2("вк")
    model.fit([("вк", "сайт"), ("час", "время")])
    first, p1, second, p2 = model.predict_top2("вк")
    assert first == "сайт"
    assert second == "время"
    assert p1 > p2


@pytest.mark.parametrize("alpha", [0.0, -1.0, math.nan, math.inf])
def test_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        NaiveBayes(alpha=alpha)


def test_invalid_ngram_range() -> None:
    with pytest.raises(ValueError):
        char_ngrams("вк", 0, 2)
    with pytest.raises(ValueError):
        NaiveBayes(n_min=4, n_max=2)


def test_empty_vectors_and_probabilities() -> None:
    assert math.isclose(cosine({}, {"a": 1.0}), 0.0, rel_tol=1e-9)
    assert math.isclose(margin({}), 0.0, rel_tol=1e-9)
    assert decide({}, 0.0, 0.0) is False
