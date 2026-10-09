import pandas as pd

from classic_nlp.baseline import candidate_models, top_terms
from classic_nlp.dataset import LABELS

TEXTS = {
    0: "language model token transformer translation text",
    1: "encryption key attack lattice cryptography secure",
    2: "image pixel segmentation camera detection vision",
    3: "gradient optimization loss training generalization learning",
}
X = pd.Series([TEXTS[k] for k in range(4) for _ in range(10)])
y = pd.Series([k for k in range(4) for _ in range(10)])


def test_logistic_regression_learns_obvious_topics():
    model = candidate_models()["TF-IDF + Logistic Regression"].fit(X, y)
    assert list(model.predict(pd.Series(["lattice encryption key", "image segmentation"]))) == [1, 2]


def test_dummy_predicts_a_single_class():
    model = candidate_models()["Majority class (dummy)"].fit(X, y)
    assert len(set(model.predict(X))) == 1


def test_top_terms_returns_words_for_every_category():
    model = candidate_models()["TF-IDF + Logistic Regression"].fit(X, y)
    terms = top_terms(model, n=3)
    assert set(terms) == set(LABELS)
    assert all(word in TEXTS[1] for term in terms["cs.CR"] for word in term.split())
