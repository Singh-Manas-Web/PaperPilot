import numpy as np
import pytest

pytest.importorskip("keras")           # skip these tests if TensorFlow/Keras isn't installed

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from classic_nlp.ann import build_model, to_dense  # noqa: E402

TEXTS = {
    0: "language model token transformer translation text",
    1: "encryption key attack lattice cryptography secure",
    2: "image pixel segmentation camera detection vision",
    3: "gradient optimization loss training generalization learning",
}
X_TEXT = [TEXTS[k] for k in range(4) for _ in range(10)]
Y = np.array([k for k in range(4) for _ in range(10)])


def test_parameter_count_matches_the_formula():
    model = build_model(n_features=100, hidden_units=8)
    # hidden: 100*8 + 8 = 808, output: 8*4 + 4 = 36
    assert model.count_params() == 844


def test_outputs_are_probabilities_over_four_categories():
    model = build_model(n_features=10, hidden_units=4)
    probs = model.predict(np.random.rand(3, 10).astype("float32"), verbose=0)
    assert probs.shape == (3, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_ann_learns_obvious_topics():
    import keras
    keras.utils.set_random_seed(0)
    vectorizer = TfidfVectorizer().fit(X_TEXT)
    model = build_model(n_features=len(vectorizer.vocabulary_), hidden_units=16, dropout=0.0,
                        learning_rate=0.01)
    model.fit(to_dense(vectorizer, X_TEXT), Y, epochs=30, batch_size=8, verbose=0)
    pred = model.predict(to_dense(vectorizer, ["lattice encryption key", "image segmentation"]),
                         verbose=0).argmax(axis=1)
    assert list(pred) == [1, 2]