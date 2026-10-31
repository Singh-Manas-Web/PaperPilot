import numpy as np
import pytest

pytest.importorskip("keras")           # skip these tests if TensorFlow/Keras isn't installed

from classic_nlp.rnn import build_model  # noqa: E402
from classic_nlp.sequences import build_vocab, encode  # noqa: E402

TEXTS = {
    0: "language model token transformer translation text",
    1: "encryption key attack lattice cryptography secure",
    2: "image pixel segmentation camera detection vision",
    3: "gradient optimization loss training generalization learning",
}
def make_toy_data(n_per_topic=30, seed=0):
    """Random 2-6 word subsets of each topic's words, in random order."""
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for label, sentence in TEXTS.items():
        words = sentence.split()
        for _ in range(n_per_topic):
            k = rng.integers(2, len(words) + 1)
            texts.append(" ".join(rng.choice(words, size=k, replace=False)))
            labels.append(label)
    return texts, np.array(labels)


def test_parameter_count_matches_the_formulas():
    model = build_model(vocab_size=100, embed_dim=8, units=4, maxlen=10)
    # Embedding 100*8 = 800, SimpleRNN 4*(4+8+1) = 52, Dense 4*4+4 = 20
    assert model.count_params() == 872


def test_outputs_are_probabilities_over_four_categories():
    model = build_model(vocab_size=50, embed_dim=4, units=4, maxlen=6)
    ids = np.random.randint(0, 50, size=(3, 6)).astype("int32")
    probs = model.predict(ids, verbose=0)
    assert probs.shape == (3, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_rnn_learns_obvious_topics():
    import keras
    keras.utils.set_random_seed(0)
    texts, labels = make_toy_data()
    vocab = build_vocab(texts, min_count=1)
    model = build_model(len(vocab) + 2, embed_dim=16, units=16, dropout=0.0,
                        learning_rate=0.01, maxlen=8)
    model.fit(encode(texts, vocab, maxlen=8), labels, epochs=40, batch_size=8, verbose=0)
    pred = model.predict(encode(["lattice encryption key", "image segmentation"], vocab,
                                maxlen=8), verbose=0).argmax(axis=1)
    assert list(pred) == [1, 2]