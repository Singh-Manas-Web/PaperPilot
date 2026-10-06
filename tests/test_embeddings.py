import pytest

pytest.importorskip("gensim")          # skip these tests if gensim isn't installed

from classic_nlp.embeddings import similar_terms, train  # noqa: E402

SENTENCES = [["lattice", "encryption", "key"], ["encryption", "key", "secure"],
             ["image", "pixel", "network"], ["pixel", "image", "segmentation"]] * 25


def test_train_gives_vectors_of_the_requested_size():
    model = train(SENTENCES, vector_size=16, min_count=1, epochs=5)
    assert model.wv["encryption"].shape == (16,)


def test_unknown_word_returns_empty_list_instead_of_error():
    model = train(SENTENCES, vector_size=16, min_count=1, epochs=5)
    assert similar_terms(model, "blockchain") == []
    assert len(similar_terms(model, "Encryption", topn=3)) == 3   # case-insensitive
