from classic_nlp.preprocess import basic_clean, preprocess, vocabulary_size


def test_basic_clean_removes_math_urls_and_symbols():
    text = "We prove $O(n^2)$ bounds! See https://arxiv.org/abs/1234 (Sec. 3)."
    assert basic_clean(text) == "we prove bounds see sec"


def test_negations_are_kept():
    assert "not" in preprocess("This scheme doesn't leak the key.")


def test_stopwords_removed_and_words_lemmatized():
    tokens = preprocess("The attacks were breaking the encryption schemes.")
    assert "the" not in tokens
    assert "attack" in tokens and "scheme" in tokens


def test_vocabulary_size_counts_unique_tokens():
    assert vocabulary_size([["a", "b"], ["b", "c"]]) == 3
