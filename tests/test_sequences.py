import numpy as np

from classic_nlp.sequences import OOV, PAD, build_vocab, encode, load_vocab, save_vocab


def test_vocab_reserves_pad_and_oov_and_drops_rare_words():
    vocab = build_vocab(["lattice key key", "key encryption lattice"], min_count=2)
    assert vocab == {"key": 2, "lattice": 3}            # "encryption" appears once -> dropped
    assert PAD not in vocab.values() and OOV not in vocab.values()


def test_vocab_size_is_capped():
    vocab = build_vocab(["a a a b b c c"], max_words=4, min_count=1)
    assert vocab == {"a": 2, "b": 3}                     # 4 IDs in total: PAD, OOV, a, b


def test_encode_pads_in_front_and_marks_unknown_words():
    vocab = {"lattice": 2, "key": 3}
    out = encode(["lattice zebra key"], vocab, maxlen=5)
    assert out.tolist() == [[PAD, PAD, 2, OOV, 3]]
    assert out.dtype == np.int32


def test_encode_keeps_the_beginning_of_long_texts():
    out = encode(["lattice zebra key"], {"lattice": 2, "key": 3}, maxlen=2)
    assert out.tolist() == [[2, OOV]]


def test_vocab_save_and_load_round_trip(tmp_path):
    vocab = {"lattice": 2, "key": 3}
    save_vocab(vocab, tmp_path / "vocab.json")
    assert load_vocab(tmp_path / "vocab.json") == vocab