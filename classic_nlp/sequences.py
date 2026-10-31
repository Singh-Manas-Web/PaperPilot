"""
Turn cleaned abstracts into fixed-length integer sequences for the RNN models (Day 5).

    vocabulary: word -> ID, built from TRAINING papers only (0 = padding, 1 = unknown word)
    encode:     "lattice encryption key"  ->  [0, 0, ..., 0, 57, 12, 9]   (length MAXLEN)
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

PAD, OOV = 0, 1          # reserved IDs; real words start at 2
MAXLEN = 200             # longest cleaned abstract is 210 tokens, 99% are under 188
MAX_WORDS = 10_000       # cap on the vocabulary size (including PAD and OOV)
MIN_COUNT = 2            # a word must appear at least twice in training to get its own ID


def build_vocab(texts, max_words=MAX_WORDS, min_count=MIN_COUNT):
    """Most frequent words first; ties broken alphabetically so the IDs never change."""
    counts = Counter(word for text in texts for word in text.split())
    ranked = sorted((w for w, c in counts.items() if c >= min_count),
                    key=lambda w: (-counts[w], w))
    return {word: i + 2 for i, word in enumerate(ranked[:max_words - 2])}


def encode(texts, vocab, maxlen=MAXLEN):
    """Texts -> int32 array (n_texts, maxlen): unknown words -> OOV, zeros in FRONT,
    and if a text is too long its BEGINNING is kept (abstracts state the topic early)."""
    out = np.full((len(texts), maxlen), PAD, dtype="int32")
    for row, text in enumerate(texts):
        ids = [vocab.get(word, OOV) for word in text.split()][:maxlen]
        if ids:
            out[row, -len(ids):] = ids
    return out


def save_vocab(vocab, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(vocab), encoding="utf-8")


def load_vocab(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))