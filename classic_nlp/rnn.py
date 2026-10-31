"""
SimpleRNN paper-category classifier (Day 5 of the roadmap).

word IDs -> Embedding(vocab, 64, mask_zero) -> SimpleRNN(64) -> Dropout(0.3) -> Dense(4, softmax)

Commands (run from the project folder):
    python -m classic_nlp.rnn train                  # read up to 200 words per abstract
    python -m classic_nlp.rnn train --maxlen 50      # read only the first 50 words
    python -m classic_nlp.rnn predict "We propose a lattice-based encryption scheme..."
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's info messages

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from classic_nlp import results
from classic_nlp.dataset import LABELS, load_splits
from classic_nlp.evaluate import report
from classic_nlp.sequences import MAXLEN, build_vocab, encode, load_vocab, save_vocab

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "rnn.keras"
VOCAB_FILE = ROOT / "models" / "rnn_vocab.json"
MODEL_NAME = "Embedding + SimpleRNN (64 units)"


def build_model(vocab_size, embed_dim=64, units=64, dropout=0.3, n_classes=len(LABELS),
                learning_rate=1e-3, maxlen=MAXLEN):
    """Embedding learned from scratch, read word by word by a SimpleRNN."""
    import keras
    from keras import layers

    model = keras.Sequential([
        keras.Input(shape=(maxlen,), dtype="int32"),
        layers.Embedding(vocab_size, embed_dim, mask_zero=True),   # ID 0 = padding, skipped
        layers.SimpleRNN(units),                                   # tanh (the default)
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(epochs=30, batch_size=32, patience=3, seed=42, maxlen=MAXLEN):
    import keras
    from keras.callbacks import EarlyStopping

    keras.utils.set_random_seed(seed)
    train_df, test_df = load_splits()                   # the SAME split as Days 3-4
    # the same stratified validation carve-out as the ANN (Day 4)
    fit_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"],
                                      random_state=seed)
    print(f"Fit: {len(fit_df)}   Validation: {len(val_df)}   Test: {len(test_df)} papers")

    vocab = build_vocab(fit_df["clean_text"])           # vocabulary from fit papers only
    X_fit = encode(fit_df["clean_text"], vocab, maxlen)
    X_val = encode(val_df["clean_text"], vocab, maxlen)
    X_test = encode(test_df["clean_text"], vocab, maxlen)
    y_fit = fit_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    print(f"Vocabulary: {len(vocab) + 2} IDs   Sequences: {X_fit.shape}")

    model = build_model(vocab_size=len(vocab) + 2, maxlen=maxlen)
    model.summary()

    weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_fit)
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=epochs,
                        batch_size=batch_size, class_weight=dict(enumerate(weights)),
                        callbacks=[early_stop], verbose=2)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\nBest epoch: {best_epoch} "
          f"(val_loss {min(history.history['val_loss']):.3f}); weights restored from it\n")

    # The test set is used exactly once, here
    pred = model.predict(X_test, verbose=0).argmax(axis=1)
    acc, f1 = report(y_test, pred)
    name = MODEL_NAME if maxlen == MAXLEN else f"{MODEL_NAME[:-1]}, first {maxlen} words)"
    results.record(name, acc, f1, notes=f"Day 5; best epoch {best_epoch}")
    print(f"\nTest accuracy {acc:.3f}   Test macro-F1 {f1:.3f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)
    save_vocab(vocab, VOCAB_FILE)
    print(f"\nSaved {MODEL_FILE.relative_to(ROOT)} and {VOCAB_FILE.relative_to(ROOT)}; "
          "scores written to results.md")


def predict_labels(texts):
    """Category IDs for already-cleaned texts (used by classic_nlp.evaluate)."""
    import keras

    model = keras.models.load_model(MODEL_FILE)
    maxlen = model.input_shape[1]                       # the length it was trained with
    x = encode(texts, load_vocab(VOCAB_FILE), maxlen)
    return model.predict(x, verbose=0).argmax(axis=1)


def predict(text):
    import keras
    from classic_nlp.preprocess import preprocess      # loads NLTK only when needed

    model = keras.models.load_model(MODEL_FILE)
    x = encode([" ".join(preprocess(text))], load_vocab(VOCAB_FILE), model.input_shape[1])
    probs = model.predict(x, verbose=0)[0]
    for k in np.argsort(probs)[::-1]:
        print(f"  {LABELS[k]}  {probs[k]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="SimpleRNN category classifier")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train the RNN and evaluate it on the test set")
    t.add_argument("--maxlen", type=int, default=MAXLEN,
                   help="words per abstract to read (default: %(default)s)")
    p = sub.add_parser("predict", help="predict the category of an abstract")
    p.add_argument("text")
    args = parser.parse_args()
    if args.command == "train":
        train(maxlen=args.maxlen)
    else:
        predict(args.text)


if __name__ == "__main__":
    main()