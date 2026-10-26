"""
Neural-network (ANN) paper-category classifier (Day 4 of the roadmap).

TF-IDF vector -> Dense(128, relu) -> Dropout(0.5) -> Dense(4, softmax)

Commands (run from the project folder):
    python -m classic_nlp.ann train
    python -m classic_nlp.ann predict "We propose a lattice-based encryption scheme..."
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's info messages

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from classic_nlp import results
from classic_nlp.baseline import tfidf
from classic_nlp.dataset import LABELS, load_splits

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "ann.keras"
VECTORIZER_FILE = ROOT / "models" / "ann_tfidf.joblib"
MODEL_NAME = "TF-IDF + ANN (Dense 128, dropout 0.5)"
MAX_FEATURES = 20_000          # keep the 20,000 most frequent terms (caps the input size)


def build_model(n_features, hidden_units=128, dropout=0.5, n_classes=len(LABELS),
                learning_rate=1e-3):
    """A one-hidden-layer network that outputs one probability per category."""
    import keras
    from keras import layers

    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def to_dense(vectorizer, texts):
    """Sparse TF-IDF matrix -> dense float32 array, which Keras Dense layers expect."""
    return vectorizer.transform(texts).toarray().astype("float32")


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))
    print("Confusion matrix (rows = true category, columns = predicted):")
    print("        " + "  ".join(f"{c:>6s}" for c in LABELS))
    for label, row in zip(LABELS, confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))):
        print(f"{label:>6s}  " + "  ".join(f"{v:6d}" for v in row))


def train(epochs=100, batch_size=32, patience=5, seed=42):
    import keras
    from keras.callbacks import EarlyStopping

    keras.utils.set_random_seed(seed)                   # same weights + shuffling every run
    train_df, test_df = load_splits()                   # the SAME split as Day 3

    # Validation set carved from TRAIN, stratified. NOT validation_split: that takes the
    # last rows, and our CSV is ordered by category, so it would be mostly cs.LG.
    fit_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"],
                                      random_state=seed)
    print(f"Fit: {len(fit_df)}   Validation: {len(val_df)}   Test: {len(test_df)} papers")

    vectorizer = tfidf().set_params(max_features=MAX_FEATURES)
    vectorizer.fit(fit_df["clean_text"])                # TF-IDF learned from fit papers only
    X_fit = to_dense(vectorizer, fit_df["clean_text"])
    X_val = to_dense(vectorizer, val_df["clean_text"])
    X_test = to_dense(vectorizer, test_df["clean_text"])
    y_fit = fit_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    model = build_model(X_fit.shape[1])
    model.summary()

    # Same idea as class_weight="balanced" in the Day 3 logistic regression
    weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_fit)
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=epochs,
                        batch_size=batch_size, class_weight=dict(enumerate(weights)),
                        callbacks=[early_stop], verbose=2)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\nBest epoch: {best_epoch} "
          f"(val_loss {min(history.history['val_loss']):.3f}); weights restored from it")

    # The test set is used exactly once, here
    pred = model.predict(X_test, verbose=0).argmax(axis=1)
    acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")
    results.record(MODEL_NAME, acc, f1, notes=f"Day 4; best epoch {best_epoch}")
    print(f"\nTest accuracy {acc:.3f}   Test macro-F1 {f1:.3f}\n")
    print_report(y_test, pred)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"\nSaved {MODEL_FILE.relative_to(ROOT)} and {VECTORIZER_FILE.relative_to(ROOT)}; "
          "scores written to results.md")


def predict(text):
    import keras
    from classic_nlp.preprocess import preprocess      # loads NLTK only when needed

    model = keras.models.load_model(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    probs = model.predict(to_dense(vectorizer, [" ".join(preprocess(text))]), verbose=0)[0]
    for k in np.argsort(probs)[::-1]:
        print(f"  {LABELS[k]}  {probs[k]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="ANN category classifier")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train the ANN and evaluate it on the test set")
    p = sub.add_parser("predict", help="predict the category of an abstract")
    p.add_argument("text")
    args = parser.parse_args()
    if args.command == "train":
        train()
    else:
        predict(args.text)


if __name__ == "__main__":
    main()