"""
Baseline paper-category classifiers (Day 3 of the roadmap).

Commands (run from the project folder):
    python -m classic_nlp.baseline train
    python -m classic_nlp.baseline predict "We propose a lattice-based homomorphic encryption scheme..."
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from classic_nlp import results
from classic_nlp.dataset import LABELS, load_splits

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "baseline_lr.joblib"


def tfidf():
    """Same TF-IDF settings as the Day 2 search engine."""
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)


def candidate_models():
    return {
        "Majority class (dummy)": DummyClassifier(strategy="most_frequent"),
        "TF-IDF + Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
    }


def top_terms(pipeline, n=8):
    """The words that push most strongly towards each category (logistic regression only)."""
    names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_
    return {LABELS[k]: list(names[np.argsort(coefs[k])[::-1][:n]]) for k in range(len(LABELS))}


def train(cv_folds=5):
    # The dummy model never predicts 3 of the 4 classes, so precision is undefined for them.
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    train_df, test_df = load_splits()
    X_train, y_train = train_df["clean_text"], train_df["label"]
    X_test, y_test = test_df["clean_text"], test_df["label"]
    print(f"Train: {len(train_df)} papers   Test: {len(test_df)} papers")
    print("Test papers per category:",
          dict(zip(LABELS, np.bincount(y_test, minlength=len(LABELS)).tolist())))

    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fitted = {}
    print(f"\n{'Model':30s} {'CV macro-F1':>12s} {'Test acc':>9s} {'Test macro-F1':>14s}")
    for name, model in candidate_models().items():
        cv_f1 = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1_macro").mean()
        model.fit(X_train, y_train)                      # Pipeline: TF-IDF fitted on TRAIN only
        pred = model.predict(X_test)
        acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")
        results.record(name, acc, f1, cv_f1, notes="Day 3 baseline")
        fitted[name] = model
        print(f"{name:30s} {cv_f1:12.3f} {acc:9.3f} {f1:14.3f}")

    best = fitted["TF-IDF + Logistic Regression"]
    pred = best.predict(X_test)
    print("\nTF-IDF + Logistic Regression, per category (test set):")
    print(classification_report(y_test, pred, target_names=LABELS, digits=3))
    print("Confusion matrix (rows = true category, columns = predicted):")
    print("        " + "  ".join(f"{c:>6s}" for c in LABELS))
    for label, row in zip(LABELS, confusion_matrix(y_test, pred)):
        print(f"{label:>6s}  " + "  ".join(f"{v:6d}" for v in row))
    print("\nWords that most indicate each category:")
    for label, words in top_terms(best).items():
        print(f"  {label}: {', '.join(words)}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_FILE)
    print(f"\nSaved {MODEL_FILE.relative_to(ROOT)}; scores written to results.md")


def predict(text):
    from classic_nlp.preprocess import preprocess        # imported here: loads NLTK only when needed
    model = joblib.load(MODEL_FILE)
    probs = model.predict_proba([" ".join(preprocess(text))])[0]
    for k in np.argsort(probs)[::-1]:
        print(f"  {LABELS[k]}  {probs[k]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Baseline category classifiers")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train and evaluate the baselines")
    p = sub.add_parser("predict", help="predict the category of an abstract")
    p.add_argument("text")
    args = parser.parse_args()
    if args.command == "train":
        train()
    else:
        predict(args.text)


if __name__ == "__main__":
    main()
