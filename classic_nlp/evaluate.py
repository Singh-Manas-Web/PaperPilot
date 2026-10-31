"""
One evaluation report for every PaperPilot classifier (Day 5 of the roadmap).

Commands (run from the project folder, after training the model):
    python -m classic_nlp.evaluate lr      # TF-IDF + logistic regression (Day 3)
    python -m classic_nlp.evaluate ann     # TF-IDF + ANN (Day 4)
    python -m classic_nlp.evaluate rnn     # Embedding + SimpleRNN (Day 5)
"""
import argparse
import importlib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from classic_nlp.dataset import LABELS, load_splits

# model name -> module that has a predict_labels(texts) function and a saved model
MODELS = {"lr": "classic_nlp.baseline", "ann": "classic_nlp.ann", "rnn": "classic_nlp.rnn"}


def report(y_true, y_pred, labels=LABELS):
    """Print per-category scores and a confusion matrix; return (accuracy, macro-F1)."""
    ids = list(range(len(labels)))
    print(classification_report(y_true, y_pred, labels=ids, target_names=labels,
                                digits=3, zero_division=0))
    print("Confusion matrix (rows = true category, columns = predicted):")
    print("        " + "  ".join(f"{c:>6s}" for c in labels))
    for label, row in zip(labels, confusion_matrix(y_true, y_pred, labels=ids)):
        print(f"{label:>6s}  " + "  ".join(f"{v:6d}" for v in row))
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def predictor(name):
    """The predict_labels function of a saved model (imports only that model's code)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODELS)}")
    return importlib.import_module(MODELS[name]).predict_labels


def evaluate(name):
    _, test_df = load_splits()                  # the same test papers as every model
    pred = predictor(name)(test_df["clean_text"])
    print(f"Model: {name}   ({len(test_df)} test papers)\n")
    acc, f1 = report(test_df["label"].to_numpy(), pred)
    print(f"\nTest accuracy {acc:.3f}   Test macro-F1 {f1:.3f}")
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved classifier on the test set")
    parser.add_argument("model", choices=list(MODELS))
    evaluate(parser.parse_args().model)


if __name__ == "__main__":
    main()