"""
One fixed train/test split and label encoding, shared by EVERY PaperPilot classifier.

The split is created once (stratified, seed 42) and saved to data/processed/split.csv,
so the baseline today and the ANN / RNN / LSTM models later are all tested on exactly
the same papers. Otherwise their scores wouldn't be comparable.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "abstracts_clean.csv"
SPLIT_FILE = ROOT / "data" / "processed" / "split.csv"

LABELS = ["cs.CL", "cs.CR", "cs.CV", "cs.LG"]          # fixed order -> label ids 0, 1, 2, 3
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


def encode_labels(categories):
    """'cs.CR' -> 1, etc. Fails loudly if an unknown category appears."""
    ids = categories.map(LABEL_TO_ID)
    if ids.isna().any():
        unknown = sorted(set(categories[ids.isna()]))
        raise ValueError(f"Unknown categories: {unknown}")
    return ids.astype(int)


def make_split(papers, test_size=0.2, seed=42):
    """Return a Series of 'train' / 'test', stratified so each category keeps its share."""
    _, test_index = train_test_split(
        papers.index, test_size=test_size, stratify=papers["category"], random_state=seed)
    split = pd.Series("train", index=papers.index)
    split[test_index] = "test"
    return split


def load_splits(data_file=DATA_FILE, split_file=SPLIT_FILE):
    """Load the cleaned papers and return (train_df, test_df), each with a 'label' column."""
    papers = pd.read_csv(data_file, dtype={"arxiv_id": str})
    papers = papers.dropna(subset=["clean_text"]).drop_duplicates("arxiv_id").reset_index(drop=True)

    saved = (pd.read_csv(split_file, dtype={"arxiv_id": str})
             if Path(split_file).exists() else None)
    if saved is not None and set(saved["arxiv_id"]) == set(papers["arxiv_id"]):
        papers["split"] = papers["arxiv_id"].map(dict(zip(saved["arxiv_id"], saved["split"])))
    else:                                               # first run, or the data changed
        papers["split"] = make_split(papers)
        Path(split_file).parent.mkdir(parents=True, exist_ok=True)
        papers[["arxiv_id", "split"]].to_csv(split_file, index=False)

    papers["label"] = encode_labels(papers["category"])
    train = papers[papers["split"] == "train"].reset_index(drop=True)
    test = papers[papers["split"] == "test"].reset_index(drop=True)
    return train, test
