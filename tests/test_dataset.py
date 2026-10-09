import pandas as pd
import pytest

from classic_nlp.dataset import LABELS, encode_labels, load_splits


def make_papers(path):
    rows = [{"arxiv_id": f"2410.{i:05d}v1", "category": LABELS[i % 4],
             "clean_text": f"word{i % 4} text"} for i in range(40)]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_encode_labels_uses_fixed_order():
    assert list(encode_labels(pd.Series(["cs.CL", "cs.LG", "cs.CR"]))) == [0, 3, 1]


def test_encode_labels_rejects_unknown_category():
    with pytest.raises(ValueError):
        encode_labels(pd.Series(["math.AG"]))


def test_split_is_stratified_and_saved(tmp_path):
    data, split = tmp_path / "papers.csv", tmp_path / "split.csv"
    make_papers(data)
    train, test = load_splits(data, split)
    assert len(train) == 32 and len(test) == 8
    assert sorted(test["category"].value_counts()) == [2, 2, 2, 2]   # 20% of each category
    assert split.exists()


def test_split_is_reused_on_the_next_run(tmp_path):
    data, split = tmp_path / "papers.csv", tmp_path / "split.csv"
    make_papers(data)
    _, first = load_splits(data, split)
    _, second = load_splits(data, split)
    assert list(first["arxiv_id"]) == list(second["arxiv_id"])
