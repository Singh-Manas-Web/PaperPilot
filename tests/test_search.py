import pandas as pd

from classic_nlp.search import TfidfSearch

PAPERS = pd.DataFrame({
    "arxiv_id": ["1", "2", "3"],
    "title": ["Lattice encryption", "Image segmentation", "Language models"],
    "category": ["cs.CR", "cs.CV", "cs.CL"],
    "clean_text": [
        "lattice encryption scheme secure key lattice attack",
        "image segmentation convolutional network pixel image",
        "language model token transformer text language",
    ],
})


def make_engine():
    return TfidfSearch(ngram_range=(1, 1), min_df=1, max_df=1.0).fit(PAPERS)


def test_search_ranks_the_matching_paper_first():
    results = make_engine().search("lattice encryption")
    assert results.loc[0, "title"] == "Lattice encryption"
    assert results["score"].is_monotonic_decreasing


def test_search_with_unknown_words_returns_nothing():
    assert make_engine().search("quantum chromodynamics").empty


def test_top_keywords_come_from_the_paper():
    terms = [term for term, _ in make_engine().top_keywords(1, n=3)]
    assert "image" in terms
    assert all(term in PAPERS.loc[1, "clean_text"] for term in terms)


def test_save_and_load_round_trip(tmp_path):
    engine = make_engine()
    engine.save(tmp_path / "search.joblib")
    loaded = TfidfSearch.load(tmp_path / "search.joblib")
    assert loaded.search("image").loc[0, "title"] == "Image segmentation"
