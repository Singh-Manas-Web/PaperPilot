"""
TF-IDF keyword search over the cleaned arXiv abstracts (Day 2 of the roadmap).

Commands (run from the project folder):
    python -m classic_nlp.search build                       # fit TF-IDF, save model + keywords
    python -m classic_nlp.search query "lattice based encryption"
    python -m classic_nlp.search keywords 0                  # top keywords of paper in row 0
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from classic_nlp.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "abstracts_clean.csv"
KEYWORDS_FILE = ROOT / "data" / "processed" / "keywords.csv"
MODEL_FILE = ROOT / "models" / "tfidf_search.joblib"


class TfidfSearch:
    """Ranks papers by cosine similarity between TF-IDF vectors."""

    def __init__(self, ngram_range=(1, 2), min_df=2, max_df=0.8):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,   # single words + two-word phrases ("neural network")
            min_df=min_df,             # ignore terms found in fewer than min_df papers
            max_df=max_df,             # ignore terms found in more than 80% of papers
            sublinear_tf=True,         # 1 + log(count): repeats matter less
        )
        self.matrix = None             # sparse matrix, one row per paper
        self.papers = None             # titles, ids, categories (row-aligned with matrix)

    def fit(self, papers):
        """Learn the vocabulary and IDF weights from the papers' clean_text column."""
        self.papers = papers.reset_index(drop=True)[["arxiv_id", "title", "category"]]
        self.matrix = self.vectorizer.fit_transform(papers["clean_text"].fillna(""))
        return self

    def search(self, query, top_k=5):
        """Return the top_k papers most similar to the query, best first."""
        query_text = " ".join(preprocess(query))            # same cleaning as the abstracts
        query_vector = self.vectorizer.transform([query_text])
        if query_vector.nnz == 0:                           # no known words in the query
            return self.papers.iloc[0:0].assign(score=[])
        # Rows are L2-normalised, so the dot product IS the cosine similarity.
        scores = linear_kernel(query_vector, self.matrix).ravel()
        best = np.argsort(scores)[::-1][:top_k]
        best = best[scores[best] > 0]                       # drop papers with zero overlap
        results = self.papers.loc[best].copy()
        results["score"] = scores[best].round(3)
        return results.reset_index(drop=True)

    def top_keywords(self, row, n=10):
        """The n highest-weighted terms of one paper, as (term, weight) pairs."""
        weights = self.matrix[row].toarray().ravel()
        best = np.argsort(weights)[::-1][:n]
        names = self.vectorizer.get_feature_names_out()
        return [(names[i], round(float(weights[i]), 3)) for i in best if weights[i] > 0]

    def save(self, path=MODEL_FILE):
        """Save only the fitted parts (plain scikit-learn / pandas objects)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix,
                     "papers": self.papers}, path)

    @classmethod
    def load(cls, path=MODEL_FILE):
        state = joblib.load(path)
        engine = cls()
        engine.vectorizer, engine.matrix, engine.papers = (
            state["vectorizer"], state["matrix"], state["papers"])
        return engine


def build():
    papers = pd.read_csv(DATA_FILE)
    engine = TfidfSearch().fit(papers)
    engine.save()
    n_docs, n_terms = engine.matrix.shape
    density = engine.matrix.nnz / (n_docs * n_terms)
    print(f"TF-IDF matrix: {n_docs} papers x {n_terms:,} terms "
          f"({100 * density:.2f}% non-zero)")

    keywords = ["; ".join(term for term, _ in engine.top_keywords(i)) for i in range(n_docs)]
    out = papers[["arxiv_id", "title", "category"]].assign(keywords=keywords)
    out.to_csv(KEYWORDS_FILE, index=False)
    print(f"Saved model to {MODEL_FILE.relative_to(ROOT)} and keywords to "
          f"{KEYWORDS_FILE.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="TF-IDF search over arXiv abstracts")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="fit TF-IDF and save the model + keywords")
    q = sub.add_parser("query", help="search the papers")
    q.add_argument("text")
    q.add_argument("--top-k", type=int, default=5)
    k = sub.add_parser("keywords", help="top keywords of one paper")
    k.add_argument("row", type=int)
    args = parser.parse_args()

    if args.command == "build":
        build()
        return
    engine = TfidfSearch.load()
    if args.command == "query":
        results = engine.search(args.text, args.top_k)
        if results.empty:
            print("No matching papers (none of the query words are in the vocabulary).")
        for rank, r in enumerate(results.itertuples(), start=1):
            print(f"{rank}. [{r.score:.3f}] ({r.category}) {r.title}")
    else:
        print(engine.papers.loc[args.row, "title"])
        for term, weight in engine.top_keywords(args.row):
            print(f"  {weight:.3f}  {term}")


if __name__ == "__main__":
    main()
