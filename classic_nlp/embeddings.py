"""
Word2Vec "similar terms" for PaperPilot (Day 2 of the roadmap).

Commands (run from the project folder):
    python -m classic_nlp.embeddings train
    python -m classic_nlp.embeddings similar encryption
"""
import argparse
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "abstracts_clean.csv"
MODEL_FILE = ROOT / "models" / "word2vec.model"


def train(token_lists, vector_size=100, window=5, min_count=3, sg=1, epochs=30, seed=42):
    """Train Word2Vec on a list of token lists.

    sg=1 uses Skip-gram, which works better than CBOW on a small corpus and for rare
    words (Day 2 notes, Module 8.6). window and vector_size are separate settings.
    """
    return Word2Vec(
        sentences=token_lists,
        vector_size=vector_size,   # length of each word vector
        window=window,             # context words on each side
        min_count=min_count,       # skip words seen fewer than min_count times
        sg=sg,                     # 1 = Skip-gram, 0 = CBOW
        negative=5,                # negative sampling
        epochs=epochs,             # small corpus -> more passes
        workers=4,
        seed=seed,
    )


def similar_terms(model, word, topn=10):
    """Nearest words by cosine similarity; [] if the word was never seen (no KeyError)."""
    word = word.lower()
    if word not in model.wv:
        return []
    return [(w, round(float(s), 3)) for w, s in model.wv.most_similar(word, topn=topn)]


def main():
    parser = argparse.ArgumentParser(description="Word2Vec similar terms")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train Word2Vec on the cleaned abstracts")
    s = sub.add_parser("similar", help="show words similar to WORD")
    s.add_argument("word")
    s.add_argument("--topn", type=int, default=10)
    args = parser.parse_args()

    if args.command == "train":
        papers = pd.read_csv(DATA_FILE)
        token_lists = [str(text).split() for text in papers["clean_text"].fillna("")]
        model = train(token_lists)
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_FILE))
        total = sum(len(t) for t in token_lists)
        print(f"Trained on {len(token_lists)} abstracts ({total:,} tokens); "
              f"vocabulary {len(model.wv):,} words x {model.wv.vector_size} dimensions")
        print(f"Saved to {MODEL_FILE.relative_to(ROOT)}")
    else:
        model = Word2Vec.load(str(MODEL_FILE))
        results = similar_terms(model, args.word, args.topn)
        if not results:
            print(f"'{args.word}' is not in the vocabulary (seen fewer than "
                  f"{model.min_count} times, or not at all).")
        for word, score in results:
            print(f"  {score:.3f}  {word}")


if __name__ == "__main__":
    main()
