"""
Text preprocessing for PaperPilot (Day 1 of the roadmap).

Pipeline for one abstract:
    lowercase -> remove LaTeX math, URLs, non-letters -> tokenize
    -> (POS tag + lemmatize) -> remove stopwords -> (optionally stem)

Run on the whole dataset:
    python -m classic_nlp.preprocess                 # lemmatize (default)
    python -m classic_nlp.preprocess --method snowball
"""
import argparse
import re
from pathlib import Path

import nltk
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "arxiv_abstracts.csv"
OUT_FILE = ROOT / "data" / "processed" / "abstracts_clean.csv"

NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                 "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]


def ensure_nltk_data():
    """Download the NLTK resources this module needs (only the first time)."""
    for package in NLTK_PACKAGES:
        nltk.download(package, quiet=True)


ensure_nltk_data()

from nltk.corpus import stopwords, wordnet                     # noqa: E402
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer  # noqa: E402
from nltk.tokenize import word_tokenize                        # noqa: E402

# ---- Resources are built ONCE, not inside loops ----
NEGATIONS = {"not", "no", "nor", "never"}
# Words that are everywhere in research abstracts and carry no topic signal
DOMAIN_STOPWORDS = {"paper", "propose", "proposed", "approach", "method", "result",
                    "show", "also", "use", "using", "based", "work", "et", "al"}
STOP = (set(stopwords.words("english")) - NEGATIONS) | DOMAIN_STOPWORDS
PORTER = PorterStemmer()
SNOWBALL = SnowballStemmer("english")
LEMMATIZER = WordNetLemmatizer()


def basic_clean(text):
    """Lowercase and strip everything that is not plain words."""
    text = str(text).lower()
    text = re.sub(r"\$[^$]*\$", " ", text)                # LaTeX math like $O(n^2)$
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)     # URLs
    text = re.sub(r"n't\b", " not", text)                  # "doesn't" -> "does not"
    text = re.sub(r"[^a-z\s]", " ", text)                  # keep letters only
    return re.sub(r"\s+", " ", text).strip()               # squeeze spaces


def to_wordnet_pos(treebank_tag):
    """Map a Penn Treebank tag (e.g. 'VBD') to a WordNet POS letter."""
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(
        treebank_tag[:1], wordnet.NOUN)


def preprocess(text, method="lemma"):
    """Clean one text and return a list of tokens.

    method: "lemma" (default, readable words), "snowball" or "porter" (stems).
    """
    tokens = word_tokenize(basic_clean(text))
    if method == "lemma":
        tagged = nltk.pos_tag(tokens)                      # tag BEFORE removing stopwords
        tokens = [LEMMATIZER.lemmatize(word, to_wordnet_pos(tag)) for word, tag in tagged]
    tokens = [t for t in tokens if t not in STOP and len(t) > 2]
    if method == "snowball":
        tokens = [SNOWBALL.stem(t) for t in tokens]
    elif method == "porter":
        tokens = [PORTER.stem(t) for t in tokens]
    return tokens


def vocabulary_size(token_lists):
    """Number of unique tokens across all documents."""
    return len({token for tokens in token_lists for token in tokens})


def main():
    parser = argparse.ArgumentParser(description="Preprocess arXiv abstracts")
    parser.add_argument("--method", choices=["lemma", "snowball", "porter"], default="lemma")
    args = parser.parse_args()

    df = pd.read_csv(RAW_FILE)
    df = df.dropna(subset=["abstract"]).drop_duplicates(subset=["arxiv_id"])
    print(f"Loaded {len(df)} abstracts from {RAW_FILE.name}")

    raw_tokens = [word_tokenize(str(t)) for t in df["abstract"]]
    clean_tokens = [preprocess(t, args.method) for t in df["abstract"]]
    df["clean_text"] = [" ".join(tokens) for tokens in clean_tokens]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    before, after = vocabulary_size(raw_tokens), vocabulary_size(clean_tokens)
    print(f"Vocabulary: {before:,} raw tokens -> {after:,} after cleaning "
          f"({100 * (1 - after / before):.0f}% smaller)")
    print(df["category"].value_counts().to_string())
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
