# Progress log

One line per day: what I built and one thing I learned.

| Day | Date | Built | Learned |
|---|---|---|---|
| 1 | 2026-10-05 | Repo setup, arXiv downloader (1,850 abstracts, 4 categories), preprocessing pipeline, 4 passing tests | Lemmatization cut the vocabulary 31,749 → 13,155 (59%); Snowball cut it to 9,782 (69%) but produced non-words. Lowercase before removing stopwords, and keep "not". |
| 2 | 2026-10-06 | TF-IDF keyword search (1,850 papers × 29,152 uni+bigram terms, 0.46% non-zero), top-10 keywords per paper, Skip-gram Word2Vec (6,569 words × 100 dims), 10 passing tests | TF-IDF only matches exact words: "cipher" and "encryption" share no results. Word2Vec found real crypto links (encryption → homomorphic, FHE, LWE) but rare words like "cipher" get noisy vectors: embeddings need lots of data, hence pretrained models later. |