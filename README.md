# PaperPilot

An AI research assistant that reads research papers, answers questions about them with citations, searches arXiv, keeps a reading list and writes literature reviews.

Built step by step over 28 days while learning NLP and Generative AI, from classic NLP (Week 1) through RAG, agents and MCP (Week 4).

## Status

| Version | Milestone | Status |
|---|---|---|
| v0.1 | Classic NLP: preprocessing, TF-IDF search, ANN/RNN/LSTM classifiers, Streamlit app | In progress |
| v0.2 | Chat with your papers (RAG with LangChain) | Planned |
| v0.3 | Agents, tools and deployment | Planned |
| v1.0 | Multi-agent system, knowledge graph, MCP server | Planned |

## Dataset

About 2,000 recent arXiv abstracts, 500 from each of four categories:

| Category | Topic |
|---|---|
| cs.CR | Cryptography and Security |
| cs.LG | Machine Learning |
| cs.CV | Computer Vision |
| cs.CL | Computation and Language (NLP) |

Downloaded with the free [arXiv API](https://info.arxiv.org/help/api/index.html). The data folder is not committed; re-create it with the script below.

## Setup

```bash
git clone https://github.com/Singh-Manas-Web/PaperPilot.git
cd PaperPilot
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python scripts/download_arxiv.py                       # -> data/raw/arxiv_abstracts.csv (~1 min)
python -m classic_nlp.preprocess                       # -> data/processed/abstracts_clean.csv
python -m classic_nlp.search build                     # TF-IDF index + data/processed/keywords.csv
python -m classic_nlp.search query "lattice encryption"
python -m classic_nlp.search keywords 0                # top keywords of one paper
python -m classic_nlp.embeddings train                 # Word2Vec (Skip-gram) on the abstracts
python -m classic_nlp.embeddings similar encryption    # nearest words
python -m classic_nlp.baseline train                   # category classifiers -> results.md
python -m classic_nlp.baseline predict "your abstract text"
pytest -v                                              # run the tests
```

## Text preprocessing pipeline

`classic_nlp/preprocess.py` cleans each abstract:

1. Lowercase
2. Remove LaTeX math (`$...$`), URLs and non-letter characters; expand "n't" to "not"
3. Tokenize with NLTK `word_tokenize`
4. POS-tag, then lemmatize with WordNet using the tag (or stem with Snowball/Porter via `--method`)
5. Remove English stopwords plus research boilerplate words ("paper", "propose", "method"), but **keep negations** ("not", "no")

## Keyword search (TF-IDF)

`classic_nlp/search.py` turns every cleaned abstract into a TF-IDF vector (unigrams + bigrams, `sublinear_tf`, terms in 2+ papers and under 80% of papers). A query is cleaned with the same pipeline, vectorized, and papers are ranked by cosine similarity. The same TF-IDF weights give each paper's top 10 keywords.

## Similar terms (Word2Vec)

`classic_nlp/embeddings.py` trains a Skip-gram Word2Vec model (100 dimensions, window 5) on the cleaned abstracts with Gensim, so you can look up words used in similar contexts, e.g. `similar encryption`.

## Category classification

Predicts a paper's arXiv category (cs.CL / cs.CR / cs.CV / cs.LG) from its cleaned abstract.

- `classic_nlp/dataset.py` creates **one stratified 80/20 train/test split** (seed 42), saved to `data/processed/split.csv`, so every model in this project is evaluated on the same papers.
- `classic_nlp/baseline.py` compares a majority-class dummy, TF-IDF + Naive Bayes and TF-IDF + Logistic Regression (`class_weight="balanced"`), using 5-fold cross-validation on the training set and macro-F1 on the test set.
- `classic_nlp/results.py` records every model's scores in `results/metrics.json` and renders **[results.md](results.md)**.

## Project structure

```
paperpilot/
├── classic_nlp/       # Week 1: preprocessing, TF-IDF search, Word2Vec, classifiers
├── scripts/           # data download
├── tests/             # pytest tests
├── pytest.ini
├── data/              # gitignored: raw and processed data
├── models/            # gitignored: saved TF-IDF and Word2Vec models
├── requirements.txt
├── results/           # metrics.json (all model scores)
├── results.md         # results table
└── PROGRESS.md        # daily log
```
