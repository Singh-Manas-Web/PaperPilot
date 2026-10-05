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
git clone https://github.com/<your-username>/paperpilot.git
cd paperpilot
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python scripts/download_arxiv.py          # -> data/raw/arxiv_abstracts.csv (~1 min)
python -m classic_nlp.preprocess          # -> data/processed/abstracts_clean.csv
pytest                                    # run the tests
```

## Text preprocessing pipeline

`classic_nlp/preprocess.py` cleans each abstract:

1. Lowercase
2. Remove LaTeX math (`$...$`), URLs and non-letter characters; expand "n't" to "not"
3. Tokenize with NLTK `word_tokenize`
4. POS-tag, then lemmatize with WordNet using the tag (or stem with Snowball/Porter via `--method`)
5. Remove English stopwords plus research boilerplate words ("paper", "propose", "method"), but **keep negations** ("not", "no")

## Project structure

```
paperpilot/
├── classic_nlp/       # Week 1: preprocessing, search, classifiers
├── scripts/           # data download
├── tests/             # pytest tests
├── data/              # gitignored: raw and processed data
├── requirements.txt
└── PROGRESS.md        # daily log
```
