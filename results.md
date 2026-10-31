# Results: paper category classification

Same stratified 80/20 split for every model (`data/processed/split.csv`). Macro-F1 averages the four categories equally, so the smaller cs.CL class counts as much as the others.

| Model | Test accuracy | Test macro-F1 | 5-fold CV macro-F1 (train) | Date | Notes |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.795 | 0.794 | 0.790 | 2026-10-09 | Day 3 baseline |
| TF-IDF + ANN (Dense 128, dropout 0.5) | 0.792 | 0.790 | — | 2026-10-26 | Day 4; best epoch 11 |
| TF-IDF + Naive Bayes | 0.757 | 0.750 | 0.734 | 2026-10-09 | Day 3 baseline |
| Embedding + SimpleRNN (64 units, first 50 words) | 0.481 | 0.467 | — | 2026-10-31 | Day 5; best epoch 4 |
| Embedding + SimpleRNN (64 units) | 0.268 | 0.256 | — | 2026-10-31 | Day 5; best epoch 1 |
| Majority class (dummy) | 0.270 | 0.106 | 0.106 | 2026-10-09 | Day 3 baseline |
