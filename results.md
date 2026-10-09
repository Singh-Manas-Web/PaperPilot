# Results: paper category classification

Same stratified 80/20 split for every model (`data/processed/split.csv`). Macro-F1 averages the four categories equally, so the smaller cs.CL class counts as much as the others.

| Model | Test accuracy | Test macro-F1 | 5-fold CV macro-F1 (train) | Date | Notes |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.795 | 0.794 | 0.790 | 2026-10-09 | Day 3 baseline |
| TF-IDF + Naive Bayes | 0.757 | 0.750 | 0.734 | 2026-10-09 | Day 3 baseline |
| Majority class (dummy) | 0.270 | 0.106 | 0.106 | 2026-10-09 | Day 3 baseline |
