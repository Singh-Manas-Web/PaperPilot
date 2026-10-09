"""
Keeps one results table for every model PaperPilot trains.

Scores are stored in results/metrics.json and rendered to results.md (both committed),
so the table grows day by day: baseline today, ANN tomorrow, then RNN, LSTM, ...
"""
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "results" / "metrics.json"
RESULTS_MD = ROOT / "results.md"


def load(path=RESULTS_JSON):
    return json.loads(Path(path).read_text()) if Path(path).exists() else []


def record(model, accuracy, macro_f1, cv_f1=None, notes="",
           json_path=RESULTS_JSON, md_path=RESULTS_MD):
    """Add or replace one model's scores, then rewrite results.md."""
    rows = [r for r in load(json_path) if r["model"] != model]
    rows.append({"model": model, "accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4),
                 "cv_f1": None if cv_f1 is None else round(cv_f1, 4),
                 "date": date.today().isoformat(), "notes": notes})
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(rows, indent=2))
    Path(md_path).write_text(render(rows))
    return rows


def render(rows):
    """Markdown table, best macro-F1 first."""
    lines = ["# Results: paper category classification", "",
             "Same stratified 80/20 split for every model (`data/processed/split.csv`). "
             "Macro-F1 averages the four categories equally, so the smaller cs.CL class counts as much as the others.",
             "",
             "| Model | Test accuracy | Test macro-F1 | 5-fold CV macro-F1 (train) | Date | Notes |",
             "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["macro_f1"], reverse=True):
        cv = "—" if r["cv_f1"] is None else f"{r['cv_f1']:.3f}"
        lines.append(f"| {r['model']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {cv} | "
                     f"{r['date']} | {r['notes']} |")
    return "\n".join(lines) + "\n"
