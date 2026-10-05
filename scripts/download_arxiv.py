"""
Download paper abstracts from the arXiv API into data/raw/arxiv_abstracts.csv.

Usage:
    python scripts/download_arxiv.py                  # 500 papers x 4 categories
    python scripts/download_arxiv.py --per-category 100

Uses only the Python standard library (no extra installs needed).
arXiv asks API users to wait ~3 seconds between requests, so the full
download (20 requests) takes about a minute.
"""
import argparse
import csv
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API_URL = "https://export.arxiv.org/api/query"
CATEGORIES = {
    "cs.CR": "Cryptography and Security",
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision",
    "cs.CL": "Computation and Language (NLP)",
}
# XML namespaces used in arXiv's Atom feed
NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}
OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "arxiv_abstracts.csv"


def fetch_page(category, start, batch_size, retries=3):
    """Download one page of results (raw XML text) for a category."""
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": batch_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    request = urllib.request.Request(f"{API_URL}?{params}",
                                     headers={"User-Agent": "PaperPilot/0.1 (student project)"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8")
        except Exception as err:                      # network hiccup: wait and retry
            print(f"  attempt {attempt} failed: {err}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not download {category} starting at {start}")


def parse_entries(xml_text, category):
    """Turn arXiv's Atom XML into a list of dictionaries (one per paper)."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", NS):
        def text(tag):
            node = entry.find(tag, NS)
            return " ".join(node.text.split()) if node is not None and node.text else ""
        arxiv_id = text("atom:id").rsplit("/abs/", 1)[-1]
        authors = [a.findtext("atom:name", default="", namespaces=NS)
                   for a in entry.findall("atom:author", NS)]
        primary = entry.find("arxiv:primary_category", NS)
        papers.append({
            "arxiv_id": arxiv_id,
            "title": text("atom:title"),
            "abstract": text("atom:summary"),
            "category": category,                     # the label we will classify on
            "primary_category": primary.get("term") if primary is not None else "",
            "published": text("atom:published")[:10],
            "authors": "; ".join(authors),
        })
    return papers


def main():
    parser = argparse.ArgumentParser(description="Download arXiv abstracts")
    parser.add_argument("--per-category", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    all_papers, seen_ids = [], set()
    for category, name in CATEGORIES.items():
        print(f"Downloading {category} ({name})")
        collected = 0
        for start in range(0, args.per_category, args.batch_size):
            batch = min(args.batch_size, args.per_category - start)
            papers = parse_entries(fetch_page(category, start, batch), category)
            for paper in papers:
                if paper["arxiv_id"] not in seen_ids:   # skip papers listed in 2 categories
                    seen_ids.add(paper["arxiv_id"])
                    all_papers.append(paper)
                    collected += 1
            print(f"  {collected} papers so far")
            time.sleep(3)                              # be polite to the arXiv API

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_papers[0].keys()))
        writer.writeheader()
        writer.writerows(all_papers)
    print(f"Saved {len(all_papers)} papers to {OUT_FILE}")


if __name__ == "__main__":
    main()
