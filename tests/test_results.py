from classic_nlp import results


def test_record_replaces_same_model_and_sorts_by_f1(tmp_path):
    js, md = tmp_path / "metrics.json", tmp_path / "results.md"
    results.record("A", 0.80, 0.70, json_path=js, md_path=md)
    results.record("B", 0.90, 0.85, json_path=js, md_path=md)
    rows = results.record("A", 0.82, 0.75, json_path=js, md_path=md)   # re-run of A
    assert len(rows) == 2
    table = md.read_text()
    assert table.index("| B |") < table.index("| A |")                # best first
    assert "0.750" in table and "0.700" not in table                   # old A score replaced
