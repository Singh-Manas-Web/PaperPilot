import pytest

from classic_nlp import evaluate


def test_report_returns_accuracy_and_macro_f1(capsys):
    acc, f1 = evaluate.report([0, 1, 2, 3, 0], [0, 1, 2, 3, 1])
    assert acc == pytest.approx(0.8)
    assert f1 == pytest.approx((2 / 3 + 2 / 3 + 1 + 1) / 4)   # per-category F1, averaged
    printed = capsys.readouterr().out
    assert "Confusion matrix" in printed and "cs.CR" in printed


def test_report_handles_a_category_that_is_never_predicted(capsys):
    acc, f1 = evaluate.report([0, 1, 2, 3], [0, 0, 0, 0])
    assert acc == pytest.approx(0.25)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        evaluate.predictor("transformer")