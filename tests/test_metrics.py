import math

from mlops.eval.metrics import binary_metrics, roc_auc, threshold_for_sensitivity


def test_roc_auc_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(y_true, y_score) == 1.0


def test_roc_auc_random_is_half():
    y_true = [0, 1, 0, 1]
    y_score = [0.5, 0.5, 0.5, 0.5]  # aucun pouvoir discriminant
    assert abs(roc_auc(y_true, y_score) - 0.5) < 1e-9


def test_binary_metrics_confusion_and_rates():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.3, 0.2, 0.8]  # seuil 0.5 → pred [1,0,0,1]
    m = binary_metrics(y_true, y_score, threshold=0.5)
    assert m["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert m["sensitivity"] == 0.5
    assert m["specificity"] == 0.5
    assert m["n"] == 4


def test_threshold_for_sensitivity_targets_recall():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]
    thr = threshold_for_sensitivity(y_true, y_score, target=1.0)
    m = binary_metrics(y_true, y_score, thr)
    assert m["sensitivity"] == 1.0


def test_auc_nan_when_single_class():
    assert math.isnan(roc_auc([1, 1, 1], [0.2, 0.5, 0.9]))
