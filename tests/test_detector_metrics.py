from types import SimpleNamespace

from mlops.eval.detector_metrics import (
    recall_confidence_curve,
    threshold_for_target_recall,
)


def test_threshold_picks_highest_conf_meeting_recall():
    # Le recall décroît quand le seuil de confiance monte.
    conf = [0.1, 0.2, 0.3, 0.4, 0.5]
    rec = [0.99, 0.95, 0.92, 0.80, 0.50]
    # target 0.90 → seuils 0.1/0.2/0.3 l'atteignent → on garde le plus haut (moins de faux positifs).
    assert threshold_for_target_recall(conf, rec, target=0.90) == 0.3


def test_threshold_returns_zero_when_target_unreachable():
    assert threshold_for_target_recall([0.1, 0.2], [0.5, 0.4], target=0.90) == 0.0


def test_recall_confidence_curve_extracts_and_averages_classes():
    # 2 classes × 3 points ; ylabel Recall, xlabel Confidence.
    res = SimpleNamespace(curves_results=[
        ([0.0, 0.5, 1.0], [[1.0, 0.8, 0.0], [0.9, 0.7, 0.1]], "Confidence", "Recall"),
        ([0.0, 0.5, 1.0], [[0.2, 0.5, 0.9]], "Confidence", "Precision"),
    ])
    conf, rec = recall_confidence_curve(res)
    assert conf == [0.0, 0.5, 1.0]
    assert rec == [0.95, 0.75, 0.05]  # moyenne des deux classes


def test_recall_confidence_curve_none_when_absent():
    assert recall_confidence_curve(SimpleNamespace(curves_results=[])) is None
