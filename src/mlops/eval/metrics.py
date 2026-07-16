# src/mlops/eval/metrics.py
# ─────────────────────────────────────────────
# Métriques cliniques d'évaluation binaire (bénin/malin) — sans dépendance
# lourde (numpy seul). En contexte médical, la SENSIBILITÉ prime (limiter les
# faux négatifs) : on rapporte sensibilité/spécificité au seuil choisi + l'AUC.
# ─────────────────────────────────────────────

from typing import Sequence

import numpy as np


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """AUC ROC via la statistique de Mann–Whitney U (moyenne des rangs), robuste aux ex æquo."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Moyenne des rangs pour les scores égaux
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.50) -> dict:
    """
    Retourne un dict de métriques au seuil donné :
    auc, sensitivity(=recall/TPR), specificity(=TNR), precision(=PPV), npv, accuracy, f1,
    et la matrice de confusion (tp, fp, tn, fn).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def _safe(num, den):
        return float(num / den) if den else float("nan")

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "auc": round(roc_auc(y_true, y_score), 4),
        "sensitivity": round(_safe(tp, tp + fn), 4),   # rappel / TPR
        "specificity": round(_safe(tn, tn + fp), 4),   # TNR
        "precision": round(_safe(tp, tp + fp), 4),     # PPV
        "npv": round(_safe(tn, tn + fn), 4),
        "accuracy": round(_safe(tp + tn, tp + tn + fp + fn), 4),
        "f1": round(_safe(2 * tp, 2 * tp + fp + fn), 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def threshold_for_sensitivity(y_true: Sequence[int], y_score: Sequence[float], target: float = 0.95) -> float:
    """Plus haut seuil atteignant au moins `target` de sensibilité (favorise la spécificité à sensibilité fixée)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    best = 0.0
    for thr in np.unique(np.concatenate([[0.0], y_score, [1.0]])):
        m = binary_metrics(y_true, y_score, float(thr))
        if not np.isnan(m["sensitivity"]) and m["sensitivity"] >= target:
            best = max(best, float(thr))
    return best
