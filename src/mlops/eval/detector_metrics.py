# src/mlops/eval/detector_metrics.py
# ─────────────────────────────────────────────
# Évaluation du détecteur/segmenteur de lésions (YOLOv8, Ultralytics).
# Le ml-service est plafonné par la détection : une lésion non détectée n'est jamais
# classée → on rapporte le mAP GLOBAL *et PAR CLASSE* (masse vs calcification, souvent
# bien pire), et on choisit un seuil de confiance orienté RECALL (ne pas rater de lésion).
#
# Ultralytics n'est requis que pour `evaluate_detector` (extra [train]) → import paresseux.
# Les utilitaires de courbe/seuil ne dépendent que de numpy (cœur) → testables sans GPU.
# ─────────────────────────────────────────────

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def threshold_for_target_recall(
    conf_thresholds: Sequence[float], recalls: Sequence[float], target: float = 0.90
) -> float:
    """Plus haut seuil de confiance atteignant au moins `target` de recall.

    En détection médicale, le RECALL prime (ne pas rater de lésion) ; à recall fixé, un
    seuil plus élevé limite les faux positifs. Renvoie 0.0 si la cible n'est jamais atteinte."""
    import numpy as np

    conf = np.asarray(conf_thresholds, dtype=float)
    rec = np.asarray(recalls, dtype=float)
    ok = rec >= target
    if not ok.any():
        return 0.0
    return float(conf[ok].max())


def recall_confidence_curve(metrics) -> tuple[list[float], list[float]] | None:
    """Extrait (confidences, recalls) de la courbe Recall–Confidence d'un résultat Ultralytics.

    `metrics.curves_results` = liste de (px, py, xlabel, ylabel) ; py peut être
    (n_classes, n_points) → moyenné sur les classes. Renvoie None si indisponible
    (structure dépendante de la version d'Ultralytics)."""
    import numpy as np

    for entry in getattr(metrics, "curves_results", None) or []:
        try:
            px, py, xlabel, ylabel = entry
        except (ValueError, TypeError):
            continue
        if "confidence" in str(xlabel).lower() and "recall" in str(ylabel).lower():
            arr = np.asarray(py, dtype=float)
            rec = arr.mean(axis=0) if arr.ndim > 1 else arr
            return [float(x) for x in px], [float(r) for r in rec]
    return None


def _class_result(box, i):
    """Renvoie (precision, recall, ap50, ap) pour la i-ème classe évaluée, ou None si indisponible."""
    try:
        return box.class_result(i)
    except Exception:  # noqa: BLE001 — API Ultralytics variable selon la version → dégradation douce
        return None


def evaluate_detector(
    weights: str | Path,
    data_yaml: str | Path,
    *,
    imgsz: int = 1280,
    conf: float = 0.001,
    iou: float = 0.6,
    device=None,
    split: str = "val",
    target_recall: float = 0.90,
) -> dict:
    """Évalue un détecteur/segmenteur YOLO ; renvoie des métriques structurées.

    Rapporte mAP50 / mAP50-95 / précision / recall GLOBAUX et PAR CLASSE, le bloc `seg`
    (masques) si le modèle segmente, et le seuil de confiance recommandé pour `target_recall`.
    `conf` bas → courbe PR complète (le seuil de production se règle ensuite). Requiert ultralytics."""
    from ultralytics import YOLO

    res = YOLO(str(weights)).val(
        data=str(data_yaml), imgsz=imgsz, conf=conf, iou=iou,
        device=device, split=split, verbose=False,
    )
    names = getattr(res, "names", {}) or {}
    box = res.box

    def _r(v):
        return round(float(v), 4) if v is not None else None

    out: dict = {
        "weights": str(weights),
        "imgsz": imgsz,
        "split": split,
        "box": {
            "map50": _r(getattr(box, "map50", None)),
            "map50_95": _r(getattr(box, "map", None)),
            "precision": _r(getattr(box, "mp", None)),
            "recall": _r(getattr(box, "mr", None)),
        },
        "per_class": {},
    }

    seg = getattr(res, "seg", None)
    if seg is not None:
        out["seg"] = {"map50": _r(seg.map50), "map50_95": _r(seg.map)}

    for i, c in enumerate(getattr(box, "ap_class_index", []) or []):
        result = _class_result(box, i)
        if result is None:  # classe non évaluée (aucune instance) → on saute
            continue
        p, r, ap50, ap = result
        out["per_class"][names.get(int(c), str(int(c)))] = {
            "precision": _r(p), "recall": _r(r), "map50": _r(ap50), "map50_95": _r(ap),
        }

    curve = recall_confidence_curve(res)
    if curve is not None:
        out["recommended_conf_for_recall"] = {
            "target_recall": target_recall,
            "conf": round(threshold_for_target_recall(curve[0], curve[1], target=target_recall), 4),
        }
    return out


def summarize(metrics: dict) -> str:
    """Rendu texte compact des métriques (pour le log d'entraînement)."""
    b = metrics.get("box", {})
    lines = [
        (f"box: mAP50={b.get('map50')} mAP50-95={b.get('map50_95')} "
         f"P={b.get('precision')} R={b.get('recall')}"),
    ]
    if "seg" in metrics:
        s = metrics["seg"]
        lines.append(f"seg: mAP50={s.get('map50')} mAP50-95={s.get('map50_95')}")
    for name, m in metrics.get("per_class", {}).items():
        lines.append(f"  [{name}] mAP50={m['map50']} P={m['precision']} R={m['recall']}")
    rc = metrics.get("recommended_conf_for_recall")
    if rc:
        lines.append(f"conf recommandé @recall≥{rc['target_recall']}: {rc['conf']}")
    return "\n".join(lines)
