# src/mlops/train/train_mammo_detector.py
# ─────────────────────────────────────────────
# Entraînement du détecteur/segmenteur de lésions (YOLOv8-seg, Ultralytics).
# Produit `yolov8_seg.pt` (segmentation → masques + mesures) attendu par le ml-service,
# ÉVALUE (mAP global + par classe, seuil orienté recall) et écrit un `manifest.json`.
#
#   python -m mlops.train.train_mammo_detector --config configs/mammo_detector.yaml
#
# ⚠️ Nécessite un GPU + un dataset YOLO-seg (images + labels polygone + data.yaml).
#    Le data.yaml est généré depuis CBIS (masques ROI → polygones), split par patient.
# ─────────────────────────────────────────────

import argparse
import shutil
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main(config_path: str) -> None:
    from ultralytics import YOLO

    from mlops.eval.detector_metrics import evaluate_detector, summarize
    from mlops.export import write_manifest

    cfg = load_config(config_path)
    tcfg = cfg["train"]
    ecfg = cfg.get("eval", {})
    aug = tcfg.get("augment", {}) or {}
    out_dir = Path(cfg["export"]["out_dir"])
    version = cfg["export"]["version"]
    imgsz = tcfg.get("imgsz", 1280)

    # base : yolov8s-seg.pt (segmentation) — cohérent avec le contrat masques du produit.
    model = YOLO(cfg["model"].get("base", "yolov8s-seg.pt"))
    results = model.train(
        data=cfg["data"]["yaml"],          # data.yaml Ultralytics (train/val + classes)
        imgsz=imgsz,
        epochs=tcfg.get("epochs", 120),
        batch=tcfg.get("batch", 8),
        patience=tcfg.get("patience", 25),
        cos_lr=tcfg.get("cos_lr", True),
        device=tcfg.get("device"),
        project=str(out_dir),
        name=version,
        **dict(aug),                       # augmentations médicales (fliplr/flipud/degrees/…)
    )

    # Copie le meilleur poids au NOM attendu par le ml-service.
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = out_dir / "yolov8_seg.pt"
    if not best.exists():
        print(f"[warn] best.pt introuvable ({best}) — vérifier l'entraînement.")
        return
    shutil.copy(best, dest)
    print(f"Poids exporté : {dest}")

    # Évaluation : mAP global + PAR CLASSE + seuil de confiance orienté recall.
    metrics = evaluate_detector(
        dest, cfg["data"]["yaml"],
        imgsz=ecfg.get("imgsz", imgsz),
        conf=ecfg.get("conf", 0.001),
        iou=ecfg.get("iou", 0.6),
        device=tcfg.get("device"),
        target_recall=ecfg.get("target_recall", 0.90),
    )
    print(summarize(metrics))

    # Manifeste versionné (métriques + seuil recommandé) au format ml-service.
    thr = (metrics.get("recommended_conf_for_recall") or {}).get("conf", 0.0)
    write_manifest(
        out_dir, model="ifar-mammo-detector", version=version,
        framework="ultralytics/yolov8-seg",
        input_spec={"size": ecfg.get("imgsz", imgsz), "classes": ["mass", "calcification"]},
        trained_on={"dataset": cfg["data"].get("name", "cbis-ddsm"), "data_yaml": cfg["data"]["yaml"]},
        metrics=metrics, threshold=thr, weights_filename="yolov8_seg.pt",
    )
    print(f"Manifeste écrit : {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
