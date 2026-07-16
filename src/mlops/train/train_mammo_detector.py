# src/mlops/train/train_mammo_detector.py
# ─────────────────────────────────────────────
# Entraînement du détecteur/segmenteur de lésions (YOLOv8-seg, Ultralytics).
# Produit `yolov8_seg.pt` (segmentation → masques + mesures) attendu par le
# ml-service. En détection seule (sans masques), le service passe en mode
# dégradé (seg_supports_masks:false).
#
#   python -m mlops.train.train_mammo_detector --config configs/mammo_detector.yaml
#
# ⚠️ Squelette. Prépare un dataset YOLO (images + labels .txt) et un data.yaml.
# ─────────────────────────────────────────────

import argparse
import shutil
from pathlib import Path

import yaml


def main(config_path: str) -> None:
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(config_path).read_text())
    # base : yolov8n-seg.pt (segmentation) — cohérent avec le contrat masques du produit.
    model = YOLO(cfg["model"].get("base", "yolov8n-seg.pt"))
    results = model.train(
        data=cfg["data"]["yaml"],          # data.yaml Ultralytics (train/val + classes)
        imgsz=cfg["train"].get("imgsz", 640),
        epochs=cfg["train"].get("epochs", 100),
        batch=cfg["train"].get("batch", 16),
        project=cfg["export"]["out_dir"],
        name=cfg["export"]["version"],
    )
    # Copie le meilleur poids au NOM attendu par le ml-service.
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = Path(cfg["export"]["out_dir"]) / "yolov8_seg.pt"
    if best.exists():
        shutil.copy(best, dest)
        print(f"Poids exporté : {dest}")
    else:
        print(f"[warn] best.pt introuvable ({best}) — vérifier l'entraînement.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
