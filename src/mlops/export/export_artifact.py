# src/mlops/export/export_artifact.py
# ─────────────────────────────────────────────
# Exporte un modèle entraîné AU FORMAT ATTENDU par le ml-service, avec un
# manifeste de version. Le ml-service charge les poids via `resolve_weights`
# (ifar/ml-service/app/utils.py) — les NOMS DE FICHIERS comptent :
#   - classification mammo : models/efficientnet_b0.pth   (state_dict)
#   - détection/seg mammo  : models/yolov8_seg.pt          (poids Ultralytics)
#   - histologie           : models/histology.pth         (à définir avec le wrapper)
# ─────────────────────────────────────────────

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# Noms de fichiers imposés par le ml-service (utils.resolve_weights + *_WEIGHT_CANDIDATES).
ARTIFACT_FILENAMES = {
    "mammo_classifier": "efficientnet_b0.pth",
    "mammo_detector": "yolov8_seg.pt",
    "histology": "histology.pth",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: str | Path, *, model: str, version: str, framework: str,
                   input_spec: dict, trained_on: dict, metrics: dict,
                   threshold: float, git_commit: str = "", validated_by: str = "",
                   weights_filename: str = "", notes: str = "") -> Path:
    """Écrit `manifest.json` (métadonnées de version, cf. MLOPS_ARCHITECTURE.md §4)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / weights_filename if weights_filename else None
    manifest = {
        "model": model,
        "version": version,
        "framework": framework,
        "input": input_spec,
        "trained_on": trained_on,
        "metrics": metrics,
        "threshold": threshold,
        "weights_file": weights_filename,
        "weights_sha256": sha256_file(weights_path) if weights_path and weights_path.exists() else "",
        "git_commit": git_commit,
        "validated": bool(validated_by),
        "validated_by": validated_by,
        "date": _now(),
        "notes": notes,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


def export_classifier(model, out_dir: str | Path, *, version: str, metrics: dict,
                      trained_on: dict, threshold: float = 0.50,
                      git_commit: str = "", validated_by: str = "",
                      as_checkpoint: bool = False) -> dict:
    """
    Sérialise le classifieur mammo au format ml-service + manifeste + model card stub.

    - `as_checkpoint=False` → torch.save(model.state_dict(), efficientnet_b0.pth) (format state_dict).
    - `as_checkpoint=True`  → torch.save({"model_state_dict": ..., "best_val_acc": ...}, ...).
    Les deux sont acceptés par ml-service/app/classification.py.
    """
    import torch  # import tardif : pas requis pour manifeste/tests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = ARTIFACT_FILENAMES["mammo_classifier"]
    weights_path = out_dir / fname

    if as_checkpoint:
        torch.save({"model_state_dict": model.state_dict(),
                    "best_val_acc": metrics.get("accuracy")}, weights_path)
    else:
        torch.save(model.state_dict(), weights_path)

    manifest = write_manifest(
        out_dir, model="ifar-mammo-classifier", version=version,
        framework="timm/efficientnet_b0",
        input_spec={"size": 224, "norm": "imagenet", "channels": 3},
        trained_on=trained_on, metrics=metrics, threshold=threshold,
        git_commit=git_commit, validated_by=validated_by, weights_filename=fname,
    )

    card = out_dir / "MODEL_CARD.md"
    if not card.exists():
        card.write_text(
            f"# ifar-mammo-classifier {version}\n\n"
            f"EfficientNetB0 (bénin/malin). Seuil {threshold}. AUC "
            f"{metrics.get('auc', 'N/A')} / sensibilité {metrics.get('sensitivity', 'N/A')}.\n\n"
            "Compléter : données, limites, sous-groupes, usage prévu (SaMD). "
            "Voir MODEL_CARD/TEMPLATE.md.\n"
        )
    return {"weights": str(weights_path), "manifest": str(manifest), "model_card": str(card)}
