# src/mlops/eval/classifier_eval.py
# ─────────────────────────────────────────────
# Évalue des poids de classifieur sur un manifeste donné — pour obtenir des métriques
# COMPARABLES entre deux modèles (candidat vs modèle en ligne).
#
# ⚠️ Pourquoi ce module existe : comparer l'AUC d'un modèle mesurée sur un split à celle d'un
#    autre modèle mesurée sur un AUTRE split ne veut RIEN dire. Toute affirmation « le nouveau
#    modèle est meilleur » exige les deux modèles sur le MÊME jeu de validation, avec le MÊME
#    prétraitement (celui de l'inférence produit).
#
# torch/timm ne sont requis que pour l'exécution (extra [train]) → imports paresseux.
# ─────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path


def load_classifier_weights(model, weights_path: str | Path, device=None):
    """Charge des poids dans le modèle, en acceptant les deux formats livrés par les notebooks.

    Identique à ml-service/app/classification.py : `state_dict` brut OU checkpoint
    `{"model_state_dict": ..., ...}`. Retourne le modèle."""
    import torch

    checkpoint = torch.load(str(weights_path), map_location=device or "cpu")
    state_dict = (checkpoint["model_state_dict"]
                  if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                  else checkpoint)
    model.load_state_dict(state_dict)
    return model


def evaluate_classifier(weights_path: str | Path, manifest_csv: str | Path, *,
                        threshold: float = 0.50, batch_size: int = 32,
                        sensitivity_target: float = 0.90, device=None) -> dict:
    """Évalue des poids sur un manifeste `path,label` et retourne les métriques cliniques.

    Le prétraitement est celui de l'INFÉRENCE (`classifier_transforms(train=False)`), donc
    exactement celui du produit — condition d'une comparaison honnête."""
    import torch
    from torch.utils.data import DataLoader

    from mlops.datasets import RoiClassificationDataset
    from mlops.eval.metrics import binary_metrics, threshold_for_sensitivity
    from mlops.models.efficientnet import NUM_CLASSES, EfficientNetClassifier

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EfficientNetClassifier(num_classes=NUM_CLASSES)
    load_classifier_weights(model, weights_path, device=device)
    model.to(device).eval()

    ds = RoiClassificationDataset(manifest_csv, train=False)          # transforms d'inférence
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    y_true, y_score = [], []
    with torch.no_grad():
        for images, labels in dl:
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            y_true.extend(labels.tolist())
            y_score.extend(probs.cpu().tolist())

    m = binary_metrics(y_true, y_score, threshold=threshold)
    op_thr = threshold_for_sensitivity(y_true, y_score, target=sensitivity_target)
    op = binary_metrics(y_true, y_score, threshold=op_thr)
    return {
        **m,
        "operating_threshold_sens": round(op_thr, 4),
        "specificity_at_target_sens": op["specificity"],
        "sensitivity_target": sensitivity_target,
        "weights": str(weights_path),
        "manifest_csv": str(manifest_csv),
    }
