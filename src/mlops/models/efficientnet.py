# src/mlops/models/efficientnet.py
# ─────────────────────────────────────────────
# Architecture EfficientNetB0 — DOIT rester IDENTIQUE à celle du produit
# (ifar/ml-service/app/classification.py::EfficientNetClassifier), sinon les
# poids exportés ne se chargeront pas dans le ml-service.
#
# ⚠️ Contrat figé : timm efficientnet_b0 (num_classes=0) + tête
#    Dropout(0.3) → Linear(1280, 256) → ReLU → Dropout(0.2) → Linear(256, 2).
#    Sortie : logits (N, 2) → softmax → [P(bénin), P(malin)]. Seuil 0.50.
# ─────────────────────────────────────────────

import torch
import torch.nn as nn

NUM_CLASSES = 2  # 0 = BENIGN, 1 = MALIGNANT
MALIGNANCY_THRESHOLD = 0.50


class EfficientNetClassifier(nn.Module):
    """EfficientNetB0 (backbone timm) + tête de classification bénin/malin."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("timm requis : pip install timm") from exc

        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0
        )
        in_features = self.backbone.num_features  # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def build_pretrained_backbone(num_classes: int = NUM_CLASSES) -> EfficientNetClassifier:
    """Variante pour l'entraînement : backbone pré-entraîné ImageNet (transfer learning)."""
    model = EfficientNetClassifier(num_classes=num_classes)
    import timm
    # Recharge un backbone pré-entraîné (l'inférence produit, elle, part de num_classes=0 sans pré-entraînement).
    model.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    return model
