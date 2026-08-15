# src/mlops/datasets/cbis_ddsm.py
# ─────────────────────────────────────────────
# Dataset de ROIs mammographiques (bénin/malin) — ex. CBIS-DDSM (public).
# Le prétraitement DOIT matcher l'inférence produit (utils.preprocess_for_classifier) :
# ROI 224×224, normalisation ImageNet.
#
# ⚠️ Gouvernance : aucune donnée brute / PHI dans git. Les données vivent dans un
#    repo dataset HF PRIVÉ (pseudonymisé) ou un stockage objet ; ici on ne lit
#    qu'un manifeste CSV local (chemins + labels) non versionné.
# ─────────────────────────────────────────────

import csv
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLF_INPUT_SIZE = 224

LABELS = {"BENIGN": 0, "MALIGNANT": 1}


# Augmentation d'ENTRAÎNEMENT (jamais appliquée en évaluation/inférence).
#   • random_resized_crop : [min, max] d'échelle — JITTER DE CADRAGE. À l'inférence, le produit
#     découpe la ROI depuis une bbox YOLO avec 10 % de marge, dont le cadrage flotte ; les ROI CBIS,
#     elles, sont serrées. Ce jitter rend le modèle robuste à cet écart. None → simple Resize.
#   • rotation (degrés) / translate (fraction) / brightness / contrast / hflip / vflip (probas).
DEFAULT_AUGMENT = {
    "random_resized_crop": [0.75, 1.0],
    "hflip": 0.5,
    "vflip": 0.2,
    "rotation": 15,
    "translate": 0.05,
    "brightness": 0.2,
    "contrast": 0.2,
}


def resolve_augment(cfg: dict | None = None) -> dict:
    """Fusionne une config d'augmentation avec les défauts (pur : ni torch ni torchvision).

    Lève ValueError sur une clé inconnue — une faute de frappe désactiverait silencieusement
    une augmentation que l'on croit active."""
    out = dict(DEFAULT_AUGMENT)
    if cfg:
        unknown = set(cfg) - set(DEFAULT_AUGMENT)
        if unknown:
            raise ValueError(
                f"Clés d'augmentation inconnues : {sorted(unknown)}. "
                f"Attendu parmi : {sorted(DEFAULT_AUGMENT)}"
            )
        out.update({k: v for k, v in cfg.items() if v is not None})
    return out


def _inference_ops(transforms):
    """CONTRAT D'INFÉRENCE — doit rester IDENTIQUE à
    ifar/ml-service/app/utils.py::preprocess_for_classifier (224 px, normalisation ImageNet).
    Toute divergence recrée le décalage entraînement ↔ inférence qui a rendu le modèle
    historique inutilisable. Ne rien ajouter ici sans changer le produit en même temps."""
    return [
        transforms.Resize((CLF_INPUT_SIZE, CLF_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def classifier_transforms(train: bool = False, augment: dict | None = None):
    """Pipeline de prétraitement. `train=False` = strictement celui de l'inférence produit."""
    from torchvision import transforms

    if not train:
        return transforms.Compose(_inference_ops(transforms))

    a = resolve_augment(augment)
    ops = []
    rrc = a.get("random_resized_crop")
    if rrc:
        # ratio proche de 1 : on ne veut pas déformer la lésion (le défaut torchvision est 3/4–4/3).
        ops.append(transforms.RandomResizedCrop(CLF_INPUT_SIZE, scale=tuple(rrc), ratio=(0.9, 1.1)))
    else:
        ops.append(transforms.Resize((CLF_INPUT_SIZE, CLF_INPUT_SIZE)))
    if a["hflip"]:
        ops.append(transforms.RandomHorizontalFlip(p=a["hflip"]))
    if a["vflip"]:
        ops.append(transforms.RandomVerticalFlip(p=a["vflip"]))
    if a["rotation"] or a["translate"]:
        ops.append(transforms.RandomAffine(
            degrees=a["rotation"], translate=(a["translate"], a["translate"])))
    if a["brightness"] or a["contrast"]:
        ops.append(transforms.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class RoiClassificationDataset:
    """
    Dataset PyTorch de ROIs. Attend un CSV `manifest` : colonnes `path,label`
    (label ∈ {BENIGN, MALIGNANT}). Les images sont lues en RGB.

    Exemple de manifeste (NON versionné, cf. data/README.md) :
        path,label
        /data/cbis/roi_0001.png,MALIGNANT
    """

    def __init__(self, manifest_csv: str | Path, train: bool = False, augment: dict | None = None):
        from torch.utils.data import Dataset  # noqa: F401 (marqueur d'API)

        self.items = []
        with open(manifest_csv, newline="") as f:
            for row in csv.DictReader(f):
                label = row["label"].strip().upper()
                if label not in LABELS:
                    continue
                self.items.append((row["path"].strip(), LABELS[label]))
        self.transform = classifier_transforms(train=train, augment=augment)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from PIL import Image

        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
