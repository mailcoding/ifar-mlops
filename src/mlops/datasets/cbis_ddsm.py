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


def classifier_transforms(train: bool = False):
    """Transforms identiques à l'inférence (+ augmentation légère en entraînement)."""
    from torchvision import transforms

    base = [transforms.Resize((CLF_INPUT_SIZE, CLF_INPUT_SIZE))]
    if train:
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(base)


class RoiClassificationDataset:
    """
    Dataset PyTorch de ROIs. Attend un CSV `manifest` : colonnes `path,label`
    (label ∈ {BENIGN, MALIGNANT}). Les images sont lues en RGB.

    Exemple de manifeste (NON versionné, cf. data/README.md) :
        path,label
        /data/cbis/roi_0001.png,MALIGNANT
    """

    def __init__(self, manifest_csv: str | Path, train: bool = False):
        from torch.utils.data import Dataset  # noqa: F401 (marqueur d'API)

        self.items = []
        with open(manifest_csv, newline="") as f:
            for row in csv.DictReader(f):
                label = row["label"].strip().upper()
                if label not in LABELS:
                    continue
                self.items.append((row["path"].strip(), LABELS[label]))
        self.transform = classifier_transforms(train=train)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from PIL import Image

        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
