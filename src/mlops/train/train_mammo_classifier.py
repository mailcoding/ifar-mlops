# src/mlops/train/train_mammo_classifier.py
# ─────────────────────────────────────────────
# Entraînement du classifieur mammo bénin/malin (EfficientNetB0), puis export
# AU FORMAT ml-service (efficientnet_b0.pth + manifest.json).
#
#   python -m mlops.train.train_mammo_classifier --config configs/mammo_classifier.yaml
#
# ⚠️ Squelette : renseigner les chemins de données (manifestes CSV) dans la config.
#    Données pseudonymisées uniquement (voir data/README.md, GOVERNANCE.md).
# ─────────────────────────────────────────────

import argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main(config_path: str) -> None:
    import torch
    from torch.utils.data import DataLoader

    from mlops.datasets import RoiClassificationDataset
    from mlops.models.efficientnet import build_pretrained_backbone, NUM_CLASSES
    from mlops.eval.metrics import binary_metrics
    from mlops.export import export_classifier

    cfg = load_config(config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    train_ds = RoiClassificationDataset(cfg["data"]["train_manifest"], train=True)
    val_ds = RoiClassificationDataset(cfg["data"]["val_manifest"], train=False)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"].get("num_workers", 2))
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = build_pretrained_backbone(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 1e-4))
    # Pondération de classe possible (déséquilibre bénin/malin) via cfg["train"]["class_weights"].
    weights = cfg["train"].get("class_weights")
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float, device=device) if weights else None
    )

    best_auc = -1.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # Évaluation
        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for images, labels in val_dl:
                probs = torch.softmax(model(images.to(device)), dim=1)[:, 1]
                y_true.extend(labels.tolist())
                y_score.extend(probs.cpu().tolist())
        m = binary_metrics(y_true, y_score, threshold=cfg["eval"].get("threshold", 0.50))
        print(f"epoch {epoch+1}/{cfg['train']['epochs']} — AUC {m['auc']} "
              f"sens {m['sensitivity']} spec {m['specificity']}")

        if m["auc"] and m["auc"] > best_auc:
            best_auc = m["auc"]
            out = export_classifier(
                model, cfg["export"]["out_dir"], version=cfg["export"]["version"],
                metrics=m, trained_on={"dataset": cfg["data"].get("name", "cbis-ddsm"),
                                       "n_train": len(train_ds), "n_val": len(val_ds)},
                threshold=cfg["eval"].get("threshold", 0.50),
            )
            print(f"  ↳ meilleur modèle exporté : {out['weights']}")

    print(f"Terminé. Meilleure AUC : {best_auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
