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


def set_backbone_trainable(model, trainable: bool) -> int:
    """Gèle (False) ou dégèle (True) le backbone. Retourne le nombre de tenseurs modifiés.

    Fine-tuning progressif : entraîner d'emblée TOUT le réseau au LR de la tête dégrade les
    features ImageNet sur un petit dataset. On entraîne d'abord la tête seule (backbone gelé),
    puis on dégèle avec un LR réduit. Volontairement sans import torch (duck-typing) → testable."""
    n = 0
    for p in model.backbone.parameters():
        p.requires_grad = trainable
        n += 1
    return n


def main(config_path: str) -> None:
    import torch
    from torch.utils.data import DataLoader

    from mlops.datasets import RoiClassificationDataset
    from mlops.eval.metrics import binary_metrics, threshold_for_sensitivity
    from mlops.export import export_classifier
    from mlops.models.efficientnet import NUM_CLASSES, build_pretrained_backbone

    cfg = load_config(config_path)
    tcfg, ecfg = cfg["train"], cfg["eval"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # `augment` : augmentation d'ENTRAÎNEMENT uniquement — la val garde le pipeline d'inférence.
    train_ds = RoiClassificationDataset(cfg["data"]["train_manifest"], train=True,
                                        augment=tcfg.get("augment"))
    val_ds = RoiClassificationDataset(cfg["data"]["val_manifest"], train=False)
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=tcfg.get("num_workers", 2))
    val_dl = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    epochs = tcfg["epochs"]
    weight_decay = tcfg.get("weight_decay", 1e-4)
    use_cosine = tcfg.get("scheduler", "cosine") == "cosine"
    model = build_pretrained_backbone(num_classes=NUM_CLASSES).to(device)

    # Fine-tuning progressif : tête seule pendant `freeze_epochs`, puis dégel à `unfreeze_lr`.
    fcfg = tcfg.get("finetune") or {}
    freeze_epochs = int(fcfg.get("freeze_epochs", 0) or 0)
    unfreeze_lr = float(fcfg.get("unfreeze_lr", tcfg["lr"]))
    if freeze_epochs > 0:
        n = set_backbone_trainable(model, False)
        print(f"Backbone GELÉ ({n} tenseurs) pour {freeze_epochs} epoch(s) — tête seule à lr={tcfg['lr']}.")

    def make_optimizer(lr):
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def make_scheduler(opt, remaining):
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, remaining)) if use_cosine else None

    optimizer = make_optimizer(tcfg["lr"])
    scheduler = make_scheduler(optimizer, epochs)

    # Pondération de classe possible (déséquilibre bénin/malin) via cfg["train"]["class_weights"].
    weights = tcfg.get("class_weights")
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float, device=device) if weights else None,
        label_smoothing=float(tcfg.get("label_smoothing", 0.0)),
    )

    threshold = ecfg.get("threshold", 0.50)
    sens_target = ecfg.get("sensitivity_target", 0.90)
    patience = tcfg.get("patience", 0)
    best_auc, epochs_no_improve = -1.0, 0

    for epoch in range(epochs):
        # Dégel du backbone : nouvel optimiseur (les params gelés n'y étaient pas) à LR réduit.
        if freeze_epochs and epoch == freeze_epochs:
            n = set_backbone_trainable(model, True)
            optimizer = make_optimizer(unfreeze_lr)
            scheduler = make_scheduler(optimizer, epochs - epoch)
            print(f"→ Backbone DÉGELÉ ({n} tenseurs) à l'epoch {epoch + 1}, lr={unfreeze_lr}.")

        model.train()
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Évaluation
        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for images, labels in val_dl:
                probs = torch.softmax(model(images.to(device)), dim=1)[:, 1]
                y_true.extend(labels.tolist())
                y_score.extend(probs.cpu().tolist())
        m = binary_metrics(y_true, y_score, threshold=threshold)
        # Point de fonctionnement clinique : seuil atteignant la sensibilité cible + spécificité obtenue.
        op_thr = threshold_for_sensitivity(y_true, y_score, target=sens_target)
        op = binary_metrics(y_true, y_score, threshold=op_thr)
        print(f"epoch {epoch+1}/{epochs} — AUC {m['auc']} sens {m['sensitivity']} spec {m['specificity']} "
              f"| @sens≥{sens_target}: thr={round(op_thr, 3)} spec {op['specificity']}")

        if m["auc"] and m["auc"] > best_auc:
            best_auc, epochs_no_improve = m["auc"], 0
            m_export = {**m, "operating_threshold_sens": round(op_thr, 4),
                        "specificity_at_target_sens": op["specificity"], "sensitivity_target": sens_target}
            out = export_classifier(
                model, cfg["export"]["out_dir"], version=cfg["export"]["version"],
                metrics=m_export, trained_on={"dataset": cfg["data"].get("name", "cbis-ddsm"),
                                              "n_train": len(train_ds), "n_val": len(val_ds)},
                threshold=threshold,
            )
            print(f"  ↳ meilleur modèle exporté : {out['weights']}")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"  ↳ early stopping (aucun gain d'AUC depuis {patience} epochs).")
                break

    print(f"Terminé. Meilleure AUC : {best_auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
