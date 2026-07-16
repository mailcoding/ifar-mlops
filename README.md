# ifar-mlops — entraînement & versioning des modèles IA (dépôt séparé)

Squelette du **dépôt MLOps** d'IFAR CancerSein AI. **Séparé du produit** : il entraîne, évalue et
**versionne** les modèles ; le produit (`ml-service`) ne consomme qu'un **artefact versionné**.

> Conception détaillée : `ifar/docs/MLOPS_ARCHITECTURE.md` (côté produit). Ce dossier `mlops/` est
> destiné à être **extrait vers un dépôt GitHub dédié** (`ifar-mlops`) — voir « Extraction » plus bas.

## Pourquoi séparé
- Dépendances lourdes (PyTorch/CUDA, Ultralytics, datasets) hors du produit.
- Poids & données hors du git produit ; gouvernance PHI distincte (finalité `research`).
- Le modèle évolue par **versions validées** ; le produit consomme une version figée.

## Contrat modèle (À RESPECTER — source de vérité : `ifar/ml-service/app/`)
| Modèle | Framework | Entrée | Fichier attendu |
|---|---|---|---|
| Détection/seg mammo | YOLOv8(-seg) Ultralytics `.pt` | 640×640 | `models/yolov8_seg.pt` |
| Classif mammo bénin/malin | EfficientNetB0 (timm) + tête custom | 224×224, norm ImageNet | `models/efficientnet_b0.pth` |
| Histologie (Nottingham) | à créer (placeholder aujourd'hui) | — | `models/histology.pth` |

L'architecture `EfficientNetClassifier` (`src/mlops/models/efficientnet.py`) est **identique** à celle
du produit — indispensable pour que les poids se chargent.

## Structure
```
src/mlops/
  models/     architectures (EfficientNetClassifier identique au produit)
  datasets/   loaders (ROIs mammo CBIS-DDSM ; histo à venir)
  train/      entraînement (mammo_classifier, mammo_detector, histology)
  eval/       métriques cliniques (AUC, sensibilité/spécificité, calibration)
  export/     export au format ml-service + manifest.json + model card
  registry/   publication/pull via Hugging Face Hub (repos model privés)
configs/      hyperparamètres & chemins (yaml)
data/         pointeurs uniquement — JAMAIS de données brutes/PHI
MODEL_CARD/   gabarit de model card (SaMD)
```

## Démarrage
```bash
pip install -e ".[train]"          # torch, timm, ultralytics, huggingface_hub, …

# Classifieur mammo (nécessite des manifestes CSV path,label — voir data/README.md)
python -m mlops.train.train_mammo_classifier --config configs/mammo_classifier.yaml

# Publier l'artefact versionné sur HF Hub (repo model privé)
HF_TOKEN=hf_xxx python -c "from mlops.registry import publish_artifact; \
  publish_artifact('artifacts/mammo-clf', 'Mailcoding/ifar-mammo-classifier', 'v0.1.0')"
```

## Cycle de vie
`datasets pseudonymisés → entraînement → évaluation → 🚦 porte de validation clinique
(SaMD / IEC 62304 / ISO 14971) → publication artefact HF (tag vX.Y.Z) → mise à jour du Space
ml-service → suivi via le dashboard ops (/metrics)`. Détails : `GOVERNANCE.md`, `MODEL_CARD/TEMPLATE.md`.

## Extraction vers un dépôt dédié
Ce squelette vit temporairement dans le monorepo produit. **Runbook complet : [`INFRA_SETUP.md`](INFRA_SETUP.md)**
(création du repo GitHub, extraction avec historique, repos HF privés, CI, 1ʳᵉ publication). En bref :
```bash
# Historique préservé (recommandé) — voir INFRA_SETUP.md pour les garde-fous et la suite
bash scripts/extract_to_repo.sh git@github.com:mailcoding/ifar-mlops.git
# Repos HF privés (model + dataset), idempotent :
HF_TOKEN=hf_xxx python scripts/bootstrap_hf.py
```

## Gouvernance
Aucune donnée patient dans ce dépôt. Voir `GOVERNANCE.md` (pseudonymisation, finalité research V6,
résidence, DPIA, rétention).
