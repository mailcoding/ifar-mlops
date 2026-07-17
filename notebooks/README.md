# Notebooks (Colab GPU)

Miroir des scripts `src/mlops/train/` pour l'exécution sur GPU gratuit (Google Colab).

## Disponible
- **`train_mammo_classifier.ipynb`** — pipeline de bout en bout du classifieur bénin/malin :
  dataset **HF privé** (`snapshot_download`) → manifestes (split par patient stratifié) → entraînement
  des deux variantes (`cropped` **et** `full`) → comparaison des métriques → publication de l'artefact
  versionné (variante `cropped`, alignée sur l'inférence produit). Cibles : AUC ≥ 0,85 / sensibilité ≥ 0,90.
- **`train_mammo_classifier_drive.ipynb`** — **même pipeline**, mais le dataset est lu depuis **Google
  Drive monté** (aucun repo HF dataset requis ; le token HF `write` ne sert qu'à la publication finale).
  Adapter `DRIVE_DATA_DIR` + les chemins des CSV à ton arborescence Drive.

- Importer ici les **notebooks d'entraînement existants** (qui ont produit
  `cbis_ddsm_efficientnet_final.pth`, `yolov8_*.pt`), puis les factoriser dans `src/mlops/`.
- Un notebook = expérience reproductible : monter le dataset (repo HF privé), lancer l'entraînement,
  évaluer, **exporter au format ml-service** (`from mlops.export import export_classifier`), publier
  l'artefact (`from mlops.registry import publish_artifact`).
- ⚠️ Ne jamais committer de sorties contenant des données patient. Vider les sorties avant commit.

## Installer le package dans Colab
Les deux notebooks **clonent automatiquement `ifar-mlops`** dans `/content/ifar-mlops` puis font
`pip install "/content/ifar-mlops[train]"`. Si le dépôt est **privé**, mets un **PAT** dans l'URL de
clone (commentée dans la cellule d'installation) :
`!git clone https://<PAT>@github.com/mailcoding/ifar-mlops.git /content/ifar-mlops`.
