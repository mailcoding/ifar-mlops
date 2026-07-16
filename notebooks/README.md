# Notebooks (Colab GPU)

Miroir des scripts `src/mlops/train/` pour l'exécution sur GPU gratuit (Google Colab).

- Importer ici les **notebooks d'entraînement existants** (qui ont produit
  `cbis_ddsm_efficientnet_final.pth`, `yolov8_*.pt`), puis les factoriser dans `src/mlops/`.
- Un notebook = expérience reproductible : monter le dataset (repo HF privé), lancer l'entraînement,
  évaluer, **exporter au format ml-service** (`from mlops.export import export_classifier`), publier
  l'artefact (`from mlops.registry import publish_artifact`).
- ⚠️ Ne jamais committer de sorties contenant des données patient. Vider les sorties avant commit.
