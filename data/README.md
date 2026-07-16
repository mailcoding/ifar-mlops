# Données d'entraînement — pointeurs uniquement

⚠️ **Aucune donnée brute ni PHI ici.** Ce dossier ne contient que des **pointeurs** (manifestes,
hachages, identifiants de repos dataset HF). `.gitignore` bloque tout le reste.

## Où vivent les données
- **Repo dataset Hugging Face PRIVÉ**, pseudonymisé (ex. `Mailcoding/ifar-mammo-rois`,
  `Mailcoding/ifar-histo`), ou stockage objet (R2/S3) à accès restreint.
- Téléchargement local hors git (ex. `~/data/ifar/…`) pour l'entraînement.

## Format attendu
### Classifieur mammo (ROIs)
Manifeste CSV `path,label` (label ∈ `BENIGN`/`MALIGNANT`) — consommé par
`RoiClassificationDataset` :
```
path,label
/data/cbis/roi_0001.png,MALIGNANT
/data/cbis/roi_0002.png,BENIGN
```
- **CBIS-DDSM** (public) est une base de référence pour la mammographie.

### Détecteur mammo (YOLO)
Dataset Ultralytics : images + labels `.txt` + un `data.yaml` (train/val + classes). Chemin du
`data.yaml` renseigné dans `configs/mammo_detector.yaml`.

### Histologie (Nottingham)
Lames annotées avec scores SBR (tubule/pléomorphisme/mitose) — dataset à constituer (pseudonymisé).

## Pseudonymisation
Toujours avant usage : retirer identifiants directs, dissocier les clés. Voir `../GOVERNANCE.md`.
