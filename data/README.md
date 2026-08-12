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

**Génération automatique** (split PAR PATIENT anti-fuite) depuis les CSV de description de cas CBIS :
```bash
python -m mlops.datasets.build_cbis_manifest \
    --case-csv mass_case_description_train_set.csv \
    --case-csv calc_case_description_train_set.csv \
    --case-csv mass_case_description_test_set.csv \
    --case-csv calc_case_description_test_set.csv \
    --dicom-info dicom_info.csv --images-root /data/cbis \
    --use cropped --val-frac 0.2 --out-dir data
```
Écrit `data/train.csv` + `data/val.csv`, garantit qu'aucun patient n'est à la fois en train et val,
mappe `BENIGN_WITHOUT_CALLBACK → BENIGN`, et **suggère les `class_weights`** (déséquilibre) à reporter
dans `configs/mammo_classifier.yaml`. Options :
- `--use cropped` (ROI, défaut) ou `--use full` (mammographie entière) ;
- `--dicom-info dicom_info.csv` (export Kaggle) → jointure `SeriesInstanceUID → image_path` ; sinon
  repli sur un arbre DICOM converti en jpeg (`…/SeriesUID/000000.jpg`) via `--images-root` ;
- `--no-verify` pour planifier à sec sans vérifier l'existence des fichiers.

### Détecteur mammo (YOLO)
Dataset Ultralytics : images + labels `.txt` + un `data.yaml` (train/val + classes `0=mass`,
`1=calcification`). Chemin du `data.yaml` renseigné dans `configs/mammo_detector.yaml`.

**Génération depuis CBIS** (masques ROI → polygones YOLO-seg, split officiel par patient, image liée
en pleine résolution) :
```bash
python -m mlops.datasets.build_yolo_seg_dataset \
    --train-csv mass_case_description_train_set.csv --train-csv calc_case_description_train_set.csv \
    --val-csv   mass_case_description_test_set.csv  --val-csv   calc_case_description_test_set.csv  \
    --images-root /data/cbis --out-dir /data/ifar/yolo_seg
```

**Généralisation — fusion VinDr-Mammo** (FFDM moderne, findings en bounding boxes). Se déverse dans
le **même** `--out-dir` (fichiers préfixés `vindr_`, `data.yaml` existant préservé) :
```bash
python -m mlops.datasets.build_vindr_yolo_dataset \
    --annotations finding_annotations.csv --images-root /data/vindr/images \
    --out-dir /data/ifar/yolo_seg --mode seg
```
- `--mode seg` : box → polygone rectangle (compatible dataset CBIS-seg) ; `--mode detect` : box `xywh`.
- Classes VinDr mappées `Mass→mass`, `*Calcification→calcification` (autres findings ignorés) ;
  split officiel VinDr (`training`/`test`) disjoint par étude ; images « No Finding » → **négatifs**
  (labels vides) pour réduire les faux positifs. `--dry-run` planifie sans écrire.
- ⚠️ VinDr n'a **pas** de vérité histologique (BI-RADS seulement) → il sert le **détecteur**
  (localisation/généralisation), pas le classifieur bénin/malin.

### Histologie (Nottingham)
Lames annotées avec scores SBR (tubule/pléomorphisme/mitose) — dataset à constituer (pseudonymisé).

## Pseudonymisation
Toujours avant usage : retirer identifiants directs, dissocier les clés. Voir `../GOVERNANCE.md`.
