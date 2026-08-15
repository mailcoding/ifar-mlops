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

### 🚦 Porte de validation clinique
**Seul** moyen de passer un artefact à `validated: true` (le produit et le déploiement lisent ce
champ ; un artefact non validé est refusé au déploiement) :
```bash
python -m mlops.validation.gate --artifact artifacts/mammo-clf \
    --validated-by "Dr X (radiologue)" \
    --test-set "CBIS-DDSM test — indépendant, jamais vu à l'entraînement"
# --profile detector | --target auc=0.90 (surcharge) | --dry-run (évaluer sans écrire)
```
Contrôles (**fail closed** : une métrique absente = échec) : métriques ≥ cibles
(classifieur AUC ≥ 0,85 · sensibilité ≥ 0,90 · spécificité ≥ 0,75), **validateur nommé**,
**jeu de test indépendant** déclaré, version + hash des poids. Les cibles sont **figées dans la
décision** (critères définis *avant*, cf. SaMD §4.6) et le procès-verbal go/no-go est écrit dans
`manifest.json` (`validation`). Code de sortie 1 si no-go → bloque une CI.

> La porte vérifie que les **preuves existent et atteignent les seuils**. Elle ne remplace pas
> l'étude clinique (rétrospective/prospective, représentativité, sous-groupes, faux négatifs).

### 🔁 Voie « amélioration » (déploiement sans validation absolue)
Le modèle **en production n'a lui non plus jamais passé de porte**. Refuser un candidat *meilleur*
au seul motif qu'il n'atteint pas les cibles absolues revient à **préserver le pire modèle**. Cette
voie répond à une question différente — *« est-ce mieux que ce qui tourne aujourd'hui ? »* :
```bash
python -m mlops.validation.improvement --artifact artifacts/mammo-clf \
    --incumbent-metrics '{"auc":0.77,"sensitivity":0.63,"specificity":0.74}' \
    --approved-by "Dr X" --rationale "sensibilité +0,20 ; spécificité −0,04 assumée" \
    --review-by 2026-11-15 --same-test-set
```
Elle **n'est pas** une porte dérobée : `validated` n'est jamais modifié (seule la porte clinique le
peut), la porte clinique doit avoir été **exécutée** au préalable, la comparaison exige une
**attestation de jeu de test identique** (cf. `mlops.eval.classifier_eval` — comparer deux modèles
mesurés sur des splits différents ne prouve rien), et la décision est **nominative, justifiée et
bornée par une date de revue**. Règle par défaut : la **sensibilité doit progresser**, avec une
régression tolérée bornée ailleurs (spécificité ≤ 0,05, AUC 0).

Côté produit, le déploiement d'un tel artefact exige un **opt-in explicite**
(`--accept-improvement` / variable `ML_ACCEPT_IMPROVEMENT`), s'affiche en avertissement, et
`/health` expose `deployment.basis` pour que la carte ops montre « non validé cliniquement ».

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
