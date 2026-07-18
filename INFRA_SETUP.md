# INFRA_SETUP — Mise en place du dépôt MLOps `ifar-mlops`

Runbook de bout en bout pour **sortir ce scaffold `mlops/` du monorepo produit** vers un **dépôt GitHub
dédié `ifar-mlops`**, créer les **repos Hugging Face privés** (model + dataset), et **brancher la CI**.

> Contexte : le classifieur livré est peu fiable (val acc ~62 %, AUC ~0,6). Le dépôt MLOps permet de
> **ré-entraîner et versionner** un modèle atteignant les cibles (**AUC ≥ 0,85**, **sensibilité ≥ 0,90**).
> Le produit (`ml-service`) ne consomme qu'un **artefact versionné** publié sur HF. Conception :
> `ifar/docs/MLOPS_ARCHITECTURE.md` (côté produit).

Ordre recommandé : **1 → 6**. Les commandes supposent que tu es à la **racine du monorepo produit**
(le dossier qui contient `mlops/`), sauf indication contraire.

---

## Pré-requis
- `git` + (recommandé) **`git-filter-repo`** pour préserver l'historique : `pip install git-filter-repo`.
- Un **compte GitHub** avec droit de créer un repo sous `mailcoding/`.
- Un **compte Hugging Face** (`Mailcoding`) et un **token WRITE** : <https://huggingface.co/settings/tokens>.
- `pip install huggingface_hub` (pour `scripts/bootstrap_hf.py`).

---

## 1. Créer le dépôt GitHub `ifar-mlops` (PRIVÉ)
Crée un dépôt **privé et VIDE** `mailcoding/ifar-mlops` (sans README/licence auto : le contenu vient de
l'extraction). Via l'UI GitHub (**New repository** → Private → *Create*), ou en CLI si tu as `gh` :
```bash
gh repo create mailcoding/ifar-mlops --private --description "IFAR CancerSein AI — entraînement & versioning des modèles (MLOps)"
```
> Note : ce dépôt est **hors du périmètre** de la session Claude Code (limitée à
> `mailcoding/ifarcancerseinai`) — création à faire par toi.

## 2. Extraire `mlops/` avec l'historique et pousser
Depuis la racine du monorepo :
```bash
bash mlops/scripts/extract_to_repo.sh git@github.com:mailcoding/ifar-mlops.git
```
Le script travaille sur un **clone jetable** (ton dépôt courant n'est jamais modifié), applique
`git filter-repo --subdirectory-filter mlops` (le `.github/` remonte à la racine), configure le remote,
et affiche le dossier prêt. Termine par :
```bash
cd <dossier-affiché-par-le-script>/ifar-mlops
git branch -M main
git remote set-url origin https://github.com/mailcoding/ifar-mlops.git
git push -u origin main
```
Garde-fou : le script **refuse** de tourner si des changements non committés traînent sous `mlops/`
(ils seraient perdus). Commit d'abord.

## 3. Créer les repos Hugging Face PRIVÉS (model + dataset)
Depuis le nouveau dépôt `ifar-mlops` (ou n'importe où — le script est autonome) :
```bash
pip install huggingface_hub
HF_TOKEN=hf_xxx python scripts/bootstrap_hf.py           # création réelle (privée, idempotente)
# ou, pour juste voir ce qui serait créé :
python scripts/bootstrap_hf.py --dry-run
```
Repos créés (privés) — `--owner Mailcoding` par défaut :
- **model** : `Mailcoding/ifar-mammo-classifier`, `Mailcoding/ifar-mammo-detector`, `Mailcoding/ifar-histo-grade`
- **dataset** : `Mailcoding/ifar-mammo-rois`, `Mailcoding/ifar-histo`

Aucune donnée n'est poussée ici — seulement des dépôts vides. **Jamais de PHI** (voir `GOVERNANCE.md`).

## 4. Poser le secret Actions `HF_TOKEN`
Dans le dépôt `ifar-mlops` → **Settings → Secrets and variables → Actions → New repository secret** :
- Nom : `HF_TOKEN` — Valeur : ton token HF **WRITE**.

Nécessaire au workflow de publication (`.github/workflows/publish.yml`). La CI de test (`ci.yml`)
n'en a pas besoin.

## 5. Vérifier la CI de test
Au 1er `git push` sur `main` (ou sur une PR), le workflow **CI** (`.github/workflows/ci.yml`) tourne :
`ruff check` + `pytest` (métriques/export, **sans GPU**). Il doit être **vert**. En local :
```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

## 6. Première publication d'artefact (dry-run de bout en bout)
Objectif : valider la chaîne export → publication → tag, avant tout vrai entraînement.
```bash
pip install -e ".[train]"      # torch, timm, … (GPU non requis pour ce test jouet)
python - <<'PY'
from mlops.models.efficientnet import build_pretrained_backbone
from mlops.export import export_classifier
model = build_pretrained_backbone()               # modèle jouet (non entraîné)
out = export_classifier(
    model, "artifacts/mammo-clf", version="v0.0.1",
    metrics={"auc": None, "sensitivity": None, "note": "artefact de test, NON validé"},
    trained_on={"dataset": "none", "n_train": 0, "n_val": 0},
)
print(out)   # → efficientnet_b0.pth + manifest.json + MODEL_CARD.md
PY
# Publier (repo HF privé) + créer le tag v0.0.1 :
HF_TOKEN=hf_xxx python -c "from mlops.registry import publish_artifact; \
  print(publish_artifact('artifacts/mammo-clf', 'Mailcoding/ifar-mammo-classifier', 'v0.0.1'))"
```
> ⚠️ Cet artefact `v0.0.1` est un **modèle jouet non validé** : ne JAMAIS le charger en prod. Il sert
> uniquement à prouver que le pipeline de versioning fonctionne. Supprime le tag si besoin.

Alternative CI : onglet **Actions → Publish artifact (HF Hub) → Run workflow** (fournir `repo_id`,
`version`, `artifacts_dir`) — utile quand l'artefact est produit/téléchargé dans le run.

---

## Après la mise en place (étapes suivantes, hors de ce runbook)
1. **Rendre l'entraînement exécutable** : construire les manifestes CBIS (`train.csv`/`val.csv`,
   **split par patient**), renforcer la recette (`configs/mammo_classifier.yaml` : class weights,
   augmentations, calibration), lancer sur Colab GPU jusqu'à **AUC ≥ 0,85 / sensibilité ≥ 0,90**.
2. **Porte de validation clinique** (SaMD / IEC 62304 / ISO 14971) avant publication d'une version
   « validée » (`manifest.json` → `validated: true`).
3. **Consommation produit** : le Space `ml-service` récupère la version publiée (`pull_artifact` /
   `deploy-ml-space.yml`) ; le nom de fichier doit rester `efficientnet_b0.pth` (cf. `export/`).
4. **Nettoyage** : une fois `ifar-mlops` opérationnel, **retirer `mlops/` du monorepo produit**
   (action séparée, non destructive tant que non confirmée).

## Consommation d'une version par le ml-service
```bash
HF_TOKEN=hf_read python -c "from mlops.registry import pull_artifact; \
  print(pull_artifact('Mailcoding/ifar-mammo-classifier', 'v1.0.0', 'ml-service/models'))"
```
Le `ml-service` charge ensuite `models/efficientnet_b0.pth` via `resolve_weights`
(`ifar/ml-service/app/utils.py`). Voir `MLOPS_ARCHITECTURE.md` pour le contrat complet.
