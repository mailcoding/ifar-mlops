# `deployed/` — descripteurs de ce qui tourne réellement en production

Ce dossier est **suivi par git**, contrairement à `artifacts/` que `.gitignore` exclut.
La distinction est volontaire :

| Dossier | Contenu | Versionné ? |
|---|---|---|
| `artifacts/` | sorties de build : **poids**, manifeste et carte fraîchement produits | non — les poids vont sur le Hub |
| `deployed/` | **métadonnées seules** décrivant l'artefact effectivement servi en production | oui |

**Règle : aucun poids ici.** Uniquement des `manifest.json` et des `MODEL_CARD.md`. Les motifs
`*.pt`, `*.pth`, `*.onnx` et `*.safetensors` du `.gitignore` racine s'appliquent de toute façon,
mais la règle vaut d'être dite : ce dossier décrit, il ne transporte pas.

## Pourquoi il existe

Le manifeste est normalement écrit par `mlops export_classifier`, à côté des poids, au moment de
l'export. Quand des poids ont été déposés à la main sur le Space — ce qui est arrivé — aucun
manifeste ne les accompagne, et `/health` répond `version: "unknown"`, `source: "none"`. Le
modèle en production devient alors non identifiable : impossible de rattacher un verdict rendu à
un artefact précis, ce qui est une lacune d'auditabilité sur un dispositif médical.

Un manifeste rétrospectif écrit ici comble ce trou. Il ne régularise pas la situation au regard
de `GOVERNANCE.md` (tag sémantique, métriques, porte de validation clinique) : il la **rend
visible**, ce qui est le premier pas pour la corriger.

## Publier vers le Space

```bash
python scripts/publish_space_manifest.py --dry-run          # montre ce qui serait envoyé
HF_TOKEN=… python scripts/publish_space_manifest.py --apply  # envoie
```

Le jeton est un secret CI/local, jamais versionné (`GOVERNANCE.md` §Accès).

## Contenu

| Artefact | Statut |
|---|---|
| `mammo-clf/` | classifieur mammographique servi par le Space `Mailcoding/ifar_ml` — **non validé cliniquement**, sans évaluation consignée |
