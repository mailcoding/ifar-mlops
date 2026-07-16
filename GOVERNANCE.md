# Gouvernance des données & des modèles — ifar-mlops

Ce dépôt entraîne des modèles à partir de données de santé. Règles impératives (RGPD, loi sénégalaise
2008-12 / CDP, cadre SaMD).

## Données
- **Jamais de PHI ni de données brutes dans git.** Seuls des **pointeurs** (manifestes, hachages) sont
  versionnés. Les images/labels vivent dans un **repo dataset Hugging Face privé** (ou stockage objet).
- **Pseudonymisation/anonymisation obligatoire** avant tout usage d'entraînement : suppression des
  identifiants directs, dissociation des clés. Aucun numéro de dossier, nom, CNI, etc.
- **Finalité `research`** — distincte du soin. Cohérente avec le consentement capturé côté produit
  (V6 de l'audit). Ne pas entraîner sur des données dont la finalité research n'est pas couverte.
- **Résidence & sous-traitants** : documenter où sont stockés datasets et poids (cf.
  `SOUS_TRAITANTS_ET_RESIDENCE.md` du produit).
- **DPIA** : mettre à jour si de nouvelles catégories de données d'entraînement sont introduites.
- **Rétention** : politique dédiée aux datasets d'entraînement (distincte de la rétention clinique).

## Modèles & versions
- Chaque modèle publié = **tag sémantique** (`vX.Y.Z`) + `manifest.json` + **model card**
  (`MODEL_CARD/TEMPLATE.md`) : données, métriques (AUC, sensibilité/spécificité), seuils, limites,
  sous-groupes, usage prévu, avertissements.
- **Porte de validation clinique** avant toute mise en production (dispositif médical logiciel) :
  décision documentée go/no-go, seuils cliniques, cf. `SaMD_CADRE_REGLEMENTAIRE.md` (IEC 62304 / ISO 14971).
- Traçabilité : le manifeste lie l'artefact au **commit d'entraînement**, au **hash du dataset** et aux
  **métriques** — reproductibilité et auditabilité.

## Accès (moindre privilège)
- Dépôt MLOps et repos HF (model/dataset) à **accès restreint** ; pas de compte/clé partagé avec le
  produit. Le token HF (`HF_TOKEN`) est un secret CI/local, jamais versionné.

## Sécurité
- Pas de secrets en clair dans le code ni les configs (`.env`/secrets CI uniquement).
- `data/.gitignore` bloque tout dépôt accidentel de données.
