# Model Card — <nom-du-modèle> <vX.Y.Z>

> Gabarit SaMD. À publier avec chaque artefact (repo model HF, à côté de `manifest.json`).
> Aligné sur `SaMD_CADRE_REGLEMENTAIRE.md` et `MODEL_CARD.md` (produit).

## Identité
- **Nom / version** : …
- **Tâche** : (ex. classification mammo bénin/malin ; grade de Nottingham)
- **Framework / architecture** : (ex. timm efficientnet_b0 + tête custom)
- **Fichier d'artefact** : (ex. `efficientnet_b0.pth`) · **SHA-256** : …
- **Commit d'entraînement** : … · **Date** : …

## Usage prévu
- **Indication** : aide à la décision (jamais un diagnostic autonome).
- **Utilisateurs** : radiologues / anatomopathologistes / cliniciens habilités.
- **Hors périmètre** : … (ce pour quoi le modèle n'est PAS validé)

## Données
- **Jeu d'entraînement** : (source, taille, pseudonymisation) · **hash** : …
- **Jeu de validation/test** : … · **Répartition des classes** : …
- **Limites de représentativité** : (population, appareils, sous-groupes)

## Performance
| Métrique | Valeur | Seuil |
|---|---|---|
| AUC | … | — |
| Sensibilité | … | (seuil clinique) |
| Spécificité | … | |
| Calibration | … | |
- **Par sous-groupe** (âge, densité mammaire, appareil…) : …

## Sécurité & risques (ISO 14971)
- **Modes de défaillance** & atténuations : …
- **Faux négatifs / faux positifs** : impact clinique et conduite à tenir.

## Validation
- **Porte de validation** : go/no-go, date, responsable : …
- **Suivi post-déploiement** : indicateurs surveillés (dashboard ops), critères de retrait.
