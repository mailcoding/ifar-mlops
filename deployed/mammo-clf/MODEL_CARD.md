# Model Card — ifar-mammo-classifier `unpublished-cbis-ddsm-d4177e86`

> Carte **rétrospective**, établie après coup pour un artefact déjà en production. Elle ne
> régularise rien : elle documente ce qui tourne, et ce qui manque. Gabarit :
> `MODEL_CARD/TEMPLATE.md`.

## Identité
- **Nom / version** : ifar-mammo-classifier · `unpublished-cbis-ddsm-d4177e86`
  Le libellé n'est **pas** un tag sémantique, délibérément : `GOVERNANCE.md` réserve `vX.Y.Z`
  aux modèles publiés par le pipeline. Ces poids ne l'ont pas été.
- **Tâche** : classification mammographique bénin / malin (aide à la décision).
- **Framework / architecture** : timm `efficientnet_b0`, entrée 224 px, normalisation ImageNet.
- **Fichier d'artefact** : `models/cbis_ddsm_efficientnet_final.pth` (17 649 755 octets)
  **SHA-256** : `d4177e86ebc6eb1c408533468067c77d930a4d89ffadf4e6662fc0086b22e78a`
- **Commit d'entraînement** : **inconnu** — poids déposés à la main sur le Space, hors pipeline.
- **Déploiement** : Space HF `Mailcoding/ifar_ml`. Le ml-service essaie `efficientnet_b0.pth`,
  puis ce fichier, puis `best_model.pth` ; le premier étant absent, c'est bien celui-ci qui sert,
  ce que `/health` confirme.

## Usage prévu
- **Indication** : aide à la décision. **Jamais un diagnostic autonome.** Tout verdict est
  soumis à la validation d'un radiologue.
- **Utilisateurs** : radiologues habilités.
- **Hors périmètre** : tout. Ce modèle **n'est validé pour aucun usage clinique** (voir
  §Validation). Il ne doit pas servir de fondement à une décision non revue par un praticien.

## Données
- **Jeu d'entraînement** : CBIS-DDSM. Taille, répartition et découpage **non consignés** ;
  hash du jeu **inconnu** — la reproductibilité exigée par `GOVERNANCE.md` n'est pas assurée.
- **Jeu de validation / test** : **non identifié.**
- **Limites de représentativité** : CBIS-DDSM est un corpus nord-américain numérisé à partir de
  films. La population cible (Sénégal), les appareils et les protocoles d'acquisition en usage
  ne sont pas représentés. Aucune évaluation par sous-groupe n'existe.

## Performance

| Métrique | Valeur | Seuil |
|---|---|---|
| AUC | **non mesurée** | cible ≥ 0,85 |
| Sensibilité | **non mesurée** | cible ≥ 0,90 |
| Spécificité | **non mesurée** | — |
| Calibration | **non mesurée** | — |

`metrics` est vide dans le manifeste, et c'est un constat, pas un oubli : **aucune évaluation
reproductible de ces poids n'est consignée dans ce dépôt.** Des chiffres d'environ 62 %
d'exactitude en validation et AUC ~0,6 ont été *rapportés* à l'énoncé du projet, sans jeu de
test identifié ni date ; ils ne sont pas repris comme mesures et ne doivent pas être consommés
par `validation/gate.py`.

- **Par sous-groupe** : aucune.
- **Calibration** : le classifieur émet une softmax brute, non calibrée. C'est la raison pour
  laquelle sa sortie n'est pas transposée en bandes de valeur prédictive positive BI-RADS.

## Sécurité & risques (ISO 14971)
- **Faux négatifs** : le risque dominant — un cancer non signalé. Le seuil de malignité est
  abaissé à **0,40** (`ml-service/app/classification.py`, `MALIGNANCY_THRESHOLD`) pour favoriser
  la sensibilité, mais **sans mesure d'accompagnement** : le gain réel n'est pas quantifié, et
  l'arbitrage sensibilité/spécificité attend le comité clinique.
- **Faux positifs** : biopsies évitables, anxiété. Non quantifiés.
- **Mode de défaillance silencieux** : sans évaluation, une dérive de performance ne serait pas
  détectable. Aucun indicateur de suivi post-déploiement n'est en place.
- **Atténuation en vigueur** : validation humaine systématique par le radiologue, et affichage
  du caractère non validé du modèle sur la carte ops (via ce manifeste).

## Validation
- **Porte de validation clinique** : **non franchie.** Aucune décision go/no-go documentée,
  aucun responsable nommé. `validated: false`, `validated_by: ""`.
- **Autorisation de déploiement « amélioration »** : aucune non plus — le champ `deployment`
  est absent du manifeste, donc `/health` rend `deployment: null`.
- **Suivi post-déploiement** : à mettre en place. Ce manifeste en est le préalable : sans
  identification de l'artefact, aucun indicateur n'est rattachable à une version.

## Ce qu'il faut pour lever ces réserves
1. Évaluer ces poids sur un jeu de test identifié et versionné, et consigner AUC, sensibilité,
   spécificité et calibration — `mlops.eval.classifier_eval` est prévu pour cela.
2. Soumettre le résultat à `validation/gate.py` (cibles AUC ≥ 0,85 / sensibilité ≥ 0,90).
3. Faire trancher par le comité clinique le seuil de 0,40 et le critère de transmission
   (cf. `ifarCancerSeinAI/ifar/docs/CORRECTION_SCORES_ET_TRANSMISSION.md`).
4. Republier alors l'artefact par le pipeline, avec un vrai tag sémantique.
