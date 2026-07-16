# src/mlops/train/train_histology.py
# ─────────────────────────────────────────────
# Entraînement du modèle d'histologie (grade de Nottingham) — CIBLE PRIORITAIRE :
# le produit n'a aujourd'hui qu'un placeholder heuristique
# (ifar/ml-service/app/histology.py, model="heuristic-placeholder").
#
# Contrat de sortie à respecter (cf. apps/ml_gateway/models.py::HistologyAnalysis
# et ml-service schemas.py) :
#   tubule_score, pleomorphism_score, mitosis_score ∈ {1,2,3}
#   nottingham_total = somme (3–9)  →  nottingham_grade ∈ {1: I (3–5), 2: II (6–7), 3: III (8–9)}
#   pathology (BENIGN/MALIGNANT), malignancy_probability, model_name (= version publiée)
#
# Approches possibles :
#   (a) 3 têtes de régression/classification (tubule/pléomorphisme/mitose) → grade dérivé ;
#   (b) classification directe du grade (I/II/III) + score de malignité.
#
# ⚠️ Squelette : nécessite un dataset histologique annoté (lames + scores SBR),
#    pseudonymisé, hébergé hors git (repo dataset HF privé). Voir data/README.md.
# ─────────────────────────────────────────────

import argparse
from pathlib import Path

import yaml


def nottingham_grade(total: int) -> int:
    """Grade de Nottingham/SBR à partir du score total (3–9)."""
    if total <= 5:
        return 1  # Grade I
    if total <= 7:
        return 2  # Grade II
    return 3      # Grade III


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())  # noqa: F841 (utilisé au moment de l'implémentation)
    raise NotImplementedError(
        "Entraînement histologie à implémenter (dataset annoté requis).\n"
        "Étapes : (1) loader dataset pseudonymisé ; (2) backbone (ex. EfficientNet/ViT) "
        "avec 3 têtes de scores SBR ; (3) évaluation par score et par grade dérivé "
        "(nottingham_grade) ; (4) export au format ml-service (histology.pth) + manifeste ; "
        "(5) intégration via un wrapper HistologyAnalyzer respectant le contrat de sortie."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
