#!/usr/bin/env python3
"""Création idempotente des repos Hugging Face PRIVÉS d'ifar-mlops (model + dataset).

Usage :
    HF_TOKEN=hf_xxx python scripts/bootstrap_hf.py [--owner Mailcoding] [--dry-run]

Garanties :
- Ne pousse AUCUNE donnée ni poids : crée seulement les dépôts vides, en PRIVÉ.
- Idempotent (`exist_ok=True`) → réexécutable sans erreur.
- Le token doit avoir la portée WRITE. Rien de sensible n'est journalisé.

Les noms doivent rester alignés avec :
- registry (`src/mlops/registry/hf_hub.py`) et l'exemple du README ;
- data/README.md (repos dataset).
"""
from __future__ import annotations

import argparse
import os
import sys

# Repos "model" (poids versionnés, tags sémantiques vX.Y.Z + manifest + model card).
MODEL_REPOS = [
    "ifar-mammo-classifier",   # EfficientNetB0 bénin/malin
    "ifar-mammo-detector",     # YOLOv8-seg masses/calcifications
    "ifar-histo-grade",        # histologie / Nottingham (à venir)
]
# Repos "dataset" PRIVÉS et PSEUDONYMISÉS (jamais de PHI — cf. GOVERNANCE.md).
DATASET_REPOS = [
    "ifar-mammo-rois",         # ROIs mammo (manifestes path,label)
    "ifar-histo",              # lames histo annotées (SBR)
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap des repos Hugging Face privés d'ifar-mlops (model + dataset)."
    )
    parser.add_argument(
        "--owner",
        default=os.getenv("HF_OWNER", "Mailcoding"),
        help="Compte/organisation HF propriétaire (défaut : Mailcoding).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="N'appelle pas l'API : affiche seulement ce qui serait créé.",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN", "").strip()
    if not token and not args.dry_run:
        print(
            "ERREUR : HF_TOKEN manquant (token Hugging Face WRITE).\n"
            "  Ex. : HF_TOKEN=hf_xxx python scripts/bootstrap_hf.py",
            file=sys.stderr,
        )
        return 2

    plan = [("model", r) for r in MODEL_REPOS] + [("dataset", r) for r in DATASET_REPOS]
    mode = "DRY-RUN" if args.dry_run else "création"
    print(f"Owner : {args.owner} — {len(plan)} repo(s) privé(s) à garantir ({mode}) :")
    for repo_type, name in plan:
        print(f"  - [{repo_type:7}] {args.owner}/{name}")

    if args.dry_run:
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "ERREUR : huggingface_hub non installé. `pip install huggingface_hub`.",
            file=sys.stderr,
        )
        return 2

    api = HfApi(token=token)
    failures = 0
    for repo_type, name in plan:
        repo_id = f"{args.owner}/{name}"
        try:
            url = api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=True,
                exist_ok=True,
            )
            print(f"  ✓ {repo_type:7} {repo_id} → {url}")
        except Exception as exc:  # noqa: BLE001 — on rapporte proprement, sans crash
            failures += 1
            print(
                f"  ✗ {repo_type:7} {repo_id} : {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if failures:
        print(
            f"{failures} échec(s). Vérifie la portée WRITE du token et tes droits sur "
            f"l'owner « {args.owner} ».",
            file=sys.stderr,
        )
        return 1

    print("OK — tous les repos privés sont garantis (privés, vides ou existants).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
