#!/usr/bin/env python3
"""
Publie le manifeste d'un modèle déployé vers le Space HF, à l'emplacement où le ml-service
le cherche : `models/manifest.json`.

Pourquoi ce script existe : les poids ont été déposés à la main sur le Space, sans passer par
`mlops export_classifier` qui aurait écrit le manifeste à côté. Sans lui, `/health` répond
`version: "unknown"`, `source: "none"` — le modèle en production n'est pas identifiable.

Sûr par défaut : sans `--apply`, rien n'est envoyé et aucun jeton n'est requis.

    python scripts/publish_space_manifest.py                       # simulation
    HF_TOKEN=… python scripts/publish_space_manifest.py --apply     # envoi réel

Le jeton est un secret CI/local, jamais versionné (GOVERNANCE.md §Accès).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MANIFEST = "deployed/mammo-clf/manifest.json"
DEFAULT_SPACE = "Mailcoding/ifar_ml"
# Chemin imposé par le ml-service : MANIFEST_CANDIDATES dans app/model_manifest.py.
TARGET_PATH = "models/manifest.json"


def _load(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        sys.exit(f"Manifeste introuvable : {manifest_path}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"Manifeste illisible ({manifest_path}) : {exc}")


def _check(manifest: dict) -> list[str]:
    """
    Refuse de publier un manifeste qui affirmerait plus qu'il ne sait.

    Un manifeste sans empreinte ne trace rien ; un `validated: true` sans validateur nommé
    annoncerait une validation clinique qui n'a pas eu lieu. Les deux valent un arrêt, pas un
    avertissement : c'est en production que le mensonge se lirait.
    """
    problems = []
    if not manifest.get("version"):
        problems.append("`version` vide — le modèle resterait non identifiable.")
    if not manifest.get("weights_sha256"):
        problems.append("`weights_sha256` vide — aucun rattachement possible à un artefact.")
    if manifest.get("validated") and not manifest.get("validated_by"):
        problems.append("`validated: true` sans `validated_by` — validation non nominative.")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help=f"Manifeste à publier (défaut : {DEFAULT_MANIFEST})")
    parser.add_argument("--space", default=DEFAULT_SPACE,
                        help=f"Space HF cible (défaut : {DEFAULT_SPACE})")
    parser.add_argument("--apply", action="store_true",
                        help="Envoie réellement. Sans ce drapeau, simulation seule.")
    args = parser.parse_args()

    manifest_path = (REPO_ROOT / args.manifest).resolve()
    manifest = _load(manifest_path)
    raw = manifest_path.read_bytes()

    # `relative_to` lève pour un manifeste situé hors du dépôt : on n'affiche un chemin
    # relatif que lorsqu'il en est un.
    shown = (manifest_path.relative_to(REPO_ROOT)
             if manifest_path.is_relative_to(REPO_ROOT) else manifest_path)
    print(f"Manifeste : {shown}")
    print(f"Cible     : space {args.space} → {TARGET_PATH}")
    print(f"Taille    : {len(raw)} octets · sha256 {hashlib.sha256(raw).hexdigest()[:12]}")
    print()
    print(f"  version        : {manifest.get('version')}")
    print(f"  weights_file   : {manifest.get('weights_file')}")
    print(f"  weights_sha256 : {manifest.get('weights_sha256', '')[:12]}")
    print(f"  validated      : {manifest.get('validated')}"
          f"{' par ' + manifest['validated_by'] if manifest.get('validated_by') else ''}")
    print(f"  metrics        : {manifest.get('metrics') or '(aucune)'}")
    print()

    problems = _check(manifest)
    if problems:
        print("Publication refusée :")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    if not args.apply:
        print("Simulation — rien n'a été envoyé. Ajoute --apply pour publier.")
        return

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        sys.exit("HF_TOKEN absent de l'environnement (jeton d'écriture requis).")

    try:
        from huggingface_hub import upload_file
    except ImportError:
        sys.exit("huggingface_hub non installé : pip install -e .")

    url = upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo=TARGET_PATH,
        repo_id=args.space,
        repo_type="space",
        token=token,
        commit_message=f"manifeste du modèle déployé — {manifest.get('version')}",
    )
    print(f"Publié : {url}")
    print("Le Space redémarre ; vérifie ensuite le bloc `model` de /health "
          "(source doit passer à \"manifest\").")


if __name__ == "__main__":
    main()
