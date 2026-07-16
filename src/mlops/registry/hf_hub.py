# src/mlops/registry/hf_hub.py
# ─────────────────────────────────────────────
# Registre d'artefacts modèles via Hugging Face Hub (repos "model" PRIVÉS).
# Une VERSION = un tag sémantique (v1.2.0) portant poids (LFS) + manifest.json
# + MODEL_CARD.md. Le ml-service récupère ensuite ces poids (Space HF).
#
# Auth : variable d'environnement HF_TOKEN (write). Rien n'est versionné en clair.
# ─────────────────────────────────────────────

import os
from pathlib import Path


def _token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN manquant (token Hugging Face WRITE).")
    return token


def publish_artifact(local_dir: str | Path, repo_id: str, version: str,
                     private: bool = True, commit_message: str | None = None) -> str:
    """
    Publie le contenu de `local_dir` (poids + manifest.json + MODEL_CARD.md) dans un
    repo model HF, puis crée un TAG de version. Retourne l'URL du repo.

    Exemple : publish_artifact("artifacts/mammo-clf", "Mailcoding/ifar-mammo-classifier", "v1.2.0")
    """
    from huggingface_hub import HfApi

    api = HfApi(token=_token())
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message or f"Publish {version}",
    )
    # Tag de version (immuable) pour la traçabilité (SaMD / manifeste).
    try:
        api.create_tag(repo_id, tag=version, repo_type="model")
    except Exception as exc:  # noqa: BLE001 — tag déjà existant, etc.
        print(f"[warn] tag {version} non créé : {exc}")
    return f"https://huggingface.co/{repo_id}"


def pull_artifact(repo_id: str, version: str, dest: str | Path) -> str:
    """Télécharge une version (tag) d'un artefact modèle vers `dest`. Retourne le chemin local."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id, repo_type="model", revision=version,
        local_dir=str(dest), token=os.getenv("HF_TOKEN") or None,
    )
    return path
