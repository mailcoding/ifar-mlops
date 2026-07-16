#!/usr/bin/env bash
# Extraction du sous-dossier mlops/ vers un dépôt dédié « ifar-mlops », HISTORIQUE PRÉSERVÉ.
#
#   bash mlops/scripts/extract_to_repo.sh git@github.com:mailcoding/ifar-mlops.git
#
# - Historique préservé via `git filter-repo --subdirectory-filter mlops` (repli : copie simple).
# - Travaille sur un CLONE JETABLE dans un dossier temporaire : ne modifie JAMAIS ton dépôt courant.
# - Le dossier .github/ vivant DANS mlops/ remonte automatiquement à la racine du nouveau dépôt.
# - Refuse de tourner si des modifications non committées existent sous mlops/ (elles seraient perdues,
#   un clone ne prenant que l'historique committé).
set -euo pipefail

REMOTE="${1:-}"
if [[ -z "$REMOTE" ]]; then
  echo "Usage : bash mlops/scripts/extract_to_repo.sh <git-remote-url>" >&2
  echo "  ex. : bash mlops/scripts/extract_to_repo.sh git@github.com:mailcoding/ifar-mlops.git" >&2
  exit 2
fi

# On doit être à la racine du monorepo produit (là où vit mlops/).
if [[ ! -d mlops ]]; then
  echo "ERREUR : lance ce script depuis la RACINE du monorepo (dossier mlops/ introuvable)." >&2
  exit 2
fi

SRC="$(git rev-parse --show-toplevel)"

# Garde-fou : rien de non committé sous mlops/ (sinon perte silencieuse au clone).
if [[ -n "$(git -C "$SRC" status --porcelain -- mlops)" ]]; then
  echo "ERREUR : des modifications NON COMMITTÉES existent sous mlops/." >&2
  echo "        Commit (ou stash) d'abord — le clone ne prend que l'historique committé." >&2
  exit 2
fi

WORK="$(mktemp -d)"
DEST="$WORK/ifar-mlops"
echo "Clone jetable → $DEST"
git clone --no-local --quiet "$SRC" "$DEST"
cd "$DEST"

if git filter-repo --help >/dev/null 2>&1; then
  echo "Extraction AVEC historique (git filter-repo --subdirectory-filter mlops)…"
  git filter-repo --force --subdirectory-filter mlops
else
  echo "git-filter-repo introuvable → repli COPIE SIMPLE (SANS historique)."
  echo "  (Pour préserver l'historique : pip install git-filter-repo, puis relance.)"
  cd "$WORK"
  rm -rf "$DEST"
  mkdir -p "$DEST"
  cp -a "$SRC/mlops/." "$DEST/"
  cd "$DEST"
  git init -q
  git add -A
  git -c user.name=ifar -c user.email=ifar@local \
      commit -qm "init ifar-mlops (copie du scaffold, sans historique)"
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"

echo
echo "✅ Dépôt prêt dans : $DEST"
echo "   Contenu à la racine :"
ls -A "$DEST" | sed 's/^/     /'
echo
echo "Étapes suivantes :"
echo "  cd \"$DEST\""
echo "  git branch -M main"
echo "  git push -u origin main"
echo
echo "(Ce dossier temporaire n'est pas nettoyé automatiquement : $WORK)"
