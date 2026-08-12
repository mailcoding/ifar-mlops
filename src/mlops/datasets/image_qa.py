# src/mlops/datasets/image_qa.py
# ─────────────────────────────────────────────
# Contrôle qualité des ROIs : détection des MASQUES binaires livrés à tort comme
# « cropped image » dans l'export Kaggle CBIS-DDSM (awsaf49). Un masque ROI est une
# image quasi-binaire (fond noir 0 / lésion blanche 255) ; s'entraîner dessus au lieu
# de la vraie ROI en niveaux de gris effondre l'AUC (cause typique d'un classifieur
# à ~0,6 d'AUC). On repère ces images par DEUX signaux conjoints :
#   • forte proportion de pixels aux extrêmes (≈0 / ≈255), ET
#   • faible entropie de l'histogramme (peu de niveaux de gris réels).
# Les deux conditions doivent être vraies → une vraie ROI contrastée n'est pas écartée.
#
# numpy est une dépendance cœur ; PIL n'est requis (extra [train]) que si l'on appelle
# réellement la détection → import paresseux à l'intérieur des fonctions.
# ─────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path

# Un pixel est « extrême » s'il est quasi-noir (<= NEAR_BLACK) ou quasi-blanc (>= NEAR_WHITE).
NEAR_BLACK = 16
NEAR_WHITE = 239
# Masque probable si AU MOINS cette fraction de pixels est extrême…
EXTREME_FRACTION_THRESHOLD = 0.90
# …ET l'entropie de l'histogramme (en bits) est <= ce seuil (un masque binaire ≈ 1 bit).
ENTROPY_BITS_THRESHOLD = 2.5
# Taille de travail (downscale, NEAREST pour préserver le caractère binaire d'un masque).
_WORK_SIZE = 64


def _gray_array(path: str | Path):
    """Ouvre l'image en niveaux de gris, downscale NEAREST → np.ndarray uint8 (H, W)."""
    import numpy as np
    from PIL import Image  # import paresseux (extra [train])

    with Image.open(path) as im:
        im = im.convert("L").resize((_WORK_SIZE, _WORK_SIZE), Image.NEAREST)
        return np.asarray(im, dtype=np.uint8)


def image_stats(path: str | Path) -> dict:
    """Statistiques d'intensité d'une image : extreme_fraction, entropy_bits, n_levels."""
    import numpy as np

    arr = _gray_array(path)
    total = int(arr.size)
    hist = np.bincount(arr.reshape(-1), minlength=256).astype(float)
    extreme = float(hist[: NEAR_BLACK + 1].sum() + hist[NEAR_WHITE:].sum()) / total
    p = hist / total
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    return {"extreme_fraction": extreme, "entropy_bits": entropy, "n_levels": int((hist > 0).sum())}


def mask_score(path: str | Path) -> float:
    """Score ∈ [0, 1] : proportion de pixels aux extrêmes (≈0 / ≈255). Plus haut = plus « masque »."""
    return image_stats(path)["extreme_fraction"]


def is_probable_mask(path: str | Path) -> bool:
    """True si l'image ressemble à un masque ROI binaire (à écarter de l'entraînement)."""
    st = image_stats(path)
    return st["extreme_fraction"] >= EXTREME_FRACTION_THRESHOLD and st["entropy_bits"] <= ENTROPY_BITS_THRESHOLD
