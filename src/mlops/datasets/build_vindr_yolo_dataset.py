# src/mlops/datasets/build_vindr_yolo_dataset.py
# ─────────────────────────────────────────────
# Construit un dataset YOLO pour le DÉTECTEUR à partir de VinDr-Mammo (FFDM moderne,
# ~5000 examens annotés en bounding boxes). Objectif : GÉNÉRALISATION du détecteur
# au-delà de CBIS-DDSM (numérisé sur film, population US) → images numériques + volume.
#
#   python -m mlops.datasets.build_vindr_yolo_dataset \
#       --annotations finding_annotations.csv --images-root /data/vindr/images \
#       --out-dir /data/ifar/yolo_seg --mode seg    # (fusionne avec le dataset CBIS)
#
# Points clés :
#   • Classes mappées sur le contrat produit : 0=mass, 1=calcification (autres findings ignorés).
#   • SPLIT OFFICIEL VinDr (colonne `split` : training→train, test→val) — disjoint par étude.
#   • FUSIONNE avec un dataset YOLO existant (même out_dir) : fichiers préfixés `vindr_` → pas de
#     collision avec CBIS. `mode=seg` (polygone RECTANGLE depuis la box, compatible YOLO-seg) ou
#     `mode=detect` (box xywh). Bornes normalisées via height/width fournis par les CSV VinDr.
#   • Images sans finding (« No Finding ») → labels VIDES (négatifs) : réduit les faux positifs.
#   • Boxes issues des CSV (pas de masque) → AUCUNE dépendance cv2 ; seul le lien d'images touche le disque.
#
# ⚠️ Schéma attendu (VinDr-Mammo 1.0, finding_annotations.csv) : colonnes study_id, series_id,
#    image_id, laterality, view_position, height, width, finding_categories, xmin, ymin, xmax, ymax,
#    split. Adapter la constante COLUMN_* si l'export diffère.
#
# ⚠️ Gouvernance : ni PHI ni images dans git — ce module lit un dataset local (pseudonymisé,
#    finalité research) et écrit des fichiers locaux non versionnés.
# ─────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from mlops.datasets.build_yolo_seg_dataset import CLASS_NAMES, dataset_yaml_text, polygon_to_label_line


# VinDr `finding_categories` (chaîne, parfois liste "['Mass']") → classe produit. Autres → None (ignoré).
def _map_category(finding_categories: str) -> int | None:
    """'Mass' → 0 ; '… Calcification' → 1 ; 'No Finding'/autres findings → None."""
    s = (finding_categories or "").lower()
    if "calcification" in s:
        return 1
    if "mass" in s:
        return 0
    return None


def _to_float(v: str) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x


def box_to_detect(xmin: float, ymin: float, xmax: float, ymax: float, w: float, h: float) -> list[float]:
    """Box absolue → YOLO detect [x_center, y_center, largeur, hauteur] normalisé, borné [0,1]."""
    def _clip(x):
        return max(0.0, min(1.0, x))
    return [_clip((xmin + xmax) / 2 / w), _clip((ymin + ymax) / 2 / h),
            _clip((xmax - xmin) / w), _clip((ymax - ymin) / h)]


def box_to_seg_polygon(xmin: float, ymin: float, xmax: float, ymax: float, w: float, h: float) -> list[float]:
    """Box absolue → polygone RECTANGLE YOLO-seg [x1,y1,…,x4,y4] normalisé (compatible dataset seg)."""
    def _c(x, d):
        return max(0.0, min(1.0, x / d))
    x1, x2, y1, y2 = _c(xmin, w), _c(xmax, w), _c(ymin, h), _c(ymax, h)
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def detect_label_line(class_id: int, box_norm: list[float]) -> str:
    """Ligne label YOLO detect : '<cls> xc yc w h' (6 décimales)."""
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in box_norm)


# Noms de colonnes (adapter si l'export VinDr diffère).
COL_IMAGE, COL_STUDY, COL_SPLIT = "image_id", "study_id", "split"
COL_CAT = "finding_categories"
COL_W, COL_H = "width", "height"
COL_BOX = ("xmin", "ymin", "xmax", "ymax")


def _read_annotations(annotations_csv: str) -> list[dict]:
    """Lit finding_annotations.csv → lignes normalisées (strip headers/valeurs)."""
    out: list[dict] = []
    with open(annotations_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [(x or "").strip() for x in (reader.fieldnames or [])]
        for raw in reader:
            out.append({(k or "").strip(): (v or "").strip() for k, v in raw.items()})
    return out


def _split_name(raw_split: str) -> str | None:
    s = (raw_split or "").strip().lower()
    if s in ("training", "train"):
        return "train"
    if s in ("test", "val", "validation"):
        return "val"
    return None


def resolve(annotations_csv: str, *, include_negatives: bool = True) -> tuple[dict, dict]:
    """Résout les annotations en images groupées par split (pur : CSV seul, sans image ni cv2).

    Retourne ({split: {image_id: {study_id, width, height, findings:[(cls, box)], negative}}}, stats)."""
    rows = _read_annotations(annotations_csv)
    splits: dict[str, dict] = {"train": {}, "val": {}}
    counts = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}}
    skipped_split, skipped_category = 0, 0

    for r in rows:
        split = _split_name(r.get(COL_SPLIT, ""))
        if split is None:
            skipped_split += 1
            continue
        image_id = r.get(COL_IMAGE, "")
        if not image_id:
            continue
        w, h = _to_float(r.get(COL_W, "")), _to_float(r.get(COL_H, ""))
        entry = splits[split].setdefault(
            image_id, {"study_id": r.get(COL_STUDY, ""), "width": w, "height": h,
                       "findings": [], "negative": True})
        if entry["width"] is None:
            entry["width"], entry["height"] = w, h

        cls = _map_category(r.get(COL_CAT, ""))
        box = tuple(_to_float(r.get(k, "")) for k in COL_BOX)
        has_box = all(v is not None for v in box)
        if cls is None:
            if has_box:
                skipped_category += 1  # finding réel hors mass/calc (distorsion, asymétrie…)
            continue                   # sans box (« No Finding ») → l'image reste un négatif
        if not has_box:
            continue                   # classe connue mais coordonnées manquantes → on saute
        entry["findings"].append((cls, box))
        entry["negative"] = False
        counts[split][cls] += 1

    # Retire les images négatives si non demandées.
    if not include_negatives:
        for split, images in splits.items():
            splits[split] = {k: v for k, v in images.items() if not v["negative"]}

    def _stats(split):
        imgs = splits[split]
        return {
            "n_images": len(imgs),
            "n_positive": sum(1 for v in imgs.values() if not v["negative"]),
            "n_negative": sum(1 for v in imgs.values() if v["negative"]),
            "n_findings": {"mass": counts[split][0], "calcification": counts[split][1]},
            "studies": sorted({v["study_id"] for v in imgs.values() if v["study_id"]}),
        }

    stats = {"train": _stats("train"), "val": _stats("val"),
             "skipped_split": skipped_split, "skipped_category": skipped_category}
    return splits, stats


def index_images_by_id(images_root: str) -> dict[str, str]:
    """Indexe les images par image_id (nom de fichier sans extension) → chemin."""
    index: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(images_root):
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                index.setdefault(os.path.splitext(name)[0], os.path.join(dirpath, name))
    return index


def _materialize(splits: dict, out_dir: Path, image_index: dict[str, str], *,
                 mode: str, link: bool) -> dict:
    """Écrit images (liées) + labels YOLO préfixés `vindr_`. Retourne les compteurs par split."""
    result = {}
    for split, images in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        written, unresolved = 0, 0
        for image_id, entry in images.items():
            src = image_index.get(image_id)
            if src is None:
                unresolved += 1
                continue
            w, h = entry["width"], entry["height"]
            lines = []
            for cls, (xmin, ymin, xmax, ymax) in entry["findings"]:
                if not w or not h:
                    continue
                if mode == "detect":
                    lines.append(detect_label_line(cls, box_to_detect(xmin, ymin, xmax, ymax, w, h)))
                else:
                    lines.append(polygon_to_label_line(cls, box_to_seg_polygon(xmin, ymin, xmax, ymax, w, h)))
            # image négative → fichier label VIDE (négatif explicite) ; positive sans ligne → on saute.
            if not lines and not entry["negative"]:
                continue

            name = f"{split}_vindr_{image_id}{os.path.splitext(src)[1]}"
            dst = img_dir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            src_abs = os.path.abspath(src)
            if link:
                os.symlink(src_abs, dst)
            else:
                import shutil
                shutil.copy(src_abs, dst)
            (lbl_dir / f"{split}_vindr_{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            written += 1
        result[split] = {"written": written, "unresolved_image": unresolved}
    return result


def build(annotations_csv: str, images_root: str, out_dir: str, *, mode: str = "seg",
          include_negatives: bool = True, link: bool = True, materialize: bool = True) -> dict:
    """Construit (ou fusionne) le dataset YOLO VinDr. Retourne un rapport (stats + fuite étude)."""
    if mode not in ("seg", "detect"):
        raise ValueError("mode doit être 'seg' ou 'detect'")
    splits, stats = resolve(annotations_csv, include_negatives=include_negatives)
    leak = sorted(set(stats["train"]["studies"]) & set(stats["val"]["studies"]))
    report = {"mode": mode, "train": stats["train"], "val": stats["val"],
              "skipped_split": stats["skipped_split"], "skipped_category": stats["skipped_category"],
              "study_leak": leak}

    if materialize:
        out = Path(out_dir)
        image_index = index_images_by_id(images_root)
        report["n_images_indexed"] = len(image_index)
        report["materialized"] = _materialize(splits, out, image_index, mode=mode, link=link)
        yaml_path = out / "data.yaml"
        if not yaml_path.exists():                        # ne pas écraser le data.yaml d'un dataset fusionné
            yaml_path.write_text(dataset_yaml_text(out))
        report["data_yaml"] = str(yaml_path)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Construit un dataset YOLO depuis VinDr-Mammo (généralisation détecteur).")
    p.add_argument("--annotations", required=True, help="finding_annotations.csv de VinDr-Mammo.")
    p.add_argument("--images-root", required=True, help="Racine des images VinDr (converties png/jpg, nommées image_id.*).")
    p.add_argument("--out-dir", required=True, help="Dossier de sortie (peut être un dataset CBIS existant → fusion).")
    p.add_argument("--mode", choices=["seg", "detect"], default="seg",
                   help="'seg' (polygone rectangle, fusionne avec CBIS-seg) ou 'detect' (box xywh).")
    p.add_argument("--no-negatives", action="store_true", help="Ne pas inclure les images sans finding (négatifs).")
    p.add_argument("--copy", action="store_true", help="Copier les images au lieu de les lier (symlink par défaut).")
    p.add_argument("--dry-run", action="store_true", help="Résoudre et rapporter sans écrire (CSV seul).")
    args = p.parse_args(argv)

    report = build(args.annotations, args.images_root, args.out_dir, mode=args.mode,
                   include_negatives=not args.no_negatives, link=not args.copy,
                   materialize=not args.dry_run)

    tr, va = report["train"], report["val"]
    print(f"mode : {report['mode']} | classes : {CLASS_NAMES}")
    print(f"  train : {tr['n_images']} images ({tr['n_positive']} pos / {tr['n_negative']} nég), "
          f"findings {tr['n_findings']} — {len(tr['studies'])} étude(s)")
    print(f"  val   : {va['n_images']} images ({va['n_positive']} pos / {va['n_negative']} nég), "
          f"findings {va['n_findings']} — {len(va['studies'])} étude(s)")
    print(f"  findings ignorés (hors mass/calc) : {report['skipped_category']}")

    if report["study_leak"]:
        print(f"ERREUR : fuite d'étude train/val ({len(report['study_leak'])}) — abandon.", file=sys.stderr)
        return 1
    print("  fuite étude : AUCUNE")

    if not args.dry_run:
        m = report["materialized"]
        print(f"  images indexées : {report['n_images_indexed']}")
        print(f"  écrit train : {m['train']} | val : {m['val']}")
        print(f"  data.yaml : {report['data_yaml']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Boucle d'usage typique (fusion CBIS + VinDr) — voir notebooks/ pour l'exécution GPU :
#   1) python -m mlops.datasets.build_yolo_seg_dataset  … --out-dir /data/ifar/yolo_seg
#   2) python -m mlops.datasets.build_vindr_yolo_dataset … --out-dir /data/ifar/yolo_seg --mode seg
#   → un seul data.yaml, images/labels préfixés (cbis) `{split}_{uid}` et (vindr) `{split}_vindr_{id}`.
