# src/mlops/datasets/build_yolo_seg_dataset.py
# ─────────────────────────────────────────────
# Construit un dataset YOLO-seg (images + labels polygone + data.yaml) pour le DÉTECTEUR
# de lésions, à partir des CSV CBIS-DDSM (masse + calcification) et de leurs masques ROI.
#
#   python -m mlops.datasets.build_yolo_seg_dataset \
#       --train-csv mass_case_description_train_set.csv \
#       --train-csv calc_case_description_train_set.csv \
#       --val-csv   mass_case_description_test_set.csv  \
#       --val-csv   calc_case_description_test_set.csv  \
#       --images-root /data/cbis --out-dir /data/ifar/yolo_seg
#
# Points clés :
#   • SPLIT OFFICIEL CBIS (train_set → train, test_set → val) : disjoint par patient
#     (anti-fuite), vérifié dans le rapport.
#   • Une image = un fichier label, regroupant TOUTES ses lésions (évite les faux négatifs).
#   • Image LIÉE en PLEINE RÉSOLUTION (pas de resize destructif) → la recette 1280px profite
#     vraiment de la résolution (décisif pour les microcalcifications). YOLO redimensionne à imgsz.
#   • Polygones dérivés des masques ROI (contour cv2) → format YOLO-seg normalisé.
#   • Classes : 0=mass, 1=calcification (cohérent avec app/segmentation.py CLASS_MAP).
#
# ⚠️ Gouvernance : ni PHI ni images dans git — ce module lit un dataset local et écrit des
#    fichiers locaux (non versionnés). cv2/pillow ne sont requis (extra [train]) que pour la
#    matérialisation (extraction des polygones) → import paresseux ; la résolution est testable sans.
# ─────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Iterable
from pathlib import Path

# type de lésion (colonne dérivée du CSV source) → identifiant de classe YOLO.
LESION_CLASS = {"mass": 0, "calc": 1}
CLASS_NAMES = ["mass", "calcification"]

POLYGON_POINTS = 50     # points max par polygone (sous-échantillonnage)
MIN_CONTOUR_AREA = 20   # aire minimale (pixels) d'un contour retenu


def _norm_class(lesion_type: str) -> int | None:
    """'mass'/'calc' (insensible à la casse, tolère 'calcification') → 0/1 ; sinon None."""
    t = (lesion_type or "").strip().lower()
    if t.startswith("mass"):
        return LESION_CLASS["mass"]
    if t.startswith("calc"):
        return LESION_CLASS["calc"]
    return None


def index_jpegs(images_root: str) -> dict[str, str]:
    """Indexe les JPEG par UID (dossier parent) → chemin du PLUS GROS fichier de l'UID.

    Le plus gros JPEG d'un dossier d'UID de masque est le masque pleine taille (et non la
    petite vignette recadrée) ; pour une image, c'est la mammographie entière."""
    index: dict[str, tuple[int, str]] = {}
    for dirpath, _dirs, files in os.walk(images_root):
        uid = os.path.basename(dirpath)
        for name in files:
            if not name.lower().endswith((".jpg", ".jpeg")):
                continue
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            best = index.get(uid)
            if best is None or size > best[0]:
                index[uid] = (size, path)
    return {uid: path for uid, (_size, path) in index.items()}


def _extract_uid(case_path: str, uid_index: dict[str, str]) -> str | None:
    """UID = 1er segment (en partant de la fin) du chemin CSV présent dans l'index JPEG."""
    for seg in reversed(str(case_path).replace("\\", "/").split("/")):
        if seg in uid_index:
            return seg
    return None


def polygon_to_label_line(class_id: int, polygon: list[float]) -> str:
    """Ligne label YOLO-seg : '<cls> x1 y1 x2 y2 …' (coordonnées normalisées, 6 décimales)."""
    coords = " ".join(f"{v:.6f}" for v in polygon)
    return f"{class_id} {coords}"


def dataset_yaml_text(out_dir: str | Path, names: list[str] = CLASS_NAMES) -> str:
    """Contenu du data.yaml Ultralytics (chemins relatifs images/train, images/val)."""
    import yaml

    spec = {
        "path": str(Path(out_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }
    return yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)


def _read_case_rows(case_csv: str, lesion_type: str) -> list[dict]:
    """(patient_id, class_id, image_path, mask_path) d'un CSV de description de cas CBIS."""
    class_id = _norm_class(lesion_type)
    out: list[dict] = []
    with open(case_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [(h or "").strip() for h in (reader.fieldnames or [])]
        for raw in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            pid = row.get("patient_id", "")
            img = row.get("image file path", "")
            mask = row.get("ROI mask file path", "")
            if class_id is None or not pid or not img or not mask:
                continue
            out.append({"patient_id": pid, "class_id": class_id,
                        "image_path": img, "mask_path": mask})
    return out


def resolve_split(case_csvs: Iterable[tuple[str, str]], uid_index: dict[str, str]) -> tuple[dict, dict]:
    """Résout un split → images regroupées par UID (pur, sans cv2 : CSV + index JPEG).

    case_csvs : itérable de (chemin_csv, lesion_type). uid_index : sortie de index_jpegs.
    Retourne ({img_uid: {img_path, patient_id, lesions:[(class_id, mask_path)]}}, stats)."""
    rows: list[dict] = []
    for path, lesion_type in case_csvs:
        rows.extend(_read_case_rows(path, lesion_type))

    images: dict[str, dict] = {}
    unresolved_img, unresolved_mask, counts = 0, 0, {0: 0, 1: 0}
    for r in rows:
        img_uid = _extract_uid(r["image_path"], uid_index)
        if img_uid is None:
            unresolved_img += 1
            continue
        mask_uid = _extract_uid(r["mask_path"], uid_index)
        if mask_uid is None:
            unresolved_mask += 1
            continue
        entry = images.setdefault(
            img_uid, {"img_path": uid_index[img_uid], "patient_id": r["patient_id"], "lesions": []}
        )
        entry["lesions"].append((r["class_id"], uid_index[mask_uid]))
        counts[r["class_id"]] += 1

    stats = {
        "n_rows": len(rows),
        "n_images": len(images),
        "n_lesions": {"mass": counts[0], "calcification": counts[1]},
        "unresolved_image": unresolved_img,
        "unresolved_mask": unresolved_mask,
        "patients": sorted({e["patient_id"] for e in images.values()}),
    }
    return images, stats


def mask_to_yolo_polygon(mask_path: str, img_path: str,
                         n_points: int = POLYGON_POINTS, min_area: int = MIN_CONTOUR_AREA) -> list[float] | None:
    """Contour du masque ROI → polygone YOLO-seg normalisé [x1,y1,…] ∈ [0,1]. Requiert cv2/numpy."""
    import cv2
    import numpy as np

    img = cv2.imread(img_path)
    if img is None:
        return None
    img_h, img_w = img.shape[:2]

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    epsilon = 0.005 * cv2.arcLength(largest, True)
    pts = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2)
    if len(pts) > n_points:
        pts = pts[np.linspace(0, len(pts) - 1, n_points, dtype=int)]
    if len(pts) < 3:
        return None

    norm = pts.astype(float)
    norm[:, 0] = np.clip(norm[:, 0] / img_w, 0, 1)
    norm[:, 1] = np.clip(norm[:, 1] / img_h, 0, 1)
    return norm.flatten().tolist()


def _materialize(images: dict, split: str, out_dir: Path, *,
                 n_points: int, min_area: int, link: bool) -> dict:
    """Écrit images (liées, pleine résolution) + labels polygone d'un split. Requiert cv2."""
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written, skipped_no_polygon = 0, 0
    for uid, entry in images.items():
        lines = []
        for class_id, mask_path in entry["lesions"]:
            poly = mask_to_yolo_polygon(mask_path, entry["img_path"], n_points, min_area)
            if poly and len(poly) >= 6:  # ≥ 3 points
                lines.append(polygon_to_label_line(class_id, poly))
        if not lines:
            skipped_no_polygon += 1
            continue

        name = f"{split}_{uid}"
        dst = img_dir / f"{name}.jpg"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        src = os.path.abspath(entry["img_path"])
        if link:
            os.symlink(src, dst)          # pleine résolution préservée (pas de resize destructif)
        else:
            import shutil
            shutil.copy(src, dst)
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
        written += 1

    return {"written": written, "skipped_no_polygon": skipped_no_polygon}


def build(train_case_csvs: Iterable[tuple[str, str]], val_case_csvs: Iterable[tuple[str, str]],
          images_root: str, out_dir: str, *, n_points: int = POLYGON_POINTS,
          min_area: int = MIN_CONTOUR_AREA, link: bool = True, materialize: bool = True) -> dict:
    """Construit le dataset YOLO-seg complet (train + val). Retourne un rapport (stats + fuite)."""
    uid_index = index_jpegs(images_root)
    train_images, train_stats = resolve_split(train_case_csvs, uid_index)
    val_images, val_stats = resolve_split(val_case_csvs, uid_index)

    leak = sorted(set(train_stats["patients"]) & set(val_stats["patients"]))
    report = {"train": train_stats, "val": val_stats, "patient_leak": leak,
              "n_jpeg_indexed": len(uid_index)}

    if materialize:
        out = Path(out_dir)
        report["train"]["materialized"] = _materialize(
            train_images, "train", out, n_points=n_points, min_area=min_area, link=link)
        report["val"]["materialized"] = _materialize(
            val_images, "val", out, n_points=n_points, min_area=min_area, link=link)
        (out / "data.yaml").write_text(dataset_yaml_text(out))
        report["data_yaml"] = str(out / "data.yaml")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Construit un dataset YOLO-seg depuis les CSV CBIS + masques ROI.")
    p.add_argument("--train-csv", action="append", required=True, dest="train_csvs",
                   help="CSV de cas pour le split TRAIN (répéter ; type déduit du nom mass/calc).")
    p.add_argument("--val-csv", action="append", required=True, dest="val_csvs",
                   help="CSV de cas pour le split VAL (répéter ; type déduit du nom mass/calc).")
    p.add_argument("--images-root", required=True, help="Racine des JPEG CBIS (arbre à indexer).")
    p.add_argument("--out-dir", required=True, help="Dossier de sortie du dataset YOLO-seg.")
    p.add_argument("--copy", action="store_true", help="Copier les images au lieu de les lier (symlink par défaut).")
    p.add_argument("--polygon-points", type=int, default=POLYGON_POINTS)
    p.add_argument("--min-area", type=int, default=MIN_CONTOUR_AREA)
    p.add_argument("--dry-run", action="store_true",
                   help="Résoudre et rapporter sans écrire (ne nécessite pas cv2).")
    args = p.parse_args(argv)

    def _typed(paths: list[str]) -> list[tuple[str, str]]:
        # Type de lésion déduit du nom de fichier : 'calc*' → calc, sinon mass.
        return [(pp, "calc" if os.path.basename(pp).lower().startswith("calc") else "mass") for pp in paths]

    report = build(_typed(args.train_csvs), _typed(args.val_csvs), args.images_root, args.out_dir,
                   n_points=args.polygon_points, min_area=args.min_area,
                   link=not args.copy, materialize=not args.dry_run)

    tr, va = report["train"], report["val"]
    print(f"JPEG indexés : {report['n_jpeg_indexed']}")
    print(f"  train : {tr['n_images']} images, lésions {tr['n_lesions']} "
          f"(non résolues img {tr['unresolved_image']} / masque {tr['unresolved_mask']}) "
          f"— {len(tr['patients'])} patient(s)")
    print(f"  val   : {va['n_images']} images, lésions {va['n_lesions']} "
          f"(non résolues img {va['unresolved_image']} / masque {va['unresolved_mask']}) "
          f"— {len(va['patients'])} patient(s)")

    if report["patient_leak"]:
        print(f"ERREUR : fuite patient train/val ({len(report['patient_leak'])}) — abandon.", file=sys.stderr)
        return 1
    print("  fuite patient : AUCUNE")

    if not args.dry_run:
        print(f"  écrit train : {tr['materialized']} | val : {va['materialized']}")
        print(f"  data.yaml : {report['data_yaml']}")
        if tr["materialized"]["written"] == 0 or va["materialized"]["written"] == 0:
            print("ERREUR : split train ou val vide après matérialisation.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
