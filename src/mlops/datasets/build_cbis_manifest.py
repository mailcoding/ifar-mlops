# src/mlops/datasets/build_cbis_manifest.py
# ─────────────────────────────────────────────
# Génère les manifestes train.csv / val.csv (colonnes `path,label`) consommés par
# RoiClassificationDataset, à partir des CSV de description de cas CBIS-DDSM.
#
#   python -m mlops.datasets.build_cbis_manifest \
#       --case-csv mass_case_description_train_set.csv \
#       --case-csv calc_case_description_train_set.csv \
#       --case-csv mass_case_description_test_set.csv  \
#       --case-csv calc_case_description_test_set.csv  \
#       --dicom-info dicom_info.csv --images-root /data/cbis \
#       --use cropped --val-frac 0.2 --out-dir data
#
# Points clés :
#   • SPLIT PAR PATIENT (anti-fuite) : aucun patient présent à la fois en train et en val
#     (CBIS a plusieurs vues/abnormalités par patient → grouper par patient_id est impératif).
#   • label : MALIGNANT si pathology == MALIGNANT, sinon BENIGN
#     (BENIGN_WITHOUT_CALLBACK → BENIGN).
#   • Résolution du chemin image selon la disposition du dataset (voir --dicom-info / --images-root).
#   • Aucune dépendance lourde (csv/argparse/random/pathlib) → testable en CI sans GPU.
#
# ⚠️ Gouvernance : ces CSV/manifestes ne contiennent que des chemins + labels, jamais de PHI.
# ─────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Iterable

# Colonnes de chemin selon le type d'image à classer.
PATH_COLUMN = {
    "cropped": "cropped image file path",  # ROI recadrée (défaut — proche de l'entrée du classifieur)
    "full": "image file path",             # mammographie entière
}
# SeriesDescription correspondant dans dicom_info.csv (export Kaggle).
SERIES_DESC = {
    "cropped": "cropped images",
    "full": "full mammogram images",
}
LABELS = {"BENIGN", "MALIGNANT"}


def _norm_label(pathology: str) -> str | None:
    """MALIGNANT → MALIGNANT ; BENIGN / BENIGN_WITHOUT_CALLBACK → BENIGN ; sinon None."""
    p = (pathology or "").strip().upper()
    if p == "MALIGNANT":
        return "MALIGNANT"
    if p.startswith("BENIGN"):
        return "BENIGN"
    return None


def _series_uid(dicom_path: str) -> str:
    """SeriesInstanceUID = dernier segment ressemblant à un UID DICOM du chemin."""
    uids = [seg for seg in (dicom_path or "").split("/") if seg.startswith("1.3.6.1.4.1")]
    return uids[-1] if uids else ""


def _read_rows(case_csv: str, use: str) -> list[dict]:
    """Extrait (patient_id, label, series_uid, case_path) d'un CSV de description de cas."""
    col = PATH_COLUMN[use]
    out: list[dict] = []
    with open(case_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [(h or "").strip() for h in (reader.fieldnames or [])]
        for raw in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            label = _norm_label(row.get("pathology", ""))
            pid = row.get("patient_id", "")
            case_path = row.get(col, "")
            if not label or not pid or not case_path:
                continue
            out.append(
                {
                    "patient_id": pid,
                    "label": label,
                    "series_uid": _series_uid(case_path),
                    "case_path": case_path,
                }
            )
    return out


def _load_dicom_info(dicom_info_csv: str, use: str) -> dict[str, str]:
    """Map SeriesInstanceUID → image_path (jpeg), filtré sur le bon SeriesDescription si présent."""
    want_desc = SERIES_DESC[use]
    mapping: dict[str, str] = {}
    with open(dicom_info_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [(h or "").strip() for h in (reader.fieldnames or [])]
        for raw in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            uid = row.get("SeriesInstanceUID", "")
            img = row.get("image_path", "")
            if not uid or not img:
                continue
            desc = (row.get("SeriesDescription", "") or "").lower()
            # Priorité au bon type ; sinon on garde un repli si l'UID n'a pas encore de mapping.
            if desc == want_desc or uid not in mapping:
                mapping[uid] = img
    return mapping


def _candidates(row: dict, dicom_map: dict[str, str] | None, images_root: str | None) -> list[str]:
    """Chemins candidats du fichier image, du plus probable au repli.

    Robuste au piège fréquent du dataset Kaggle CBIS (awsaf49) : la colonne `image_path` de
    `dicom_info.csv` est préfixée `CBIS-DDSM/jpeg/...`, alors que `images_root` pointe déjà sur
    `.../CBIS-DDSM` → on essaie aussi le chemin AVEC LE 1er COMPOSANT RETIRÉ (`jpeg/...`)."""
    if dicom_map is not None:
        img = dicom_map.get(row["series_uid"])
        if not img:
            return []
        rels = [img]
        parts = img.split("/")
        if len(parts) > 1:
            rels.append("/".join(parts[1:]))   # retire le 1er composant (préfixe dataset dupliqué)
    else:
        # Repli : arbre DICOM préservé en jpeg (…/SeriesUID/000000.dcm → …/000000.jpg)
        rel = row["case_path"]
        if rel.lower().endswith(".dcm"):
            rel = rel[:-4] + ".jpg"
        rels = [rel]
        parts = rel.split("/")
        if len(parts) > 1:
            rels.append("/".join(parts[1:]))

    if images_root:
        cands = [os.path.join(images_root, r) for r in rels]
        cands += rels  # au cas où image_path serait déjà absolu / relatif au cwd
    else:
        cands = list(rels)
    # dédoublonnage en conservant l'ordre
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _split_by_patient(rows: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split PAR PATIENT, STRATIFIÉ par label patient : les patients (pas les images) sont
    répartis train/val, et la fraction val est prélevée séparément dans chaque strate
    (malin/bénin) → les deux splits contiennent les deux classes en proportion, et la classe
    minoritaire (maligne) n'atterrit jamais entièrement d'un seul côté.

    Strate d'un patient : MALIGNANT s'il a au moins une lésion maligne, sinon BENIGN."""
    stratum: dict[str, str] = {}
    for r in rows:
        pid = r["patient_id"]
        if r["label"] == "MALIGNANT":
            stratum[pid] = "MALIGNANT"
        else:
            stratum.setdefault(pid, "BENIGN")

    rng = random.Random(seed)
    val_patients: set[str] = set()
    for label in ("MALIGNANT", "BENIGN"):
        pts = sorted(p for p, s in stratum.items() if s == label)
        rng.shuffle(pts)
        n_val = max(1, round(len(pts) * val_frac)) if pts else 0
        val_patients.update(pts[:n_val])

    train = [r for r in rows if r["patient_id"] not in val_patients]
    val = [r for r in rows if r["patient_id"] in val_patients]
    return train, val


def _counts(rows: list[dict]) -> dict[str, int]:
    c = {"BENIGN": 0, "MALIGNANT": 0}
    for r in rows:
        c[r["label"]] += 1
    return c


def _suggest_class_weights(train: list[dict]) -> list[float]:
    """Poids inverse-fréquence [w_benin, w_malin] normalisés (min = 1.0) pour CrossEntropyLoss."""
    c = _counts(train)
    b, m = max(c["BENIGN"], 1), max(c["MALIGNANT"], 1)
    inv = [1.0 / b, 1.0 / m]
    lo = min(inv)
    return [round(x / lo, 3) for x in inv]


def build(
    case_csvs: Iterable[str],
    use: str = "cropped",
    dicom_info: str | None = None,
    images_root: str | None = None,
    val_frac: float = 0.2,
    seed: int = 42,
    verify: bool = True,
) -> dict:
    """Construit les listes train/val résolues. Retourne un rapport (listes + stats)."""
    rows: list[dict] = []
    for c in case_csvs:
        rows.extend(_read_rows(c, use))

    dicom_map = _load_dicom_info(dicom_info, use) if dicom_info else None

    resolved, unresolved, missing = [], 0, 0
    example_missing: list[str] = []
    for r in rows:
        cands = _candidates(r, dicom_map, images_root)
        if not cands:
            unresolved += 1
            continue
        if verify:
            path = next((c for c in cands if Path(c).exists()), None)
            if path is None:
                missing += 1
                if len(example_missing) < 3:
                    example_missing.append(cands[0])
                continue
        else:
            path = cands[0]
        resolved.append({"path": path, "label": r["label"], "patient_id": r["patient_id"]})

    train, val = _split_by_patient(resolved, val_frac, seed)
    leak = {r["patient_id"] for r in train} & {r["patient_id"] for r in val}
    return {
        "train": train,
        "val": val,
        "stats": {
            "n_rows": len(rows),
            "n_resolved": len(resolved),
            "unresolved": unresolved,
            "missing_files": missing,
            "example_missing": example_missing,
            "train": _counts(train),
            "val": _counts(val),
            "n_patients_train": len({r["patient_id"] for r in train}),
            "n_patients_val": len({r["patient_id"] for r in val}),
            "patient_leak": sorted(leak),
            "suggested_class_weights": _suggest_class_weights(train),
        },
    }


def _write_manifest(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        for r in rows:
            w.writerow([r["path"], r["label"]])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Génère train.csv/val.csv (path,label) depuis les CSV CBIS.")
    p.add_argument("--case-csv", action="append", required=True, dest="case_csvs",
                   help="CSV de description de cas CBIS (répéter l'option pour plusieurs).")
    p.add_argument("--use", choices=list(PATH_COLUMN), default="cropped",
                   help="Image à classer : 'cropped' (ROI, défaut) ou 'full' (mammographie entière).")
    p.add_argument("--dicom-info", help="dicom_info.csv (export Kaggle) : mapping SeriesUID→jpeg.")
    p.add_argument("--images-root", help="Racine des images (préfixe des chemins résolus).")
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction de PATIENTS en val (défaut 0.2).")
    p.add_argument("--seed", type=int, default=42, help="Graine du split (défaut 42).")
    p.add_argument("--out-dir", default="data", help="Dossier de sortie des manifestes (défaut data/).")
    p.add_argument("--no-verify", action="store_true",
                   help="Ne pas vérifier l'existence des fichiers (planification à sec).")
    args = p.parse_args(argv)

    report = build(
        args.case_csvs, use=args.use, dicom_info=args.dicom_info, images_root=args.images_root,
        val_frac=args.val_frac, seed=args.seed, verify=not args.no_verify,
    )
    s = report["stats"]

    if s["patient_leak"]:
        print(f"ERREUR : fuite patient détectée ({len(s['patient_leak'])}) — abandon.", file=sys.stderr)
        return 1
    if not report["train"] or not report["val"]:
        print(
            "ERREUR : train ou val vide. Vérifie --dicom-info / --images-root (résolus="
            f"{s['n_resolved']}, non résolus={s['unresolved']}, fichiers manquants={s['missing_files']}).",
            file=sys.stderr,
        )
        if s.get("example_missing"):
            print("  Exemples de chemins essayés (introuvables) :", file=sys.stderr)
            for ex in s["example_missing"]:
                print(f"    - {ex}", file=sys.stderr)
        return 1

    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "val.csv")
    _write_manifest(report["train"], train_path)
    _write_manifest(report["val"], val_path)

    print(f"Écrit : {train_path} ({len(report['train'])} lignes) / {val_path} ({len(report['val'])} lignes)")
    print(f"  lignes CSV lues : {s['n_rows']} — résolues : {s['n_resolved']} "
          f"(non résolues {s['unresolved']}, fichiers manquants {s['missing_files']})")
    print(f"  train : {s['train']} — {s['n_patients_train']} patient(s)")
    print(f"  val   : {s['val']} — {s['n_patients_val']} patient(s)")
    print(f"  fuite patient : {'AUCUNE' if not s['patient_leak'] else s['patient_leak']}")
    print(f"  class_weights suggérés (config) : {s['suggested_class_weights']}  # [bénin, malin]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
