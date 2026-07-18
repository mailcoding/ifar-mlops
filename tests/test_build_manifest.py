import csv
import os
from pathlib import Path

from mlops.datasets.build_cbis_manifest import (
    _candidates,
    _norm_label,
    _series_uid,
    _split_by_patient,
    build,
)

CASE_HEADER = [
    "patient_id", "breast density", "left or right breast", "image view",
    "abnormality id", "abnormality type", "calc type", "calc distribution",
    "assessment", "pathology", "subtlety", "image file path",
    "cropped image file path", "ROI mask file path",
]


def _write_case_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CASE_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in CASE_HEADER})


def _row(pid, pathology, view="CC"):
    base = f"Calc-Test_{pid}_LEFT_{view}"
    uid = f"1.3.6.1.4.1.9590.100.1.2.{abs(hash((pid, view))) % (10**30)}"
    return {
        "patient_id": pid,
        "left or right breast": "LEFT",
        "image view": view,
        "abnormality type": "calcification",
        "pathology": pathology,
        "image file path": f"{base}/1.3.6.1.4.1.9590.100.1.2.999/{uid}/000000.dcm",
        "cropped image file path": f"{base}_1/1.3.6.1.4.1.9590.100.1.2.888/{uid}/000000.dcm",
    }


def test_norm_label_maps_benign_without_callback():
    assert _norm_label("MALIGNANT") == "MALIGNANT"
    assert _norm_label("BENIGN") == "BENIGN"
    assert _norm_label("BENIGN_WITHOUT_CALLBACK") == "BENIGN"
    assert _norm_label("") is None


def test_series_uid_extracts_last_uid_segment():
    uid = _series_uid("Folder/1.3.6.1.4.1.9590.100.1.2.111/1.3.6.1.4.1.9590.100.1.2.222/000000.dcm")
    assert uid == "1.3.6.1.4.1.9590.100.1.2.222"


def test_split_by_patient_has_no_leak():
    rows = [{"patient_id": f"P_{i:03d}", "label": "BENIGN", "series_uid": "", "case_path": ""}
            for i in range(10) for _ in range(2)]  # 10 patients, 2 vues chacun
    train, val = _split_by_patient(rows, val_frac=0.3, seed=1)
    tp = {r["patient_id"] for r in train}
    vp = {r["patient_id"] for r in val}
    assert tp and vp
    assert tp.isdisjoint(vp)                      # aucune fuite patient
    assert tp | vp == {f"P_{i:03d}" for i in range(10)}


def test_candidates_strip_duplicated_prefix():
    # image_path préfixé 'CBIS-DDSM/...' + images_root pointant déjà sur .../CBIS-DDSM
    row = {"series_uid": "1.3.6.1.4.1.9590.100.1.2.42", "case_path": ""}
    dicom_map = {"1.3.6.1.4.1.9590.100.1.2.42": "CBIS-DDSM/jpeg/uid/1-1.jpg"}
    cands = _candidates(row, dicom_map, "/data/CBIS-DDSM")
    # doit inclure la variante préfixe-retiré (jpeg/...) sous images_root
    assert "/data/CBIS-DDSM/jpeg/uid/1-1.jpg" in cands
    # et la variante brute avec le préfixe dupliqué (essayée en premier)
    assert "/data/CBIS-DDSM/CBIS-DDSM/jpeg/uid/1-1.jpg" in cands


def test_build_verify_resolves_via_stripped_prefix(tmp_path):
    # Fichier réel à <root>/jpeg/uid/1-1.jpg ; dicom_info dit 'CBIS-DDSM/jpeg/uid/1-1.jpg'.
    root = tmp_path / "CBIS-DDSM"
    (root / "jpeg" / "uid1").mkdir(parents=True)
    (root / "jpeg" / "uid1" / "1-1.jpg").write_bytes(b"x")
    (root / "jpeg" / "uid2").mkdir(parents=True)
    (root / "jpeg" / "uid2" / "1-1.jpg").write_bytes(b"x")

    case = tmp_path / "calc.csv"
    rows = [
        {"patient_id": "P_1", "pathology": "MALIGNANT", "image view": "CC", "left or right breast": "LEFT",
         "abnormality type": "calcification", "cropped image file path": "F_P1/std/1.3.6.1.4.1.9590.100.1.2.1/000000.dcm"},
        {"patient_id": "P_2", "pathology": "BENIGN", "image view": "CC", "left or right breast": "LEFT",
         "abnormality type": "calcification", "cropped image file path": "F_P2/std/1.3.6.1.4.1.9590.100.1.2.2/000000.dcm"},
    ]
    _write_case_csv(case, rows)
    dinfo = tmp_path / "dicom_info.csv"
    with open(dinfo, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SeriesInstanceUID", "SeriesDescription", "image_path"])
        w.writerow(["1.3.6.1.4.1.9590.100.1.2.1", "cropped images", "CBIS-DDSM/jpeg/uid1/1-1.jpg"])
        w.writerow(["1.3.6.1.4.1.9590.100.1.2.2", "cropped images", "CBIS-DDSM/jpeg/uid2/1-1.jpg"])

    rep = build([str(case)], use="cropped", dicom_info=str(dinfo), images_root=str(root),
                val_frac=0.5, seed=1, verify=True)
    s = rep["stats"]
    assert s["n_resolved"] == 2 and s["missing_files"] == 0
    # les chemins retenus pointent bien sur le fichier réel (préfixe retiré)
    allpaths = [r["path"] for r in rep["train"] + rep["val"]]
    assert all(os.path.exists(p) for p in allpaths)


def test_build_end_to_end_no_verify(tmp_path):
    csv_path = tmp_path / "calc_test.csv"
    rows = []
    # 6 patients : 3 malins, 3 bénins (dont un BENIGN_WITHOUT_CALLBACK), 2 vues chacun.
    for i in range(3):
        rows += [_row(f"P_{i:03d}", "MALIGNANT", "CC"), _row(f"P_{i:03d}", "MALIGNANT", "MLO")]
    for i in range(3, 6):
        patho = "BENIGN_WITHOUT_CALLBACK" if i == 5 else "BENIGN"
        rows += [_row(f"P_{i:03d}", patho, "CC"), _row(f"P_{i:03d}", patho, "MLO")]
    _write_case_csv(csv_path, rows)

    report = build([str(csv_path)], use="cropped", val_frac=0.34, seed=3, verify=False)
    s = report["stats"]
    assert s["n_rows"] == 12
    assert s["n_resolved"] == 12                  # --no-verify → chemins conservés tels quels
    assert not s["patient_leak"]
    # 6 patients (3 malins, 3 bénins) → stratifié, val_frac 0.34 → 1 par strate = 2 patients en val
    assert s["n_patients_val"] == 2
    assert s["n_patients_train"] == 4
    total = s["train"]["BENIGN"] + s["train"]["MALIGNANT"] + s["val"]["BENIGN"] + s["val"]["MALIGNANT"]
    assert total == 12
    # BENIGN_WITHOUT_CALLBACK compté comme BENIGN → 6 bénins, 6 malins au total
    assert s["train"]["BENIGN"] + s["val"]["BENIGN"] == 6
    # Split stratifié → chaque classe présente des deux côtés (1 patient × 2 vues = 2 images)
    assert s["val"]["MALIGNANT"] >= 2 and s["val"]["BENIGN"] >= 2
    assert s["train"]["MALIGNANT"] >= 2 and s["train"]["BENIGN"] >= 2
    assert isinstance(s["suggested_class_weights"], list) and len(s["suggested_class_weights"]) == 2
