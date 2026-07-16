import csv
from pathlib import Path

from mlops.datasets.build_cbis_manifest import (
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
