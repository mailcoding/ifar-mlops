import csv
from pathlib import Path

from mlops.datasets.build_yolo_seg_dataset import (
    _norm_class,
    dataset_yaml_text,
    index_jpegs,
    polygon_to_label_line,
    resolve_split,
)

CASE_HEADER = ["patient_id", "pathology", "image file path", "ROI mask file path"]


def _write_case_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CASE_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in CASE_HEADER})


def test_norm_class_maps_types():
    assert _norm_class("mass") == 0
    assert _norm_class("MASS") == 0
    assert _norm_class("calc") == 1
    assert _norm_class("calcification") == 1
    assert _norm_class("") is None


def test_polygon_to_label_line_format():
    line = polygon_to_label_line(1, [0.1, 0.2, 0.3, 0.4])
    assert line == "1 0.100000 0.200000 0.300000 0.400000"


def test_dataset_yaml_text_has_paths_and_classes():
    txt = dataset_yaml_text("/data/ifar/yolo_seg")
    assert "images/train" in txt and "images/val" in txt
    assert "nc: 2" in txt
    assert "mass" in txt and "calcification" in txt


def test_index_jpegs_keeps_largest_per_uid(tmp_path):
    # Deux JPEG sous le même UID → on garde le plus gros (masque pleine taille).
    uiddir = tmp_path / "series" / "1.3.6.uidA"
    uiddir.mkdir(parents=True)
    (uiddir / "small.jpg").write_bytes(b"x" * 10)
    (uiddir / "big.jpg").write_bytes(b"x" * 500)
    other = tmp_path / "series" / "1.3.6.uidB"
    other.mkdir(parents=True)
    (other / "only.jpg").write_bytes(b"y" * 20)

    idx = index_jpegs(str(tmp_path))
    assert idx["1.3.6.uidA"].endswith("big.jpg")
    assert idx["1.3.6.uidB"].endswith("only.jpg")


def test_resolve_split_groups_lesions_and_resolves_paths(tmp_path):
    # Arbre JPEG : une image + deux masques (deux lésions sur la même mammographie).
    for uid, fname in [("img1", "1-1.jpg"), ("maskA", "1-1.jpg"), ("maskB", "1-1.jpg")]:
        d = tmp_path / uid
        d.mkdir()
        (d / fname).write_bytes(b"z" * 100)
    uid_index = index_jpegs(str(tmp_path))

    case = tmp_path / "mass.csv"
    _write_case_csv(case, [
        {"patient_id": "P_1", "pathology": "MALIGNANT",
         "image file path": "Mass/x/img1/000000.dcm", "ROI mask file path": "Mass/x/maskA/000000.dcm"},
        {"patient_id": "P_1", "pathology": "BENIGN",
         "image file path": "Mass/x/img1/000000.dcm", "ROI mask file path": "Mass/x/maskB/000000.dcm"},
    ])

    images, stats = resolve_split([(str(case), "mass")], uid_index)
    assert stats["n_images"] == 1                     # une seule mammographie
    assert stats["n_lesions"] == {"mass": 2, "calcification": 0}
    assert stats["unresolved_image"] == 0 and stats["unresolved_mask"] == 0
    assert stats["patients"] == ["P_1"]
    # les deux lésions sont regroupées sous l'UID image, masques résolus.
    entry = images["img1"]
    assert entry["img_path"].endswith("1-1.jpg") and len(entry["lesions"]) == 2


def test_resolve_split_counts_unresolved(tmp_path):
    uid_index = index_jpegs(str(tmp_path))  # index vide → tout non résolu
    case = tmp_path / "calc.csv"
    _write_case_csv(case, [
        {"patient_id": "P_9", "pathology": "MALIGNANT",
         "image file path": "Calc/x/imgX/000000.dcm", "ROI mask file path": "Calc/x/maskX/000000.dcm"},
    ])
    images, stats = resolve_split([(str(case), "calc")], uid_index)
    assert images == {} and stats["n_images"] == 0
    assert stats["unresolved_image"] == 1
