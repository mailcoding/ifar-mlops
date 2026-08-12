import csv
from pathlib import Path

from mlops.datasets.build_vindr_yolo_dataset import (
    _map_category,
    box_to_detect,
    box_to_seg_polygon,
    detect_label_line,
    index_images_by_id,
    resolve,
)

ANN_HEADER = ["study_id", "image_id", "split", "finding_categories",
              "width", "height", "xmin", "ymin", "xmax", "ymax"]


def _write_ann(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ANN_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in ANN_HEADER})


def test_map_category():
    assert _map_category("Mass") == 0
    assert _map_category("['Mass']") == 0
    assert _map_category("Suspicious Calcification") == 1
    assert _map_category("Architectural Distortion") is None
    assert _map_category("No Finding") is None


def test_box_to_detect_normalizes_and_clips():
    # box 20..60 sur largeur 100, 10..30 sur hauteur 50 → centre (0.4,0.4), taille (0.4,0.4)
    assert box_to_detect(20, 10, 60, 30, 100, 50) == [0.4, 0.4, 0.4, 0.4]
    # débordement borné à [0,1]
    xc, _yc, bw, bh = box_to_detect(-10, 0, 120, 60, 100, 50)
    assert 0.0 <= xc <= 1.0 and bw == 1.0 and bh == 1.0


def test_box_to_seg_polygon_is_normalized_rectangle():
    poly = box_to_seg_polygon(20, 10, 60, 30, 100, 50)
    assert poly == [0.2, 0.2, 0.6, 0.2, 0.6, 0.6, 0.2, 0.6]  # 4 coins (x1y1 x2y1 x2y2 x1y2)


def test_detect_label_line_format():
    assert detect_label_line(1, [0.4, 0.4, 0.4, 0.4]) == "1 0.400000 0.400000 0.400000 0.400000"


def test_index_images_by_id(tmp_path):
    (tmp_path / "sA").mkdir()
    (tmp_path / "sA" / "img001.png").write_bytes(b"x")
    (tmp_path / "sB").mkdir()
    (tmp_path / "sB" / "img002.jpg").write_bytes(b"y")
    idx = index_images_by_id(str(tmp_path))
    assert idx["img001"].endswith("img001.png") and idx["img002"].endswith("img002.jpg")


def test_resolve_groups_findings_uses_official_split_and_negatives(tmp_path):
    ann = tmp_path / "finding_annotations.csv"
    _write_ann(ann, [
        # étude S1 : image I1 avec 2 findings (mass + calc) → train
        {"study_id": "S1", "image_id": "I1", "split": "training", "finding_categories": "Mass",
         "width": "100", "height": "80", "xmin": "10", "ymin": "10", "xmax": "40", "ymax": "40"},
        {"study_id": "S1", "image_id": "I1", "split": "training", "finding_categories": "Suspicious Calcification",
         "width": "100", "height": "80", "xmin": "50", "ymin": "50", "xmax": "70", "ymax": "70"},
        # étude S1 : image I2 sans finding → négatif train
        {"study_id": "S1", "image_id": "I2", "split": "training", "finding_categories": "No Finding",
         "width": "100", "height": "80"},
        # étude S2 : image I3 mass → val (test)
        {"study_id": "S2", "image_id": "I3", "split": "test", "finding_categories": "Mass",
         "width": "200", "height": "100", "xmin": "20", "ymin": "20", "xmax": "60", "ymax": "60"},
        # finding non pertinent → ignoré
        {"study_id": "S2", "image_id": "I3", "split": "test", "finding_categories": "Architectural Distortion",
         "width": "200", "height": "100", "xmin": "5", "ymin": "5", "xmax": "9", "ymax": "9"},
    ])
    splits, stats = resolve(str(ann), include_negatives=True)

    assert stats["train"]["n_images"] == 2                       # I1 (positif) + I2 (négatif)
    assert stats["train"]["n_positive"] == 1 and stats["train"]["n_negative"] == 1
    assert stats["train"]["n_findings"] == {"mass": 1, "calcification": 1}
    assert stats["val"]["n_images"] == 1 and stats["val"]["n_findings"] == {"mass": 1, "calcification": 0}
    assert stats["skipped_category"] == 1                         # Architectural Distortion
    # I1 regroupe bien ses deux lésions
    assert len(splits["train"]["I1"]["findings"]) == 2 and splits["train"]["I1"]["negative"] is False
    assert splits["train"]["I2"]["negative"] is True
    # split officiel disjoint par étude → pas de fuite
    assert set(stats["train"]["studies"]).isdisjoint(stats["val"]["studies"])


def test_resolve_excludes_negatives_when_disabled(tmp_path):
    ann = tmp_path / "ann.csv"
    _write_ann(ann, [
        {"study_id": "S1", "image_id": "I1", "split": "training", "finding_categories": "Mass",
         "width": "100", "height": "80", "xmin": "10", "ymin": "10", "xmax": "40", "ymax": "40"},
        {"study_id": "S1", "image_id": "I2", "split": "training", "finding_categories": "No Finding",
         "width": "100", "height": "80"},
    ])
    _splits, stats = resolve(str(ann), include_negatives=False)
    assert stats["train"]["n_images"] == 1 and stats["train"]["n_negative"] == 0
