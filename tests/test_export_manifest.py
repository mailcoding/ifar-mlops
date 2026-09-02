"""
Tests de l'empreinte des poids dans le manifeste.

L'empreinte est ce qui relie un verdict rendu en production à un artefact précis. Un manifeste
qui annonce une empreinte fausse est pire que pas de manifeste du tout : il donne une traçabilité
apparente. D'où la garde testée ici.
"""
import json

import pytest

from mlops.export.export_artifact import sha256_file, write_manifest

COMMON = dict(
    model="ifar-mammo-classifier", framework="timm/efficientnet_b0",
    input_spec={"size": 224}, trained_on={}, metrics={}, threshold=0.40,
)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_empreinte_calculee_quand_les_poids_sont_la(tmp_path):
    (tmp_path / "w.pth").write_bytes(b"des poids")

    path = write_manifest(tmp_path, version="v1.0.0", weights_filename="w.pth", **COMMON)

    assert _read(path)["weights_sha256"] == sha256_file(tmp_path / "w.pth")


def test_empreinte_fournie_reprise_quand_les_poids_sont_ailleurs(tmp_path):
    # Cas réel : les poids déployés vivent sur le Hub, pas sur ce disque ; l'empreinte est
    # relevée sur le pointeur LFS.
    distante = "d4177e86ebc6eb1c408533468067c77d930a4d89ffadf4e6662fc0086b22e78a"

    path = write_manifest(tmp_path, version="unpublished-cbis-ddsm-d4177e86",
                          weights_filename="absent.pth", weights_sha256=distante, **COMMON)

    assert _read(path)["weights_sha256"] == distante


def test_desaccord_entre_fichier_et_empreinte_annoncee(tmp_path):
    (tmp_path / "w.pth").write_bytes(b"des poids")

    with pytest.raises(ValueError, match="Empreinte incohérente"):
        write_manifest(tmp_path, version="v1.0.0", weights_filename="w.pth",
                       weights_sha256="0" * 64, **COMMON)


def test_sans_poids_ni_empreinte_le_champ_reste_vide(tmp_path):
    # Pas d'erreur : un manifeste sans empreinte est incomplet, pas mensonger.
    path = write_manifest(tmp_path, version="v1.0.0", **COMMON)

    assert _read(path)["weights_sha256"] == ""


def test_validated_suit_la_presence_dun_validateur(tmp_path):
    # `validated` ne doit jamais être vrai sans nom : c'est une décision nominative.
    sans = _read(write_manifest(tmp_path / "a", version="v1", **COMMON))
    avec = _read(write_manifest(tmp_path / "b", version="v1", validated_by="Dr X", **COMMON))

    assert sans["validated"] is False and sans["validated_by"] == ""
    assert avec["validated"] is True and avec["validated_by"] == "Dr X"
