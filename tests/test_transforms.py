import pytest

from mlops.datasets.cbis_ddsm import (
    CLF_INPUT_SIZE,
    DEFAULT_AUGMENT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    resolve_augment,
)

# ── resolve_augment : pur, aucune dépendance torch ──


def test_resolve_augment_returns_defaults_when_empty():
    assert resolve_augment() == DEFAULT_AUGMENT
    assert resolve_augment({}) == DEFAULT_AUGMENT


def test_resolve_augment_overrides_and_ignores_none():
    out = resolve_augment({"rotation": 30, "vflip": None})
    assert out["rotation"] == 30
    assert out["vflip"] == DEFAULT_AUGMENT["vflip"]        # None → on garde le défaut


def test_resolve_augment_rejects_unknown_key():
    # Une faute de frappe désactiverait silencieusement une augmentation crue active.
    with pytest.raises(ValueError, match="inconnues"):
        resolve_augment({"rotaton": 30})


def test_resolve_augment_does_not_mutate_defaults():
    resolve_augment({"rotation": 99})
    assert DEFAULT_AUGMENT["rotation"] == 15


# ── composition des transforms : nécessite torchvision ──


def _names(compose):
    return [type(t).__name__ for t in compose.transforms]


def test_eval_pipeline_is_exactly_the_inference_contract():
    """GARDE-FOU : `train=False` doit rester le pipeline du produit
    (ml-service/app/utils.py::preprocess_for_classifier) — 224 px + normalisation ImageNet,
    AUCUNE augmentation. Toute divergence recrée le décalage entraînement ↔ inférence."""
    pytest.importorskip("torchvision")
    from mlops.datasets.cbis_ddsm import classifier_transforms

    t = classifier_transforms(train=False)
    assert _names(t) == ["Resize", "ToTensor", "Normalize"]

    resize, _, norm = t.transforms
    assert tuple(resize.size) == (CLF_INPUT_SIZE, CLF_INPUT_SIZE)
    assert list(norm.mean) == IMAGENET_MEAN
    assert list(norm.std) == IMAGENET_STD


def test_eval_pipeline_ignores_augment_argument():
    """Même si une config d'augmentation est passée, l'évaluation n'en applique aucune."""
    pytest.importorskip("torchvision")
    from mlops.datasets.cbis_ddsm import classifier_transforms

    assert _names(classifier_transforms(train=False, augment={"rotation": 45})) == [
        "Resize", "ToTensor", "Normalize"]


def test_train_pipeline_applies_requested_augmentations():
    pytest.importorskip("torchvision")
    from mlops.datasets.cbis_ddsm import classifier_transforms

    names = _names(classifier_transforms(train=True))
    # jitter de cadrage + flips + affine + colorimétrie, puis la queue commune
    assert "RandomResizedCrop" in names
    assert "RandomHorizontalFlip" in names and "RandomVerticalFlip" in names
    assert "RandomAffine" in names and "ColorJitter" in names
    assert names[-2:] == ["ToTensor", "Normalize"]


def test_train_pipeline_can_disable_everything():
    """Tout désactivé → on retombe sur le pipeline d'inférence (utile pour isoler un effet)."""
    pytest.importorskip("torchvision")
    from mlops.datasets.cbis_ddsm import classifier_transforms

    off = {"random_resized_crop": None, "hflip": 0, "vflip": 0,
           "rotation": 0, "translate": 0, "brightness": 0, "contrast": 0}
    # resolve_augment ignore les None → on force random_resized_crop via une valeur fausse
    t = classifier_transforms(train=True, augment={**off, "random_resized_crop": []})
    assert _names(t) == ["Resize", "ToTensor", "Normalize"]


def test_train_pipeline_crop_scale_is_honoured():
    pytest.importorskip("torchvision")
    from mlops.datasets.cbis_ddsm import classifier_transforms

    t = classifier_transforms(train=True, augment={"random_resized_crop": [0.5, 0.9]})
    rrc = next(x for x in t.transforms if type(x).__name__ == "RandomResizedCrop")
    assert tuple(rrc.scale) == (0.5, 0.9)
