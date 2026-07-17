from .cbis_ddsm import RoiClassificationDataset, classifier_transforms

# NB : le générateur de manifestes est un module exécutable
# (`python -m mlops.datasets.build_cbis_manifest`) — importer explicitement
# `from mlops.datasets.build_cbis_manifest import build`. Pas de ré-export eager
# ici pour éviter un Runtimerunpy warning au lancement `-m`.
__all__ = ["RoiClassificationDataset", "classifier_transforms"]
