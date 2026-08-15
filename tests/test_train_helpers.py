from mlops.train.train_mammo_classifier import set_backbone_trainable


class _Param:
    def __init__(self):
        self.requires_grad = True


class _Backbone:
    def __init__(self, n):
        self._params = [_Param() for _ in range(n)]

    def parameters(self):
        return self._params


class _Model:
    """Modèle factice : `set_backbone_trainable` est duck-typé → testable sans torch."""

    def __init__(self, n=5):
        self.backbone = _Backbone(n)
        self.head = [_Param()]          # la tête ne doit jamais être touchée


def test_freeze_backbone_only():
    m = _Model(5)
    n = set_backbone_trainable(m, False)
    assert n == 5
    assert all(p.requires_grad is False for p in m.backbone.parameters())
    assert m.head[0].requires_grad is True      # la tête reste entraînable


def test_unfreeze_restores_backbone():
    m = _Model(3)
    set_backbone_trainable(m, False)
    n = set_backbone_trainable(m, True)
    assert n == 3
    assert all(p.requires_grad is True for p in m.backbone.parameters())


def test_toggle_is_idempotent():
    m = _Model(2)
    set_backbone_trainable(m, False)
    set_backbone_trainable(m, False)
    assert all(p.requires_grad is False for p in m.backbone.parameters())
