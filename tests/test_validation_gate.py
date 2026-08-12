import json

from mlops.validation.gate import (
    CLASSIFIER_TARGETS,
    apply_gate,
    check_metrics,
    evaluate_gate,
    main,
)

GOOD_METRICS = {"auc": 0.88, "sensitivity": 0.92, "specificity": 0.78}


def _manifest(metrics=None, **over):
    m = {"model": "ifar-mammo-classifier", "version": "v1.0.0",
         "weights_sha256": "a" * 64, "metrics": metrics if metrics is not None else dict(GOOD_METRICS)}
    m.update(over)
    return m


def _gate(manifest, **over):
    kw = {"targets": CLASSIFIER_TARGETS, "validated_by": "Dr Diop (radiologue)",
          "test_set": "CBIS-DDSM test, indépendant"}
    kw.update(over)
    return evaluate_gate(manifest, **kw)


def test_gate_passes_when_all_criteria_met():
    r = _gate(_manifest())
    assert r["passed"] is True
    assert r["decision"]["decision"] == "go"
    assert r["decision"]["targets"] == CLASSIFIER_TARGETS  # critères figés avec la décision


def test_gate_fails_when_sensitivity_below_target():
    # Cas réel du modèle actuel : sensibilité 0,63 → cancers manqués.
    r = _gate(_manifest({"auc": 0.772, "sensitivity": 0.63, "specificity": 0.74}))
    assert r["passed"] is False and r["decision"]["decision"] == "no-go"
    failed = {c["check"] for c in r["checks"] if not c["passed"]}
    assert "metric:sensitivity" in failed and "metric:auc" in failed


def test_gate_fails_closed_on_missing_metric():
    # Métrique absente → ÉCHEC (jamais un succès par défaut).
    r = _gate(_manifest({"auc": 0.9, "sensitivity": 0.95}))  # specificity absente
    assert r["passed"] is False
    spec = next(c for c in r["checks"] if c["check"] == "metric:specificity")
    assert spec["passed"] is False and spec["value"] is None


def test_gate_fails_closed_on_empty_metrics():
    r = _gate(_manifest({}))
    assert r["passed"] is False
    assert all(not c["passed"] for c in r["checks"] if c["check"].startswith("metric:"))


def test_gate_fails_closed_on_non_numeric_metric():
    r = _gate(_manifest({**GOOD_METRICS, "auc": "0.99"}))  # chaîne, pas un nombre
    assert r["passed"] is False


def test_gate_requires_named_validator():
    r = _gate(_manifest(), validated_by="   ")
    assert r["passed"] is False
    assert not next(c for c in r["checks"] if c["check"] == "validated_by")["passed"]


def test_gate_requires_independent_test_set():
    r = _gate(_manifest(), test_set="")
    assert r["passed"] is False
    assert not next(c for c in r["checks"] if c["check"] == "independent_test_set")["passed"]


def test_gate_requires_traceability_version_and_hash():
    r = _gate(_manifest(version="", weights_sha256=""))
    assert r["passed"] is False
    failed = {c["check"] for c in r["checks"] if not c["passed"]}
    assert {"version", "weights_sha256"} <= failed


def test_check_metrics_supports_dotted_paths_for_detector():
    metrics = {"box": {"map50": 0.42, "recall": 0.85}}
    checks = check_metrics(metrics, {"box.map50": 0.35, "box.recall": 0.80})
    assert all(c["passed"] for c in checks)
    assert not check_metrics(metrics, {"box.map50": 0.50})[0]["passed"]


def test_apply_gate_writes_validated_true_and_decision(tmp_path):
    art = tmp_path / "artifact"
    art.mkdir()
    (art / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")

    r = apply_gate(art, validated_by="Dr Diop", test_set="CBIS test indépendant")
    assert r["passed"] is True

    saved = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert saved["validated"] is True
    assert saved["validated_by"] == "Dr Diop"
    assert saved["validation"]["decision"] == "go"
    assert saved["validation"]["test_set"] == "CBIS test indépendant"


def test_apply_gate_no_go_keeps_validated_false_and_traces_failure(tmp_path):
    art = tmp_path / "artifact"
    art.mkdir()
    (art / "manifest.json").write_text(
        json.dumps(_manifest({"auc": 0.60, "sensitivity": 0.50, "specificity": 0.50})), encoding="utf-8")

    r = apply_gate(art, validated_by="Dr Diop", test_set="CBIS test")
    assert r["passed"] is False

    saved = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert saved["validated"] is False
    assert saved["validated_by"] == ""                      # aucune signature sur un no-go
    assert saved["validation"]["decision"] == "no-go"       # l'échec est tracé, pas silencieux


def test_cli_exit_code_reflects_decision(tmp_path, capsys):
    art = tmp_path / "a"
    art.mkdir()
    (art / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")

    ok = main(["--artifact", str(art), "--validated-by", "Dr X", "--test-set", "CBIS test"])
    assert ok == 0

    # Cible surchargée plus exigeante → no-go, code de sortie 1 (bloque une CI).
    ko = main(["--artifact", str(art), "--validated-by", "Dr X", "--test-set", "CBIS test",
               "--target", "auc=0.99", "--dry-run"])
    assert ko == 1
    assert "NO-GO" in capsys.readouterr().out
