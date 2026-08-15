import json

from mlops.validation.improvement import (
    DEFAULT_RULE,
    apply_improvement_approval,
    compare_metrics,
    evaluate_improvement,
    main,
)

# Chiffres réels du projet : nouveau modèle (crops) vs modèle en ligne (entraîné sur images entières).
CANDIDATE = {"auc": 0.8522, "sensitivity": 0.8271, "specificity": 0.7030}
INCUMBENT = {"auc": 0.7720, "sensitivity": 0.6300, "specificity": 0.7400}


def _manifest(metrics=None, validated=False, with_gate=True):
    m = {"version": "v0.1.0", "weights_sha256": "a" * 64, "validated": validated,
         "metrics": metrics if metrics is not None else dict(CANDIDATE)}
    if with_gate:
        m["validation"] = {"decision": "no-go", "validated_by": "", "checks": []}
    return m


def _eval(manifest, **over):
    kw = {"incumbent": {"metrics": INCUMBENT}, "approved_by": "Dr Ndiaye (radiologue)",
          "rationale": "Sensibilité +0,20 : cancers manqués 37 % → 17 %. Radiologue dans la boucle.",
          "review_by": "2026-11-15", "same_test_set": True}
    kw.update(over)
    return evaluate_improvement(manifest, **kw)


# ── Comparaison ──


def test_real_case_is_approved_sensitivity_up_specificity_within_tolerance():
    # +0,20 de sensibilité, −0,037 de spécificité (tolérance 0,05) → autorisé.
    r = _eval(_manifest())
    assert r["passed"] is True
    assert r["decision"]["approved"] is True
    sens = next(c for c in r["checks"] if c["check"] == "improve:sensitivity")
    assert sens["delta"] == 0.1971
    spec = next(c for c in r["checks"] if c["check"] == "no_regression:specificity")
    assert spec["passed"] is True and spec["delta"] == -0.037


def test_refused_when_specificity_regression_exceeds_tolerance():
    r = _eval(_manifest({**CANDIDATE, "specificity": 0.60}))     # −0,14, hors tolérance 0,05
    assert r["passed"] is False
    assert not next(c for c in r["checks"] if c["check"] == "no_regression:specificity")["passed"]


def test_refused_when_sensitivity_does_not_improve():
    r = _eval(_manifest({**CANDIDATE, "sensitivity": 0.63}))      # égal au sortant → pas mieux
    assert r["passed"] is False
    assert not next(c for c in r["checks"] if c["check"] == "improve:sensitivity")["passed"]


def test_refused_when_auc_regresses_at_all():
    r = _eval(_manifest({**CANDIDATE, "auc": 0.70}))              # tolérance AUC = 0.0
    assert r["passed"] is False


def test_compare_metrics_fails_closed_on_missing_metric():
    checks = compare_metrics({"sensitivity": 0.9}, {"sensitivity": 0.6}, DEFAULT_RULE)
    spec = next(c for c in checks if c["check"] == "no_regression:specificity")
    assert spec["passed"] is False and spec["delta"] is None


# ── Garde-fous anti-contournement ──


def test_refused_without_same_test_set_attestation():
    """Comparer des métriques issues de splits différents ne prouve rien."""
    r = _eval(_manifest(), same_test_set=False)
    assert r["passed"] is False
    assert not next(c for c in r["checks"] if c["check"] == "same_test_set_attested")["passed"]


def test_refused_when_clinical_gate_never_ran():
    """On ne contourne pas une évaluation absente : la porte doit avoir été exécutée."""
    r = _eval(_manifest(with_gate=False))
    assert r["passed"] is False
    assert not next(c for c in r["checks"] if c["check"] == "clinical_gate_executed")["passed"]


def test_refused_without_named_approver_rationale_or_review_date():
    for field in ("approved_by", "rationale", "review_by"):
        r = _eval(_manifest(), **{field: "   "})
        assert r["passed"] is False, field
        assert not next(c for c in r["checks"] if c["check"] == field)["passed"]


def test_never_sets_validated_true(tmp_path):
    """LA garantie centrale : cette voie ne peut PAS valider cliniquement un modèle."""
    art = tmp_path / "a"
    art.mkdir()
    (art / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")

    r = apply_improvement_approval(
        art, incumbent={"metrics": INCUMBENT}, approved_by="Dr Ndiaye",
        rationale="gain de sensibilité", review_by="2026-11-15", same_test_set=True)
    assert r["passed"] is True

    saved = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert saved["validated"] is False                       # intouché
    assert saved["deployment"]["approved"] is True
    assert saved["deployment"]["clinically_validated"] is False
    assert saved["deployment"]["basis"] == "improvement-over-incumbent"
    assert saved["deployment"]["review_by"] == "2026-11-15"  # borné dans le temps
    assert saved["validation"]["decision"] == "no-go"        # la décision clinique reste visible


def test_refusal_is_recorded_not_silent(tmp_path):
    art = tmp_path / "a"
    art.mkdir()
    (art / "manifest.json").write_text(
        json.dumps(_manifest({**CANDIDATE, "sensitivity": 0.50})), encoding="utf-8")

    r = apply_improvement_approval(
        art, incumbent={"metrics": INCUMBENT}, approved_by="Dr Ndiaye",
        rationale="x", review_by="2026-11-15", same_test_set=True)
    assert r["passed"] is False
    saved = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert saved["deployment"]["approved"] is False          # le refus est tracé


def test_cli_exit_code_and_tolerance_override(tmp_path, capsys):
    art = tmp_path / "a"
    art.mkdir()
    (art / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    inc = json.dumps(INCUMBENT)

    ok = main(["--artifact", str(art), "--incumbent-metrics", inc, "--approved-by", "Dr X",
               "--rationale", "gain sensibilité", "--review-by", "2026-11-15", "--same-test-set"])
    assert ok == 0
    assert "ACCORDÉE" in capsys.readouterr().out

    # Tolérance resserrée → la régression de spécificité (−0,037) devient inacceptable.
    ko = main(["--artifact", str(art), "--incumbent-metrics", inc, "--approved-by", "Dr X",
               "--rationale", "gain sensibilité", "--review-by", "2026-11-15", "--same-test-set",
               "--max-regression", "specificity=0.01", "--dry-run"])
    assert ko == 1
    assert "REFUSÉE" in capsys.readouterr().out


def test_cli_refuses_without_same_test_set_flag(tmp_path):
    art = tmp_path / "a"
    art.mkdir()
    (art / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    rc = main(["--artifact", str(art), "--incumbent-metrics", json.dumps(INCUMBENT),
               "--approved-by", "Dr X", "--rationale", "y", "--review-by", "2026-11-15",
               "--dry-run"])
    assert rc == 1
