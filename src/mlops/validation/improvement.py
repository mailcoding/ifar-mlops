# src/mlops/validation/improvement.py
# ─────────────────────────────────────────────
# AUTORISATION DE DÉPLOIEMENT « AMÉLIORATION » — voie distincte de la validation clinique.
#
# Problème traité : le modèle actuellement en production n'a lui non plus jamais passé de porte.
# Refuser un candidat MEILLEUR au seul motif qu'il n'atteint pas les cibles absolues revient à
# PRÉSERVER ACTIVEMENT le pire modèle — et donc à exposer les patientes au plus mauvais des deux.
# Cette voie répond à une question différente de la porte clinique :
#     porte clinique  → « ce modèle est-il acceptable dans l'absolu ? »   (validated)
#     cette voie      → « ce modèle est-il meilleur que celui en ligne ? » (deployment.approved)
#
# ⚠️ CE N'EST PAS UNE PORTE DÉROBÉE. Garanties par construction :
#   • `validated` n'est JAMAIS modifié ici — seule `mlops.validation.gate` peut le passer à true.
#   • La porte clinique doit avoir été EXÉCUTÉE au préalable (bloc `validation` présent) : on
#     n'autorise pas un déploiement sans s'être d'abord mesuré aux critères absolus.
#   • La comparaison exige une ATTESTATION de jeu de test identique — comparer des métriques
#     issues de splits différents ne prouve rien (cf. mlops.eval.classifier_eval).
#   • Décision NOMINATIVE, JUSTIFIÉE et DATÉE, avec une DATE DE REVUE obligatoire : un déploiement
#     « temporairement meilleur » ne doit pas devenir un état permanent non validé.
#   • FAIL CLOSED : toute métrique manquante ou attestation absente = refus.
# ─────────────────────────────────────────────

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Règle par défaut. En oncologie la SENSIBILITÉ prime (un faux négatif = cancer manqué) : c'est
# elle qui doit progresser. Une régression bornée est tolérée sur les autres métriques, mais elle
# doit être ASSUMÉE explicitement — l'arbitraire est rendu visible, pas caché.
DEFAULT_RULE = {
    "must_improve": ["sensitivity"],
    "max_regression": {"auc": 0.0, "specificity": 0.05},
}


def compare_metrics(candidate: dict, incumbent: dict, rule: dict | None = None) -> list[dict]:
    """Compare candidat vs modèle en ligne selon `rule`. Métrique absente = ÉCHEC (fail closed)."""
    rule = rule or DEFAULT_RULE
    checks: list[dict] = []

    def _num(d, name):
        v = (d or {}).get(name)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    for name in rule.get("must_improve", []):
        c, i = _num(candidate, name), _num(incumbent, name)
        if c is None or i is None:
            checks.append({"check": f"improve:{name}", "passed": False, "candidate": c,
                           "incumbent": i, "delta": None,
                           "detail": "métrique absente chez le candidat ou le sortant"})
            continue
        delta = round(c - i, 4)
        checks.append({"check": f"improve:{name}", "passed": c > i, "candidate": c,
                       "incumbent": i, "delta": delta,
                       "detail": "" if c > i else f"pas d'amélioration ({c} ≤ {i})"})

    for name, tol in sorted((rule.get("max_regression") or {}).items()):
        c, i = _num(candidate, name), _num(incumbent, name)
        if c is None or i is None:
            checks.append({"check": f"no_regression:{name}", "passed": False, "candidate": c,
                           "incumbent": i, "delta": None,
                           "detail": "métrique absente chez le candidat ou le sortant"})
            continue
        delta = round(c - i, 4)
        ok = delta >= -float(tol)
        checks.append({"check": f"no_regression:{name}", "passed": ok, "candidate": c,
                       "incumbent": i, "delta": delta, "tolerance": float(tol),
                       "detail": "" if ok else f"régression {delta} au-delà de la tolérance -{tol}"})
    return checks


def evaluate_improvement(manifest: dict, *, incumbent: dict, approved_by: str, rationale: str,
                         review_by: str, same_test_set: bool = False,
                         rule: dict | None = None) -> dict:
    """Évalue la voie « amélioration » (FONCTION PURE — n'écrit rien)."""
    rule = rule or DEFAULT_RULE
    checks = compare_metrics(manifest.get("metrics", {}), incumbent.get("metrics", incumbent), rule)

    # La porte clinique doit avoir été exécutée : on ne contourne pas une évaluation absente.
    gate_run = isinstance(manifest.get("validation"), dict)
    checks.append({"check": "clinical_gate_executed", "passed": gate_run,
                   "detail": "" if gate_run else
                   "porte clinique jamais exécutée sur cet artefact "
                   "(lancer d'abord python -m mlops.validation.gate)"})

    # Sans même jeu de test, la comparaison ne prouve rien.
    checks.append({"check": "same_test_set_attested", "passed": bool(same_test_set),
                   "detail": "" if same_test_set else
                   "comparaison non attestée sur le MÊME jeu de test — métriques non comparables "
                   "(cf. mlops.eval.classifier_eval.evaluate_classifier)"})

    who = (approved_by or "").strip()
    checks.append({"check": "approved_by", "passed": bool(who),
                   "detail": "" if who else "autorisation non nominative"})

    why = (rationale or "").strip()
    checks.append({"check": "rationale", "passed": bool(why),
                   "detail": "" if why else "arbitrage clinique non justifié par écrit"})

    when = (review_by or "").strip()
    checks.append({"check": "review_by", "passed": bool(when),
                   "detail": "" if when else
                   "date de revue absente — un déploiement non validé doit être borné dans le temps"})

    passed = all(c["passed"] for c in checks)
    decision = {
        "approved": passed,
        "basis": "improvement-over-incumbent",
        # Rappel explicite : cette voie ne vaut PAS validation clinique.
        "clinically_validated": bool(manifest.get("validated", False)),
        "approved_by": who,
        "date": datetime.now(timezone.utc).isoformat(),
        "incumbent": incumbent,
        "rule": rule,
        "same_test_set": bool(same_test_set),
        "comparison": checks,
        "rationale": why,
        "review_by": when,
        "framework": "SaMD / IEC 62304 / ISO 14971 — déploiement d'amélioration, hors validation absolue",
    }
    return {"passed": passed, "checks": checks, "decision": decision}


def apply_improvement_approval(artifact_dir: str | Path, **kwargs) -> dict:
    """Écrit la décision dans `manifest.json` (bloc `deployment`). NE TOUCHE JAMAIS `validated`."""
    path = Path(artifact_dir)
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    result = evaluate_improvement(manifest, **kwargs)
    manifest["deployment"] = result["decision"]        # `validated` reste intouché, par conception
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    result["manifest"] = str(path)
    return result


def summarize(result: dict) -> str:
    """Procès-verbal texte de l'autorisation d'amélioration."""
    d = result["decision"]
    lines = [f"AUTORISATION DE DÉPLOIEMENT « AMÉLIORATION » — {'ACCORDÉE' if d['approved'] else 'REFUSÉE'}"]
    for c in result["checks"]:
        mark = "✔" if c["passed"] else "✘"
        if "candidate" in c:
            lines.append(f"  {mark} {c['check']}: {c['candidate']} vs {c['incumbent']} "
                         f"(Δ {c.get('delta')}){'  — ' + c['detail'] if c['detail'] else ''}")
        else:
            lines.append(f"  {mark} {c['check']}{'  — ' + c['detail'] if c['detail'] else ''}")
    if d["approved"]:
        lines.append(f"→ Déployable comme AMÉLIORATION. NON validé cliniquement "
                     f"(validated={d['clinically_validated']}). Revue avant le {d['review_by']}.")
    else:
        lines.append("→ Déploiement refusé.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Autorise le déploiement d'un modèle MEILLEUR que celui en ligne, "
                    "sans valoir validation clinique absolue.")
    p.add_argument("--artifact", required=True, help="Dossier de l'artefact (ou manifest.json).")
    p.add_argument("--incumbent-metrics", required=True,
                   help="JSON des métriques du modèle EN LIGNE, mesurées sur le MÊME jeu de test "
                        "(ex. '{\"auc\":0.77,\"sensitivity\":0.63,\"specificity\":0.74}').")
    p.add_argument("--approved-by", required=True, help="Responsable clinique autorisant le déploiement.")
    p.add_argument("--rationale", required=True, help="Justification de l'arbitrage (écrite).")
    p.add_argument("--review-by", required=True, help="Date de revue obligatoire (AAAA-MM-JJ).")
    p.add_argument("--same-test-set", action="store_true",
                   help="Atteste que les deux modèles ont été évalués sur le MÊME jeu de test.")
    p.add_argument("--max-regression", action="append", default=[], metavar="NOM=VALEUR",
                   help="Tolérance de régression (ex. --max-regression specificity=0.05). Répétable.")
    p.add_argument("--dry-run", action="store_true", help="Évaluer sans écrire.")
    args = p.parse_args(argv)

    rule = {"must_improve": list(DEFAULT_RULE["must_improve"]),
            "max_regression": dict(DEFAULT_RULE["max_regression"])}
    for item in args.max_regression:
        if "=" not in item:
            print(f"ERREUR : --max-regression attend NOM=VALEUR (reçu : {item})")
            return 2
        name, value = item.split("=", 1)
        rule["max_regression"][name.strip()] = float(value)

    incumbent = json.loads(args.incumbent_metrics)
    kwargs = {"incumbent": {"metrics": incumbent}, "approved_by": args.approved_by,
              "rationale": args.rationale, "review_by": args.review_by,
              "same_test_set": args.same_test_set, "rule": rule}

    if args.dry_run:
        mpath = Path(args.artifact)
        if mpath.is_dir():
            mpath = mpath / "manifest.json"
        result = evaluate_improvement(json.loads(mpath.read_text(encoding="utf-8")), **kwargs)
    else:
        result = apply_improvement_approval(args.artifact, **kwargs)

    print(summarize(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
