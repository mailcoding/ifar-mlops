# src/mlops/validation/gate.py
# ─────────────────────────────────────────────
# PORTE DE VALIDATION CLINIQUE (SaMD / IEC 62304 / ISO 14971).
#
# Seul point autorisé à passer un artefact à `validated: true` dans son manifest.json.
# Tant que la porte n'est pas franchie, le produit voit `validated: false` (carte ops) et
# le déploiement refuse l'artefact.
#
# Principes (cf. GOVERNANCE.md « Porte de validation clinique » et
# ifar/docs/SaMD_CADRE_REGLEMENTAIRE.md §4 « Plan de validation clinique ») :
#   • FAIL CLOSED : une métrique absente ou illisible = ÉCHEC (jamais un succès par défaut).
#   • Critères d'acceptation définis AVANT (§4.6) : les cibles sont enregistrées dans la
#     décision, avec l'artefact — pas ajustées après coup pour faire passer un modèle.
#   • Décision go/no-go DOCUMENTÉE et NOMINATIVE : un humain responsable signe (§4).
#   • Jeu de test INDÉPENDANT déclaré (§4.1) — on ne valide pas sur les données d'entraînement.
#   • Traçabilité : version + hash des poids exigés (lien artefact ↔ décision).
#
# ⚠️ Cette porte est une barrière PROCÉDURALE et TECHNIQUE : elle vérifie que les preuves
#    existent et atteignent les seuils. Elle ne remplace PAS l'étude clinique elle-même
#    (rétrospective/prospective, représentativité, analyse des faux négatifs, sous-groupes).
# ─────────────────────────────────────────────

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Cibles CLASSIFIEUR — documentées dans le projet (configs/mammo_classifier.yaml, README, notebooks).
# En oncologie la SENSIBILITÉ prime : un faux négatif = cancer manqué.
CLASSIFIER_TARGETS = {"auc": 0.85, "sensitivity": 0.90, "specificity": 0.75}

# Cibles DÉTECTEUR — objectifs TECHNIQUES internes (le détecteur ne pose pas de diagnostic ; il
# localise). Le recall prime : une lésion non détectée n'est jamais classée.
# ⚠️ Provisoires : les critères d'acceptation CLINIQUES doivent être fixés par la validation
#    clinique AVANT l'étude (§4.6) et passés explicitement via `targets`.
DETECTOR_TARGETS = {"box.map50": 0.35, "box.recall": 0.80}


def _get_path(data: dict, dotted: str):
    """Lit une valeur imbriquée via un chemin pointé ('box.recall'). None si absente."""
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check_metrics(metrics: dict, targets: dict) -> list[dict]:
    """Compare les métriques aux cibles. Une métrique absente/non numérique ÉCHOUE (fail closed)."""
    checks: list[dict] = []
    for name, minimum in sorted(targets.items()):
        value = _get_path(metrics or {}, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            checks.append({"check": f"metric:{name}", "passed": False, "value": value,
                           "required": minimum, "detail": "métrique absente ou non numérique"})
            continue
        passed = float(value) >= float(minimum)
        checks.append({"check": f"metric:{name}", "passed": passed, "value": float(value),
                       "required": minimum,
                       "detail": "" if passed else f"{value} < {minimum} (cible non atteinte)"})
    return checks


def evaluate_gate(manifest: dict, *, targets: dict, validated_by: str,
                  test_set: str, notes: str = "") -> dict:
    """Évalue la porte sur un manifeste (FONCTION PURE — n'écrit rien).

    Retourne {'passed': bool, 'checks': [...], 'decision': {...}}. `decision` est le
    procès-verbal go/no-go à archiver avec l'artefact."""
    checks = check_metrics(manifest.get("metrics", {}), targets)

    # Responsabilité humaine : une décision clinique doit être signée.
    who = (validated_by or "").strip()
    checks.append({"check": "validated_by", "passed": bool(who), "value": who or None,
                   "required": "identité du validateur",
                   "detail": "" if who else "décision go/no-go non nominative"})

    # §4.1 : la performance doit être établie sur un jeu de test INDÉPENDANT.
    ts = (test_set or "").strip()
    checks.append({"check": "independent_test_set", "passed": bool(ts), "value": ts or None,
                   "required": "jeu de test indépendant déclaré",
                   "detail": "" if ts else "aucun jeu de test indépendant déclaré"})

    # Traçabilité artefact ↔ décision.
    version = (manifest.get("version") or "").strip()
    checks.append({"check": "version", "passed": bool(version), "value": version or None,
                   "required": "version de l'artefact",
                   "detail": "" if version else "version manquante"})
    sha = (manifest.get("weights_sha256") or "").strip()
    checks.append({"check": "weights_sha256", "passed": bool(sha), "value": sha[:12] or None,
                   "required": "hash des poids",
                   "detail": "" if sha else "hash des poids manquant (traçabilité)"})

    passed = all(c["passed"] for c in checks)
    decision = {
        "decision": "go" if passed else "no-go",
        "validated_by": who,
        "date": datetime.now(timezone.utc).isoformat(),
        "targets": dict(targets),          # critères figés AVEC la décision (anti-ajustement a posteriori)
        "test_set": ts,
        "checks": checks,
        "notes": notes,
        "framework": "SaMD / IEC 62304 / ISO 14971",
    }
    return {"passed": passed, "checks": checks, "decision": decision}


def apply_gate(artifact_dir: str | Path, *, validated_by: str, test_set: str,
               targets: dict | None = None, profile: str = "classifier",
               notes: str = "") -> dict:
    """Applique la porte au `manifest.json` d'un artefact et l'ÉCRIT.

    `validated: true` UNIQUEMENT si tous les contrôles passent ; sinon le manifeste est
    marqué `validated: false` avec la décision no-go (l'échec est tracé, pas silencieux).
    Retourne le résultat d'`evaluate_gate`."""
    path = Path(artifact_dir)
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    if targets is None:
        targets = DETECTOR_TARGETS if profile == "detector" else CLASSIFIER_TARGETS

    result = evaluate_gate(manifest, targets=targets, validated_by=validated_by,
                           test_set=test_set, notes=notes)

    manifest["validated"] = result["passed"]
    manifest["validated_by"] = validated_by if result["passed"] else ""
    manifest["validation"] = result["decision"]
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    result["manifest"] = str(path)
    return result


def summarize(result: dict) -> str:
    """Rendu texte du procès-verbal (log CI / console)."""
    lines = [f"PORTE DE VALIDATION CLINIQUE — décision : {result['decision']['decision'].upper()}"]
    for c in result["checks"]:
        mark = "✔" if c["passed"] else "✘"
        detail = f"  ({c['detail']})" if c["detail"] else ""
        lines.append(f"  {mark} {c['check']}: {c['value']} [requis ≥ {c['required']}]{detail}")
    if not result["passed"]:
        lines.append("→ Artefact NON validé : publication/déploiement en production interdits.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Porte de validation clinique SaMD d'un artefact modèle.")
    p.add_argument("--artifact", required=True, help="Dossier de l'artefact (ou chemin du manifest.json).")
    p.add_argument("--validated-by", required=True,
                   help="Identité du validateur clinique responsable (décision nominative).")
    p.add_argument("--test-set", required=True,
                   help="Description du jeu de test INDÉPENDANT ayant produit les métriques.")
    p.add_argument("--profile", choices=["classifier", "detector"], default="classifier",
                   help="Profil de cibles par défaut (défaut : classifier).")
    p.add_argument("--target", action="append", default=[], metavar="NOM=VALEUR",
                   help="Surcharge/ajout de cible (ex. --target auc=0.90). Répétable.")
    p.add_argument("--notes", default="", help="Notes de la décision (limites, sous-groupes…).")
    p.add_argument("--dry-run", action="store_true", help="Évaluer sans écrire le manifeste.")
    args = p.parse_args(argv)

    targets = dict(DETECTOR_TARGETS if args.profile == "detector" else CLASSIFIER_TARGETS)
    for item in args.target:
        if "=" not in item:
            print(f"ERREUR : --target attend NOM=VALEUR (reçu : {item})")
            return 2
        name, value = item.split("=", 1)
        targets[name.strip()] = float(value)

    if args.dry_run:
        manifest_path = Path(args.artifact)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        result = evaluate_gate(json.loads(manifest_path.read_text(encoding="utf-8")),
                               targets=targets, validated_by=args.validated_by,
                               test_set=args.test_set, notes=args.notes)
    else:
        result = apply_gate(args.artifact, validated_by=args.validated_by, test_set=args.test_set,
                            targets=targets, notes=args.notes)

    print(summarize(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
