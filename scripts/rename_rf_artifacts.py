"""One-time migration: rename `rf` artifacts in outputs/ to `hgb`.

Renames file basenames `*_rf*` -> `*_hgb*` and patches the `model_type`
field inside metadata JSONs and the `model` column inside metrics CSVs.
Idempotent: re-running is a no-op once migrated.

Run from project root:
    python -m scripts.rename_rf_artifacts
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUTS

PREFIXES = ("metrics_", "test_metrics_", "elasticity_", "metadata_", "model_")
# Match e.g. metrics_rf.csv, metrics_rf_poisson.csv, model_rf_gamma.joblib
RE_RF = re.compile(r"^(metrics|test_metrics|elasticity|metadata|model)_rf(_[a-z]+)?(\.[a-z]+)$")


def _new_name(old: str) -> str | None:
    m = RE_RF.match(old)
    if not m:
        return None
    head, obj, ext = m.group(1), m.group(2) or "", m.group(3)
    return f"{head}_hgb{obj}{ext}"


def _patch_metadata(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  [skip-patch] {path.name}: {e}")
        return False
    if data.get("model_type") == "rf":
        data["model_type"] = "hgb"
        path.write_text(json.dumps(data, indent=2))
        return True
    return False


def _patch_metrics_csv(path: Path) -> bool:
    text = path.read_text()
    new = re.sub(r"(^|,)rf(,|$)", r"\1hgb\2", text, flags=re.MULTILINE)
    if new != text:
        path.write_text(new)
        return True
    return False


def main() -> int:
    out = Path(OUTPUTS)
    if not out.exists():
        print(f"outputs dir not found: {out}")
        return 1

    renamed: list[tuple[str, str]] = []
    skipped_existing: list[str] = []
    patched_meta: list[str] = []
    patched_metrics: list[str] = []

    for old_path in sorted(out.iterdir()):
        if not old_path.is_file():
            continue
        new_basename = _new_name(old_path.name)
        if not new_basename:
            continue
        new_path = out / new_basename
        if new_path.exists():
            skipped_existing.append(old_path.name)
            continue
        old_path.rename(new_path)
        renamed.append((old_path.name, new_path.name))

    # Patch JSON model_type and CSV model column on the new files
    for new_name in [n for _, n in renamed] + [
        p.name for p in out.iterdir()
        if p.is_file() and p.name.startswith(("metadata_hgb", "metrics_hgb"))
    ]:
        path = out / new_name
        if new_name.startswith("metadata_") and new_name.endswith(".json"):
            if _patch_metadata(path):
                patched_meta.append(new_name)
        elif new_name.startswith("metrics_") and new_name.endswith(".csv"):
            if _patch_metrics_csv(path):
                patched_metrics.append(new_name)

    print(f"Renamed {len(renamed)} files:")
    for old, new in renamed:
        print(f"  {old} -> {new}")
    if skipped_existing:
        print(f"\nSkipped {len(skipped_existing)} (target already exists):")
        for n in skipped_existing:
            print(f"  {n}")
    if patched_meta:
        print(f"\nPatched model_type='rf' -> 'hgb' in {len(patched_meta)} metadata files")
    if patched_metrics:
        print(f"Patched model='rf' -> 'hgb' column in {len(patched_metrics)} metrics CSVs")

    # Optuna warning
    db = out / "optuna.db"
    if db.exists():
        print(
            f"\n[warn] {db.name} still contains old `rf__*` study names. They are inert; "
            f"future `--model hgb ...` runs create fresh `hgb__*` studies."
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
