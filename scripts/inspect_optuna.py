"""Print a one-row-per-study snapshot of outputs/optuna.db.

No args. For each study prints:
    study_name | n | COMP | PRUN | FAIL | RUN | best | first_trial | last_trial | wall

Usage (from project root):
    python -m scripts.inspect_optuna
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna

from src.config import OUTPUTS


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds >= 3600:
        return f"{seconds/3600:.2f}h"
    return f"{seconds/60:.1f}m"


def main() -> None:
    db_path = OUTPUTS / "optuna.db"
    if not db_path.exists():
        raise SystemExit(f"no optuna db at {db_path}")
    storage = f"sqlite:///{db_path}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    summaries = optuna.get_all_study_summaries(storage)

    rows = []
    for s in summaries:
        study = optuna.load_study(study_name=s.study_name, storage=storage)
        trials = study.trials
        states = Counter(t.state.name for t in trials)
        starts = [t.datetime_start for t in trials if t.datetime_start]
        ends   = [t.datetime_complete for t in trials if t.datetime_complete]
        first = min(starts) if starts else None
        last  = max(ends) if ends else (max(starts) if starts else None)
        dur = (last - first).total_seconds() if first and last else None
        try:
            best = study.best_value
        except Exception:
            best = None
        rows.append((first, s.study_name, len(trials), states, best, first, last, dur))

    rows.sort(key=lambda r: (r[0] is None, r[0]))

    hdr = (f"{'study_name':<38} {'n':>4} {'COMP':>5} {'PRUN':>5} {'FAIL':>5} "
           f"{'RUN':>4} {'best':>12}  {'first_trial':<19}  {'last_trial':<19}  {'wall':>9}")
    print(f"{len(summaries)} studies in {db_path}")
    print(hdr)
    print("-" * len(hdr))
    for _, name, n, states, best, first, last, dur in rows:
        best_s = "-" if best is None else f"{best:.4f}"
        first_s = "-" if not first else first.strftime("%Y-%m-%d %H:%M:%S")
        last_s  = "-" if not last  else last.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{name:<38} {n:>4} {states.get('COMPLETE',0):>5} "
              f"{states.get('PRUNED',0):>5} {states.get('FAIL',0):>5} "
              f"{states.get('RUNNING',0):>4} {best_s:>12}  "
              f"{first_s:<19}  {last_s:<19}  {_fmt_dur(dur):>9}")


if __name__ == "__main__":
    main()
