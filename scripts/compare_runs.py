"""4-layer comparison of all tuning runs + baselines; picks the champion.

L1 CV leaderboard       -> outputs/leaderboard_cv.csv
L2 Friedman / Nemenyi   -> outputs/friedman.json, nemenyi.csv, cd_plot.png
L3 Sealed-test rank     -> outputs/leaderboard_sealed.csv
L4 Elasticity sanity    -> outputs/elasticity_scorecard.csv
Composite champion card -> outputs/champion_card.json

Champion must pass 4 gates:
  1. CV metric rank <= top_k
  2. Nemenyi vs seasonal_naive p < 0.05 (ML significantly better than baseline)
  3. Sealed-test metric rank <= top_k
  4. sign_test_pass AND share_in_soft_drink_range > 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments import summarize
from src.stats_tests import friedman_test, wilcoxon_pairwise
from src.elasticity import plausibility_scorecard


BASELINES = {"naive", "seasonal_naive"}


def _run_from_path(path: str, prefix: str, suffix: str) -> str:
    name = Path(path).name
    return re.sub(rf"^{re.escape(prefix)}|{re.escape(suffix)}$", "", name)


# ----------------------------------------------------------------------
# Layer 1 — CV leaderboard
# ----------------------------------------------------------------------
def load_cv_metrics(outputs_dir: Path) -> pd.DataFrame:
    """Concat all metrics_*.csv; overwrite `model` column with run name."""
    dfs = []
    for f in sorted(outputs_dir.glob("metrics_*.csv")):
        run = _run_from_path(str(f), "metrics_", ".csv")
        d = pd.read_csv(f)
        d["model"] = run
        dfs.append(d)
    if not dfs:
        raise SystemExit(f"No metrics_*.csv found in {outputs_dir}")
    return pd.concat(dfs, ignore_index=True)


def cv_leaderboard(all_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean + CI95 on the primary metric, plus mean-only on a few secondaries."""
    primary = summarize(all_metrics, metric=metric).rename(
        columns={"mean": f"mean_{metric}", "std": f"std_{metric}", "ci95": f"ci95_{metric}"}
    )
    primary["rank"] = primary[f"mean_{metric}"].rank(method="min").astype(int)
    secondary_cols = [c for c in ("rmse", "rmsle", "mae", "r2") if c != metric]
    for m in secondary_cols:
        if m not in all_metrics.columns:
            continue
        s = all_metrics.groupby("model")[m].mean()
        primary[f"mean_{m}"] = s
    return primary


# ----------------------------------------------------------------------
# Layer 2 — Friedman / Nemenyi / CD plot
# ----------------------------------------------------------------------
def run_significance(all_metrics: pd.DataFrame, metric: str, outputs_dir: Path) -> dict:
    out: dict = {}

    fr = friedman_test(all_metrics, metric=metric)
    (outputs_dir / "friedman.json").write_text(json.dumps(fr, indent=2))
    out["friedman"] = fr

    # Nemenyi + CD plot (optional; graceful skip if scikit-posthocs missing)
    try:
        from src.stats_tests import nemenyi_posthoc, critical_difference_diagram
        nem = nemenyi_posthoc(all_metrics, metric=metric)
        # Re-label nemenyi matrix: scikit-posthocs returns integer col names
        cols = sorted(all_metrics["model"].unique())
        nem.index = cols
        nem.columns = cols
        nem.to_csv(outputs_dir / "nemenyi.csv")
        out["nemenyi"] = nem

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, max(3.5, len(cols) * 0.25)))
        critical_difference_diagram(all_metrics, metric=metric, ax=ax)
        fig.savefig(outputs_dir / "cd_plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("    (scikit-posthocs not installed; falling back to Wilcoxon)")
        out["nemenyi"] = None

    # Always run Wilcoxon as a second-opinion pairwise
    wil = wilcoxon_pairwise(all_metrics, metric=metric)
    wil.to_csv(outputs_dir / "wilcoxon.csv", index=False)
    out["wilcoxon"] = wil
    return out


def baseline_pvalue(sig: dict, run: str, baseline: str = "seasonal_naive") -> float | None:
    """Pull the pairwise p-value between `run` and a baseline from Nemenyi
    if available, else Wilcoxon. Returns None if not found."""
    nem = sig.get("nemenyi")
    if nem is not None and run in nem.index and baseline in nem.columns:
        return float(nem.loc[run, baseline])
    wil = sig.get("wilcoxon")
    if wil is None:
        return None
    mask = (
        ((wil["model_a"] == run) & (wil["model_b"] == baseline))
        | ((wil["model_a"] == baseline) & (wil["model_b"] == run))
    )
    if mask.any():
        return float(wil.loc[mask, "p_value"].iloc[0])
    return None


# ----------------------------------------------------------------------
# Layer 3 — Sealed-test rank
# ----------------------------------------------------------------------
def load_sealed_test(outputs_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(outputs_dir.glob("test_metrics_*.json")):
        run = _run_from_path(str(f), "test_metrics_", ".json")
        m = json.loads(f.read_text())
        m["run"] = run
        rows.append(m)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("run")


# ----------------------------------------------------------------------
# Layer 4 — Elasticity plausibility
# ----------------------------------------------------------------------
def load_elasticity_scorecards(outputs_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(outputs_dir.glob("elasticity_*.csv")):
        run = _run_from_path(str(f), "elasticity_", ".csv")
        if run == "scorecard":   # our own output file; skip
            continue
        df = pd.read_csv(f)
        card = plausibility_scorecard(df)
        card["run"] = run
        rows.append(card)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("run")


# ----------------------------------------------------------------------
# Composite
# ----------------------------------------------------------------------
def composite(
    cv: pd.DataFrame,
    sealed: pd.DataFrame,
    elast: pd.DataFrame,
    sig: dict,
    metric: str,
    top_k: int,
) -> pd.DataFrame:
    """Join the 4 layers per run and flag champion gates."""
    mean_col = f"mean_{metric}"
    df = cv[[mean_col, f"ci95_{metric}", "rank"]].copy()
    df.columns = ["cv_mean", "cv_ci95", "cv_rank"]

    # Layer 2: p vs seasonal_naive (lower p = more significant improvement)
    df["p_vs_seasonal_naive"] = [
        baseline_pvalue(sig, r) for r in df.index
    ]

    # Layer 3: sealed-test metric + rank (ML runs only; baselines have no sealed)
    if not sealed.empty and metric in sealed.columns:
        df["sealed"] = sealed[metric]
        df["sealed_rank"] = sealed[metric].rank(method="min")
    else:
        df["sealed"] = np.nan
        df["sealed_rank"] = np.nan

    # Layer 4: sign + magnitude
    # Use the nonzero-aware sign test: β=0 rows are tree regions where
    # price didn't split, not "wrong direction". Monotone constraint
    # guarantees no β>0, so share_negative_among_nonzero is the honest
    # direction check.
    if not elast.empty:
        df["sign_pass"] = elast["sign_test_pass_nonzero"]
        df["share_in_range"] = elast["share_in_soft_drink_range"]
        df["share_neg_nonzero"] = elast["share_negative_among_nonzero"]
        df["share_zero_beta"] = elast["share_zero_beta"]
        df["median_beta"] = elast["median_beta"]
    else:
        df["sign_pass"] = pd.NA
        df["share_in_range"] = np.nan
        df["share_neg_nonzero"] = np.nan
        df["share_zero_beta"] = np.nan
        df["median_beta"] = np.nan

    # Champion gates (baselines excluded from champion eligibility)
    is_baseline = df.index.to_series().isin(BASELINES)
    gate_cv     = df["cv_rank"] <= top_k
    gate_sig    = df["p_vs_seasonal_naive"].fillna(1.0) < 0.05
    gate_sealed = df["sealed_rank"].fillna(99) <= top_k
    gate_elast  = df["sign_pass"].fillna(False).astype(bool) & (df["share_in_range"].fillna(0) > 0.5)
    df["gate_cv"]     = gate_cv
    df["gate_sig"]    = gate_sig
    df["gate_sealed"] = gate_sealed
    df["gate_elast"]  = gate_elast
    df["champion_candidate"] = (
        gate_cv & gate_sig & gate_sealed & gate_elast & ~is_baseline
    )
    return df.sort_values("cv_mean")


def pick_champion(composite_df: pd.DataFrame, metric: str) -> str | None:
    cands = composite_df[composite_df["champion_candidate"]]
    if cands.empty:
        return None
    # Tiebreaker: best CV metric among gate-passers
    return cands["cv_mean"].idxmin()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="wmape",
                    help="Primary ranking metric (raw-space). Default wmape.")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--top-k", type=int, default=5,
                    help="A run must rank within top_k on both CV and sealed test.")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ---- L1 CV ----
    print(f"[L1] Loading CV metrics from {outputs_dir}/metrics_*.csv ...")
    all_metrics = load_cv_metrics(outputs_dir)
    runs = sorted(all_metrics["model"].unique())
    print(f"    {len(runs)} runs ({[r for r in runs if r in BASELINES]} are baselines)")

    cv = cv_leaderboard(all_metrics, metric=args.metric)
    cv.to_csv(outputs_dir / "leaderboard_cv.csv")
    print(f"    saved {outputs_dir / 'leaderboard_cv.csv'}")

    # ---- L2 Significance ----
    print("[L2] Friedman + Nemenyi + Wilcoxon ...")
    sig = run_significance(all_metrics, args.metric, outputs_dir)
    print(f"    Friedman chi2={sig['friedman']['chi2']:.1f}  "
          f"p={sig['friedman']['p_value']:.2e}  "
          f"n_obs={sig['friedman']['n_observations']}")

    # ---- L3 Sealed ----
    print("[L3] Sealed-test rank ...")
    sealed = load_sealed_test(outputs_dir)
    if not sealed.empty:
        sealed_out = sealed[[args.metric]].copy()
        sealed_out["rank"] = sealed_out[args.metric].rank(method="min").astype(int)
        sealed_out = sealed_out.sort_values(args.metric)
        sealed_out.to_csv(outputs_dir / "leaderboard_sealed.csv")
        print(f"    saved {outputs_dir / 'leaderboard_sealed.csv'}")

        # Spearman CV vs sealed (on shared runs)
        from scipy.stats import spearmanr
        shared = cv.index.intersection(sealed.index)
        rho, rho_p = spearmanr(cv.loc[shared, f"mean_{args.metric}"],
                               sealed.loc[shared, args.metric])
        print(f"    Spearman rho(CV,sealed) = {rho:.2f}  (p={rho_p:.2e})"
              f"  [>{0.7:.1f} = stable generalization]")
    else:
        print("    (no test_metrics_*.json found)")

    # ---- L4 Elasticity ----
    print("[L4] Elasticity plausibility scorecard ...")
    elast = load_elasticity_scorecards(outputs_dir)
    if not elast.empty:
        elast.to_csv(outputs_dir / "elasticity_scorecard.csv")
        print(f"    saved {outputs_dir / 'elasticity_scorecard.csv'}")
    else:
        print("    (no elasticity_*.csv found)")

    # ---- Composite + champion ----
    print("[C] Composite + champion selection ...")
    comp = composite(cv, sealed, elast, sig, args.metric, args.top_k)
    comp.to_csv(outputs_dir / "composite.csv")
    print(f"    saved {outputs_dir / 'composite.csv'}")

    print("\n" + "=" * 80)
    print(f"  COMPOSITE TABLE (sorted by CV {args.metric}, top {min(args.top_k*3, len(comp))}):")
    print("=" * 80)
    show_cols = ["cv_mean", "cv_ci95", "cv_rank", "p_vs_seasonal_naive",
                 "sealed", "sealed_rank", "sign_pass", "share_in_range",
                 "share_neg_nonzero", "share_zero_beta", "median_beta",
                 "gate_cv", "gate_sig", "gate_sealed", "gate_elast", "champion_candidate"]
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(comp[show_cols].head(min(args.top_k * 3, len(comp))).to_string())

    champ = pick_champion(comp, args.metric)
    print("\n" + "=" * 80)
    if champ is None:
        print("  CHAMPION: none — no run passed all 4 gates.")
        print("  Candidates with most gates passed:")
        comp["gates_passed"] = (
            comp["gate_cv"].astype(int) + comp["gate_sig"].astype(int)
            + comp["gate_sealed"].astype(int) + comp["gate_elast"].astype(int)
        )
        print(comp.sort_values(["gates_passed", "cv_mean"], ascending=[False, True])
              [["gates_passed", "cv_mean", "sealed", "sign_pass", "share_in_range"]]
              .head(5).to_string())
    else:
        card = {
            "champion": champ,
            "metric": args.metric,
            "cv_mean": float(comp.loc[champ, "cv_mean"]),
            "cv_ci95": float(comp.loc[champ, "cv_ci95"]),
            "cv_rank": int(comp.loc[champ, "cv_rank"]),
            "sealed": float(comp.loc[champ, "sealed"]),
            "sealed_rank": int(comp.loc[champ, "sealed_rank"]),
            "p_vs_seasonal_naive": float(comp.loc[champ, "p_vs_seasonal_naive"]),
            "share_negative_beta": float(elast.loc[champ, "share_negative_beta"]) if champ in elast.index else None,
            "median_beta": float(elast.loc[champ, "median_beta"]) if champ in elast.index else None,
            "share_in_soft_drink_range": float(elast.loc[champ, "share_in_soft_drink_range"]) if champ in elast.index else None,
            "joblib_path": f"outputs/model_{champ}.joblib",
            "metadata_path": f"outputs/metadata_{champ}.json",
        }
        (outputs_dir / "champion_card.json").write_text(json.dumps(card, indent=2))
        print(f"  CHAMPION: {champ}")
        print(f"  CV {args.metric}:    {card['cv_mean']:.4f} ± {card['cv_ci95']:.4f}  (rank {card['cv_rank']})")
        print(f"  Sealed {args.metric}:{card['sealed']:.4f}  (rank {card['sealed_rank']})")
        print(f"  vs seasonal_naive:   p = {card['p_vs_seasonal_naive']:.2e}")
        print(f"  Elasticity:          {card['share_negative_beta']*100:.0f}% negative, "
              f"median β = {card['median_beta']:.2f}, "
              f"{card['share_in_soft_drink_range']*100:.0f}% in [-3.5, -0.5]")
        print(f"  joblib: {card['joblib_path']}")
        print(f"  saved {outputs_dir / 'champion_card.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
