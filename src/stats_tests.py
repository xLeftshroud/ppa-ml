"""Statistical significance tests for the 5-model comparison.

Given a long DataFrame (model, seed, fold, wmape, ...) with N_models x 20
observations (5 seeds x 4 folds), runs:
  - Friedman test           -> chi-square + p-value + avg rank per model
  - Nemenyi post-hoc         -> pairwise p-value matrix
  - Wilcoxon signed-rank    -> pairwise p-value + Cohen's d (effect size)
  - Critical Difference plot (Demsar 2006) via scikit-posthocs

Baselines are included in the Friedman/Nemenyi tests so any ML claim is
"significantly better than seasonal-naive" before being claimed at all.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _pivot_observations(
    metrics_df: pd.DataFrame, metric: str = "wmape"
) -> pd.DataFrame:
    """Pivot to observations (rows = seed*fold, cols = model) aligned."""
    metrics_df = metrics_df.copy()
    metrics_df["obs_id"] = (
        metrics_df["seed"].astype(str) + "_f" + metrics_df["fold"].astype(str)
    )
    wide = metrics_df.pivot(index="obs_id", columns="model", values=metric)
    return wide.dropna(how="any")


def friedman_test(metrics_df: pd.DataFrame, metric: str = "wmape") -> dict:
    """Return chi-square, p-value, and average rank per model (lower=better)."""
    from scipy.stats import friedmanchisquare

    wide = _pivot_observations(metrics_df, metric)
    cols = list(wide.columns)
    stat, p = friedmanchisquare(*(wide[c].values for c in cols))
    ranks = wide.rank(axis=1, method="average").mean().sort_values()
    return {
        "chi2": float(stat),
        "p_value": float(p),
        "avg_rank": ranks.to_dict(),
        "n_observations": int(len(wide)),
        "n_models": len(cols),
    }


def nemenyi_posthoc(metrics_df: pd.DataFrame, metric: str = "wmape") -> pd.DataFrame:
    """Pairwise Nemenyi p-values (requires scikit-posthocs)."""
    import scikit_posthocs as sp

    wide = _pivot_observations(metrics_df, metric)
    return sp.posthoc_nemenyi_friedman(wide.values)


def wilcoxon_pairwise(metrics_df: pd.DataFrame, metric: str = "wmape") -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank + Cohen's d.

    Cohen's d here uses the paired-differences interpretation:
    d = mean(diff) / std(diff).
    """
    from scipy.stats import wilcoxon

    wide = _pivot_observations(metrics_df, metric)
    models = list(wide.columns)
    rows = []
    for a, b in itertools.combinations(models, 2):
        da = wide[a].values
        db = wide[b].values
        diff = da - db
        if np.all(diff == 0):
            p, stat = 1.0, 0.0
        else:
            try:
                res = wilcoxon(da, db)
                stat, p = float(res.statistic), float(res.pvalue)
            except Exception:
                stat, p = np.nan, np.nan
        sd = float(np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else np.nan
        cohens_d = float(np.mean(diff)) / sd if sd and not np.isnan(sd) else np.nan
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "wilcoxon_stat": stat,
                "p_value": p,
                "mean_diff_a_minus_b": float(np.mean(diff)),
                "cohens_d": cohens_d,
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def critical_difference_diagram(
    metrics_df: pd.DataFrame, metric: str = "wmape", alpha: float = 0.05, ax=None
):
    """Draw a Demsar (2006) critical-difference diagram."""
    import scikit_posthocs as sp
    import matplotlib.pyplot as plt

    wide = _pivot_observations(metrics_df, metric)
    avg_ranks = wide.rank(axis=1, method="average").mean()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2.5))
    sp.critical_difference_diagram(
        ranks=avg_ranks,
        sig_matrix=sp.posthoc_nemenyi_friedman(wide.values),
        ax=ax,
        label_fmt_left="{label} ({rank:.2f})",
        label_fmt_right="({rank:.2f}) {label}",
    )
    return ax
