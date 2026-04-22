"""Industrial-grade feature selection: VIF pruning + BorutaShap + stability.

Usage:
    keep = run_full_pipeline(df, target_col='log_nielsen_total_volume', model_type='xgb')
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from .config import PROTECTED_FEATURES


def vif_prune(
    df: pd.DataFrame,
    candidate_cols: list[str],
    vif_threshold: float = 5.0,
    protected: list[str] | None = None,
) -> list[str]:
    """Iteratively drop the feature with highest VIF until all VIF < threshold.

    Protected features are never dropped (domain core).
    Requires statsmodels.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    protected = set(protected or [])
    cols = [c for c in candidate_cols if c in df.columns]
    X = df[cols].dropna().astype(float)
    if X.empty:
        return cols

    current = list(cols)
    while len(current) > 1:
        Xc = X[current].values
        vifs = []
        for i in range(Xc.shape[1]):
            try:
                v = variance_inflation_factor(Xc, i)
            except Exception:
                v = np.inf
            vifs.append(v)
        vif_series = pd.Series(vifs, index=current).sort_values(ascending=False)
        top_feat, top_vif = vif_series.index[0], vif_series.iloc[0]
        if top_vif < vif_threshold or top_feat in protected:
            # either done, or top is protected -> try next highest non-protected
            droppable = [f for f, v in vif_series.items() if f not in protected and v >= vif_threshold]
            if not droppable:
                break
            top_feat = droppable[0]
        current.remove(top_feat)
    return current


def correlation_prune(
    df: pd.DataFrame,
    candidate_cols: list[str],
    abs_threshold: float = 0.95,
    protected: list[str] | None = None,
) -> list[str]:
    """Drop one of each highly-correlated pair."""
    protected = set(protected or [])
    X = df[candidate_cols].dropna()
    if X.empty:
        return candidate_cols
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop: set[str] = set()
    for col in upper.columns:
        if col in protected or col in drop:
            continue
        high = upper.index[upper[col] > abs_threshold]
        for peer in high:
            if peer in protected:
                drop.add(col)
                break
            drop.add(peer)
    return [c for c in candidate_cols if c not in drop]


def _make_fs_estimator(model_type: str, random_state: int):
    if model_type == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6, 
            learning_rate=0.1,
            tree_method="hist", 
            random_state=random_state, 
            n_jobs=-1,
        )
    if model_type == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    if model_type == "rf":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def borutashap_select(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str = "xgb",
    n_trials: int = 50,
    random_state: int = 42,
    alpha: float = 0.05,
) -> list[str]:
    """Self-contained Boruta-SHAP (replaces the abandoned BorutaShap package).

    For each of ``n_trials``:
      1. Create shadow features by shuffling each real feature independently.
      2. Fit a tree estimator on [real | shadow] concatenated.
      3. Compute mean(|SHAP|) per feature. Count a "hit" for real feature i
         if its importance exceeds the max importance across all shadow
         features in this trial.

    A real feature is accepted if its hit count is significantly greater
    than what a 50%-coin-flip baseline would produce (one-sided binomial
    test, p < ``alpha``).

    References: Kursa & Rudnicki (2010) Boruta + Keany (2020) Boruta-SHAP.
    """
    import shap
    from scipy.stats import binomtest

    rng = np.random.default_rng(random_state)
    feature_cols = list(X.columns)
    X_real = X.reset_index(drop=True).astype(float)
    y_arr = np.asarray(y, dtype=float)
    hits = {c: 0 for c in feature_cols}

    for trial in range(n_trials):
        t0 = time.time()
        trial_seed = int(rng.integers(0, 1_000_000))
        perm = np.random.default_rng(trial_seed)
        shadow = pd.DataFrame(
            {f"shadow_{c}": perm.permutation(X_real[c].to_numpy()) for c in feature_cols}
        )
        X_aug = pd.concat([X_real, shadow], axis=1)

        est = _make_fs_estimator(model_type, random_state=trial_seed)
        est.fit(X_aug, y_arr)

        try:
            explainer = shap.TreeExplainer(est)
            shap_vals = explainer.shap_values(X_aug)
        except Exception as e:
            warnings.warn(
                f"[boruta] trial {trial+1}: SHAP failed "
                f"({type(e).__name__}: {e}); falling back to feature_importances_ "
                "(HistGBR returns zeros)",
                RuntimeWarning,
                stacklevel=2,
            )
            importances = np.asarray(getattr(est, "feature_importances_", np.zeros(X_aug.shape[1])))
            imp_series = pd.Series(importances, index=X_aug.columns)
        else:
            imp_series = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_aug.columns)

        shadow_cols = [c for c in X_aug.columns if c.startswith("shadow_")]
        max_shadow = float(imp_series[shadow_cols].max())

        for c in feature_cols:
            if float(imp_series[c]) > max_shadow:
                hits[c] += 1

        if trial == 0 or (trial + 1) % 5 == 0 or trial + 1 == n_trials:
            print(f"    [boruta] trial {trial+1}/{n_trials} took {time.time()-t0:.1f}s")

    accepted: list[str] = []
    for c in feature_cols:
        p = binomtest(hits[c], n=n_trials, p=0.5, alternative="greater").pvalue
        if p < alpha:
            accepted.append(c)
    return accepted


def stability_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str = "xgb",
    n_boot: int = 50,
    vote_threshold: float = 0.8,
    trials_per_boot: int = 30,
    random_state: int = 42,
) -> tuple[list[str], pd.Series]:
    """Bootstrap BorutaShap n_boot times; keep features selected in >= vote_threshold."""
    rng = np.random.default_rng(random_state)
    n = len(X)
    counts = pd.Series(0, index=X.columns, dtype=int)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X.iloc[idx].reset_index(drop=True)
        yb = np.asarray(y)[idx]
        try:
            sel = borutashap_select(
                Xb, yb, model_type=model_type,
                n_trials=trials_per_boot,
                random_state=int(rng.integers(0, 10_000)),
            )
        except Exception:
            continue
        for c in sel:
            counts[c] += 1
    frequencies = counts / n_boot
    kept = frequencies[frequencies >= vote_threshold].index.tolist()
    return kept, frequencies


def elastic_net_select(
    X: pd.DataFrame,
    y: np.ndarray,
    l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.7, 0.9, 0.95),
    cv: int = 4,
    coef_eps: float = 1e-8,
    random_state: int = 42,
) -> tuple[list[str], dict]:
    """L1-based feature selection for linear models (Elastic Net's "Step 3").

    Symmetric to ``borutashap_select`` for tree models: selects a subset of
    features using the model's own sparsity mechanism. ElasticNetCV finds the
    selection-optimal (alpha, l1_ratio) via internal CV on standardized
    features; features with |coef| > coef_eps are retained. Downstream Optuna
    re-tunes (alpha, l1_ratio) for predictive power on this reduced set.
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X.astype(float))
    enet = ElasticNetCV(
        l1_ratio=list(l1_ratios),
        cv=cv,
        max_iter=20_000,
        random_state=random_state,
        n_jobs=-1,
    ).fit(Xs, np.asarray(y, dtype=float))
    coefs = enet.coef_
    selected = [c for c, b in zip(X.columns, coefs) if abs(b) > coef_eps]
    info = {
        "alpha_selected": float(enet.alpha_),
        "l1_ratio_selected": float(enet.l1_ratio_),
        "n_selected": len(selected),
        "coefs": dict(zip(X.columns, [float(b) for b in coefs])),
    }
    return selected, info


def elastic_net_stability_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    n_boot: int = 50,
    vote_threshold: float = 0.8,
    l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.7, 0.9, 0.95),
    random_state: int = 42,
) -> tuple[list[str], pd.Series]:
    """Bootstrap L1 selection n_boot times; keep features selected in >= vote_threshold.

    Linear-model analogue to ``stability_selection`` (Meinshausen & Buhlmann 2010).
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    counts = pd.Series(0, index=X.columns, dtype=int)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X.iloc[idx].reset_index(drop=True)
        yb = np.asarray(y)[idx]
        try:
            sel, _ = elastic_net_select(
                Xb, yb, l1_ratios=l1_ratios,
                random_state=int(rng.integers(0, 10_000)),
            )
        except Exception:
            continue
        for c in sel:
            counts[c] += 1
    frequencies = counts / n_boot
    kept = frequencies[frequencies >= vote_threshold].index.tolist()
    return kept, frequencies


def run_full_pipeline(
    df: pd.DataFrame,
    candidate_cols: list[str],
    y: np.ndarray,
    model_type: str = "xgb",
    do_stability: bool = True,
    random_state: int = 42,
) -> dict:
    """Full 4-step industrial feature selection.

    Tree models (xgb/lgb/rf): VIF -> BorutaShap -> (optional) Stability.
    Elastic Net:              VIF -> L1 (ElasticNetCV) -> (optional) L1 Stability.
    Boruta is skipped for Elastic Net -- L1 sparsity IS the selector;
    external shadow/SHAP is not meaningful for linear models.

    Returns a dict with keys: step1_core, step2_after_vif, step3_*, step4_stable, final.
    """
    protected = [c for c in PROTECTED_FEATURES if c in candidate_cols]

    # Step 2: VIF + correlation
    after_corr = correlation_prune(df, candidate_cols, protected=protected)
    after_vif = vif_prune(df, after_corr, protected=protected)

    # Elastic Net branch: L1-based selection, symmetric to Boruta for trees
    if model_type == "elastic_net":
        X_lin = df[after_vif].dropna().astype(float)
        y_lin = np.asarray(y)[df.index.get_indexer(X_lin.index)]
        try:
            l1_sel, l1_info = elastic_net_select(X_lin, y_lin, random_state=random_state)
        except Exception as e:
            l1_sel, l1_info = after_vif, {"error": str(e)}
            print(f"[feature_selection] elastic_net_select failed: {e}. Falling back to VIF set.")

        stable_lin: list[str] = l1_sel
        freqs_lin: pd.Series | None = None
        if do_stability:
            try:
                stable_lin, freqs_lin = elastic_net_stability_selection(
                    X_lin, y_lin, random_state=random_state
                )
            except Exception as e:
                print(f"[feature_selection] elastic_net_stability_selection failed: {e}")

        final = list(dict.fromkeys(protected + stable_lin))
        return {
            "step1_core": protected,
            "step2_after_vif": after_vif,
            "step3_l1_select": l1_sel,
            "step3_info": l1_info,
            "step4_stable": stable_lin,
            "step4_frequencies": freqs_lin,
            "final": final,
        }

    # Step 3: BorutaShap single run
    X = df[after_vif].dropna().astype(float)
    y_aligned = np.asarray(y)[df.index.get_indexer(X.index)]
    try:
        boruta_sel = borutashap_select(X, y_aligned, model_type=model_type,
                                       random_state=random_state)
    except Exception as e:
        boruta_sel = after_vif  # fallback: accept all non-collinear
        print(f"[feature_selection] BorutaShap failed: {e}. Falling back to VIF set.")

    # Step 4: stability selection
    stable: list[str] = boruta_sel
    freqs: pd.Series | None = None
    if do_stability:
        try:
            stable, freqs = stability_selection(
                X, y_aligned, model_type=model_type, random_state=random_state
            )
        except Exception as e:
            print(f"[feature_selection] stability selection failed: {e}")

    # Final: union of (protected core) and (stable selected)
    final = list(dict.fromkeys(protected + stable))
    return {
        "step1_core": protected,
        "step2_after_vif": after_vif,
        "step3_boruta": boruta_sel,
        "step4_stable": stable,
        "step4_frequencies": freqs,
        "final": final,
    }
