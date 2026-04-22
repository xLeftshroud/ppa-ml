"""Hierarchical Bayesian demand model v6 -- brand-nested + flavor/pack crossed.

Industrial-grade PPA hierarchy for Coca-Cola Europe-style panels:

    leaf node = cell = (top_brand, flavor_internal, pack_tier)   ~100-150 cells

    beta_cell = mu_brand[b(c)]                   (nested: brand pool under global)
              + alpha_flavor[f(c)]               (crossed random effect)
              + alpha_pack[p(c)]                 (crossed random effect, PPA core)
              + eps_cell[c]                      (additive-break residual)

    beta_cell_final = -exp(beta_cell)            (hard constraint beta <= 0)

Intercept alpha_cell is also hierarchical (pooled to brand). Global coefficients
on promotion, seasonality, and optional intercept-shift covariates (units per
pack, log pack size, pack_type dummies, customer dummies).

All hierarchies are non-centered to avoid Neal's funnel.

Prior sensitivity (``prior_scale``):
    weak     -> mu_global ~ Normal(0,5),  sigma_* inflated
    moderate -> mu_global ~ Normal(-2,1), sigma_* nominal         (default)
    strong   -> mu_global ~ Normal(-2,0.5), sigma_* shrunk
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


PRIOR_CONFIGS: dict[str, dict] = {
    "weak":     {"mu_loc": 0.0,  "mu_scale": 5.0, "sigma_scale": 2.0},
    "moderate": {"mu_loc": -2.0, "mu_scale": 1.0, "sigma_scale": 1.0},
    "strong":   {"mu_loc": -2.0, "mu_scale": 0.5, "sigma_scale": 0.5},
}

BRAND_COL = "top_brand"
FLAVOR_COL = "flavor_internal"
PACK_COL = "pack_tier"
PACK_TYPE_COL = "pack_type_internal"
CUSTOMER_COL = "customer"
UNITS_COL = "units_per_package_internal"
SIZE_COL = "pack_size_internal"


@dataclass
class BayesianHierModel:
    prior_scale: Literal["weak", "moderate", "strong"] = "moderate"
    num_warmup: int = 1000
    num_samples: int = 2000
    num_chains: int = 2
    random_state: int = 42

    def __post_init__(self):
        self.samples_: dict | None = None
        self.idata_ = None
        # code maps
        self._brand_codes_: dict | None = None
        self._flavor_codes_: dict | None = None
        self._pack_codes_: dict | None = None
        self._cell_codes_: dict | None = None
        self._cell_keys_: list[tuple] | None = None
        # cell -> (brand, flavor, pack) index arrays
        self._cell_to_brand_: np.ndarray | None = None
        self._cell_to_flavor_: np.ndarray | None = None
        self._cell_to_pack_: np.ndarray | None = None
        # covariate encoders
        self._pack_type_levels_: list[str] | None = None
        self._customer_levels_: list[str] | None = None
        # cached posterior means (for predict)
        self._alpha_cell_mean_: np.ndarray | None = None
        self._beta_cell_mean_: np.ndarray | None = None
        self._mu_brand_mean_: np.ndarray | None = None
        self._alpha_flavor_mean_: np.ndarray | None = None
        self._alpha_pack_mean_: np.ndarray | None = None
        self._mu_alpha_brand_mean_: np.ndarray | None = None
        self._gamma_mean_: float = 0.0
        self._delta_sin_mean_: float = 0.0
        self._delta_cos_mean_: float = 0.0
        self._theta_units_mean_: float = 0.0
        self._theta_size_mean_: float = 0.0
        self._theta_packtype_mean_: np.ndarray | None = None
        self._theta_customer_mean_: np.ndarray | None = None
        # per-cell SKU counts (for reporting)
        self._cell_n_skus_: dict | None = None

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _cell_key(self, row) -> tuple:
        return (row[BRAND_COL], row[FLAVOR_COL], row[PACK_COL])

    def _encode_indices(self, df: pd.DataFrame):
        """Build brand / flavor / pack / cell / pack_type / customer code maps."""
        brand_cat = pd.Categorical(df[BRAND_COL])
        flavor_cat = pd.Categorical(df[FLAVOR_COL])
        pack_cat = pd.Categorical(df[PACK_COL])
        self._brand_codes_ = {c: i for i, c in enumerate(brand_cat.categories)}
        self._flavor_codes_ = {c: i for i, c in enumerate(flavor_cat.categories)}
        self._pack_codes_ = {c: i for i, c in enumerate(pack_cat.categories)}

        cell_keys_df = df[[BRAND_COL, FLAVOR_COL, PACK_COL]].drop_duplicates().reset_index(drop=True)
        self._cell_keys_ = [tuple(r) for r in cell_keys_df.itertuples(index=False)]
        self._cell_codes_ = {k: i for i, k in enumerate(self._cell_keys_)}

        # cell -> (brand, flavor, pack) index mappings
        self._cell_to_brand_ = np.array(
            [self._brand_codes_[b] for (b, _, _) in self._cell_keys_], dtype=np.int32
        )
        self._cell_to_flavor_ = np.array(
            [self._flavor_codes_[f] for (_, f, _) in self._cell_keys_], dtype=np.int32
        )
        self._cell_to_pack_ = np.array(
            [self._pack_codes_[p] for (_, _, p) in self._cell_keys_], dtype=np.int32
        )

        # pack_type dummy levels (first category as reference)
        if PACK_TYPE_COL in df.columns:
            pt_cat = pd.Categorical(df[PACK_TYPE_COL])
            self._pack_type_levels_ = list(pt_cat.categories)
        else:
            self._pack_type_levels_ = []

        # customer dummy levels (first category as reference)
        if CUSTOMER_COL in df.columns:
            c_cat = pd.Categorical(df[CUSTOMER_COL])
            self._customer_levels_ = list(c_cat.categories)
        else:
            self._customer_levels_ = []

        # per-cell SKU counts for reporting
        if "product_sku_code" in df.columns:
            self._cell_n_skus_ = (
                df.groupby([BRAND_COL, FLAVOR_COL, PACK_COL])["product_sku_code"]
                .nunique()
                .to_dict()
            )
        else:
            self._cell_n_skus_ = {k: 0 for k in self._cell_keys_}

        cell_idx = np.array(
            [self._cell_codes_[(b, f, p)] for b, f, p in zip(
                df[BRAND_COL].values, df[FLAVOR_COL].values, df[PACK_COL].values
            )],
            dtype=np.int32,
        )
        return cell_idx

    def _encode_pack_type_dummies(self, df: pd.DataFrame) -> np.ndarray:
        """Return (n_rows, n_levels-1) dummy matrix; first level is reference."""
        if not self._pack_type_levels_ or len(self._pack_type_levels_) < 2:
            return np.zeros((len(df), 0), dtype=np.float32)
        dummies = np.zeros((len(df), len(self._pack_type_levels_) - 1), dtype=np.float32)
        vals = df[PACK_TYPE_COL].values
        for j, lvl in enumerate(self._pack_type_levels_[1:]):
            dummies[:, j] = (vals == lvl).astype(np.float32)
        return dummies

    def _encode_customer_idx(self, df: pd.DataFrame) -> np.ndarray:
        """Return (n_rows,) int array of customer index in self._customer_levels_.
        Unknown customer -> 0 (reference)."""
        if not self._customer_levels_:
            return np.zeros(len(df), dtype=np.int32)
        lvl_map = {c: i for i, c in enumerate(self._customer_levels_)}
        return np.array(
            [lvl_map.get(c, 0) for c in df[CUSTOMER_COL].values], dtype=np.int32
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _model(
        self,
        cell_idx,
        log_price,
        promo,
        week_sin,
        week_cos,
        log_units,
        log_size,
        pack_type_dum,
        customer_idx,
        n_customer,
        y=None,
    ):
        import numpyro
        import numpyro.distributions as dist
        import jax.numpy as jnp

        pc = PRIOR_CONFIGS[self.prior_scale]
        s = pc["sigma_scale"]
        n_brand = len(self._brand_codes_)
        n_flavor = len(self._flavor_codes_)
        n_pack = len(self._pack_codes_)
        n_cell = len(self._cell_codes_)
        n_pt_dum = pack_type_dum.shape[1]

        cell_to_brand = jnp.array(self._cell_to_brand_)
        cell_to_flavor = jnp.array(self._cell_to_flavor_)
        cell_to_pack = jnp.array(self._cell_to_pack_)

        # ---------- hyper ----------
        mu_global = numpyro.sample(
            "mu_global", dist.Normal(pc["mu_loc"], pc["mu_scale"])
        )
        sigma_brand = numpyro.sample("sigma_brand", dist.HalfNormal(1.0 * s))
        sigma_flavor = numpyro.sample("sigma_flavor", dist.HalfNormal(0.5 * s))
        sigma_pack = numpyro.sample("sigma_pack", dist.HalfNormal(0.5 * s))
        sigma_cell = numpyro.sample("sigma_cell", dist.HalfNormal(0.3 * s))
        sigma_alpha_brand = numpyro.sample("sigma_alpha_brand", dist.HalfNormal(1.0))
        sigma_alpha_cell = numpyro.sample("sigma_alpha_cell", dist.HalfNormal(0.5))

        # ---------- slope hierarchy (non-centered) ----------
        mu_brand_raw = numpyro.sample(
            "mu_brand_raw", dist.Normal(0.0, 1.0).expand([n_brand]).to_event(1)
        )
        mu_brand = numpyro.deterministic("mu_brand", mu_global + sigma_brand * mu_brand_raw)

        alpha_flavor_raw = numpyro.sample(
            "alpha_flavor_raw", dist.Normal(0.0, 1.0).expand([n_flavor]).to_event(1)
        )
        alpha_flavor = numpyro.deterministic("alpha_flavor", sigma_flavor * alpha_flavor_raw)

        alpha_pack_raw = numpyro.sample(
            "alpha_pack_raw", dist.Normal(0.0, 1.0).expand([n_pack]).to_event(1)
        )
        alpha_pack = numpyro.deterministic("alpha_pack", sigma_pack * alpha_pack_raw)

        eps_cell_raw = numpyro.sample(
            "eps_cell_raw", dist.Normal(0.0, 1.0).expand([n_cell]).to_event(1)
        )
        eps_cell = sigma_cell * eps_cell_raw

        beta_cell_raw = (
            mu_brand[cell_to_brand]
            + alpha_flavor[cell_to_flavor]
            + alpha_pack[cell_to_pack]
            + eps_cell
        )
        # hard constraint: beta_cell <= 0
        beta_cell = numpyro.deterministic("beta_cell", -jnp.exp(beta_cell_raw))

        # ---------- intercept hierarchy (non-centered) ----------
        mu_alpha_global = numpyro.sample("mu_alpha_global", dist.Normal(0.0, 5.0))
        mu_alpha_brand_raw = numpyro.sample(
            "mu_alpha_brand_raw", dist.Normal(0.0, 1.0).expand([n_brand]).to_event(1)
        )
        mu_alpha_brand = numpyro.deterministic(
            "mu_alpha_brand", mu_alpha_global + sigma_alpha_brand * mu_alpha_brand_raw
        )

        alpha_cell_raw = numpyro.sample(
            "alpha_cell_raw", dist.Normal(0.0, 1.0).expand([n_cell]).to_event(1)
        )
        alpha_cell = numpyro.deterministic(
            "alpha_cell",
            mu_alpha_brand[cell_to_brand] + sigma_alpha_cell * alpha_cell_raw,
        )

        # ---------- global controls ----------
        gamma_promo = numpyro.sample("gamma_promo", dist.Normal(0.0, 1.0))
        delta_sin = numpyro.sample("delta_sin", dist.Normal(0.0, 1.0))
        delta_cos = numpyro.sample("delta_cos", dist.Normal(0.0, 1.0))

        # ---------- optional intercept-shift covariates ----------
        theta_units = numpyro.sample("theta_units", dist.Normal(0.0, 1.0))
        theta_size = numpyro.sample("theta_size", dist.Normal(0.0, 1.0))

        if n_pt_dum > 0:
            theta_packtype = numpyro.sample(
                "theta_packtype", dist.Normal(0.0, 1.0).expand([n_pt_dum]).to_event(1)
            )
            pt_contrib = pack_type_dum @ theta_packtype
        else:
            pt_contrib = 0.0

        if n_customer > 1:
            # reference category gets 0; sample n_customer-1 offsets
            theta_customer_nonref = numpyro.sample(
                "theta_customer_nonref",
                dist.Normal(0.0, 1.0).expand([n_customer - 1]).to_event(1),
            )
            theta_customer = jnp.concatenate(
                [jnp.zeros(1), theta_customer_nonref]
            )
            numpyro.deterministic("theta_customer", theta_customer)
            cust_contrib = theta_customer[customer_idx]
        else:
            cust_contrib = 0.0

        sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(1.0))

        # ---------- likelihood ----------
        mu = (
            alpha_cell[cell_idx]
            + beta_cell[cell_idx] * log_price
            + gamma_promo * promo
            + delta_sin * week_sin
            + delta_cos * week_cos
            + theta_units * log_units
            + theta_size * log_size
            + pt_contrib
            + cust_contrib
        )
        numpyro.sample("obs", dist.Normal(mu, sigma_y), obs=y)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "BayesianHierModel":
        import numpyro
        from numpyro.infer import MCMC, NUTS
        import jax.numpy as jnp
        import jax.random as jrand

        cell_idx_np = self._encode_indices(df)

        log_price = jnp.array(df["log_price_per_litre"].to_numpy(), dtype=jnp.float32)
        promo = jnp.array(df["promotion_indicator"].to_numpy(), dtype=jnp.float32)
        wsin = jnp.array(df["week_sin"].to_numpy(), dtype=jnp.float32)
        wcos = jnp.array(df["week_cos"].to_numpy(), dtype=jnp.float32)
        log_units = jnp.array(
            np.log1p(df[UNITS_COL].to_numpy().astype(np.float32))
            if UNITS_COL in df.columns else np.zeros(len(df), dtype=np.float32),
            dtype=jnp.float32,
        )
        size_arr = (
            df[SIZE_COL].to_numpy().astype(np.float32)
            if SIZE_COL in df.columns else np.ones(len(df), dtype=np.float32)
        )
        log_size = jnp.array(np.log(np.maximum(size_arr, 1e-6)), dtype=jnp.float32)
        pt_dum_np = self._encode_pack_type_dummies(df)
        pt_dum = jnp.array(pt_dum_np, dtype=jnp.float32)
        cust_idx_np = self._encode_customer_idx(df)
        cust_idx = jnp.array(cust_idx_np, dtype=jnp.int32)
        n_customer = max(len(self._customer_levels_), 1)

        y_arr = jnp.array(np.asarray(y), dtype=jnp.float32)
        cell_idx = jnp.array(cell_idx_np, dtype=jnp.int32)

        kernel = NUTS(self._model, target_accept_prob=0.9)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=True,
        )
        key = jrand.PRNGKey(self.random_state)
        mcmc.run(
            key,
            cell_idx,
            log_price,
            promo,
            wsin,
            wcos,
            log_units,
            log_size,
            pt_dum,
            cust_idx,
            n_customer,
            y_arr,
        )
        self.samples_ = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

        import arviz as az
        coords = {
            "brand": list(self._brand_codes_.keys()),
            "flavor": list(self._flavor_codes_.keys()),
            "pack": list(self._pack_codes_.keys()),
            "cell": [f"{b}||{f}||{p}" for (b, f, p) in self._cell_keys_],
        }
        dims = {
            "mu_brand": ["brand"],
            "mu_brand_raw": ["brand"],
            "alpha_flavor": ["flavor"],
            "alpha_flavor_raw": ["flavor"],
            "alpha_pack": ["pack"],
            "alpha_pack_raw": ["pack"],
            "beta_cell": ["cell"],
            "alpha_cell": ["cell"],
            "alpha_cell_raw": ["cell"],
            "eps_cell_raw": ["cell"],
            "mu_alpha_brand": ["brand"],
            "mu_alpha_brand_raw": ["brand"],
        }
        self.idata_ = az.from_numpyro(mcmc, coords=coords, dims=dims)

        # cache posterior means for predict / new-cell fallback
        self._alpha_cell_mean_ = self.samples_["alpha_cell"].mean(axis=0)
        self._beta_cell_mean_ = self.samples_["beta_cell"].mean(axis=0)
        self._mu_brand_mean_ = self.samples_["mu_brand"].mean(axis=0)
        self._alpha_flavor_mean_ = self.samples_["alpha_flavor"].mean(axis=0)
        self._alpha_pack_mean_ = self.samples_["alpha_pack"].mean(axis=0)
        self._mu_alpha_brand_mean_ = self.samples_["mu_alpha_brand"].mean(axis=0)
        self._gamma_mean_ = float(self.samples_["gamma_promo"].mean())
        self._delta_sin_mean_ = float(self.samples_["delta_sin"].mean())
        self._delta_cos_mean_ = float(self.samples_["delta_cos"].mean())
        self._theta_units_mean_ = float(self.samples_["theta_units"].mean())
        self._theta_size_mean_ = float(self.samples_["theta_size"].mean())
        if "theta_packtype" in self.samples_:
            self._theta_packtype_mean_ = self.samples_["theta_packtype"].mean(axis=0)
        else:
            self._theta_packtype_mean_ = np.zeros(0)
        if "theta_customer" in self.samples_:
            self._theta_customer_mean_ = self.samples_["theta_customer"].mean(axis=0)
        else:
            self._theta_customer_mean_ = np.zeros(max(n_customer, 1))
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def _lookup_cell_params(self, b: str, f: str, p: str) -> tuple[float, float]:
        """Return (alpha_cell, beta_cell) for (b,f,p).
        Unknown cell: fall back to additive prior mean (eps_cell = 0)."""
        key = (b, f, p)
        if key in self._cell_codes_:
            c = self._cell_codes_[key]
            return float(self._alpha_cell_mean_[c]), float(self._beta_cell_mean_[c])
        # new cell -> reconstruct from component means
        b_idx = self._brand_codes_.get(b)
        f_idx = self._flavor_codes_.get(f)
        p_idx = self._pack_codes_.get(p)
        if b_idx is None:
            # fully unknown brand -> global mean
            mu_b = float(self._mu_brand_mean_.mean())
            mu_ab = float(self._mu_alpha_brand_mean_.mean())
        else:
            mu_b = float(self._mu_brand_mean_[b_idx])
            mu_ab = float(self._mu_alpha_brand_mean_[b_idx])
        a_f = float(self._alpha_flavor_mean_[f_idx]) if f_idx is not None else 0.0
        a_p = float(self._alpha_pack_mean_[p_idx]) if p_idx is not None else 0.0
        beta_raw = mu_b + a_f + a_p
        beta = -float(np.exp(beta_raw))
        alpha = mu_ab  # sigma_alpha_cell * raw defaults to 0 for new cell
        return alpha, beta

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.samples_ is not None, "call fit() first"
        n = len(df)
        alphas = np.zeros(n, dtype=np.float64)
        betas = np.zeros(n, dtype=np.float64)
        brand_vals = df[BRAND_COL].values
        flavor_vals = df[FLAVOR_COL].values
        pack_vals = df[PACK_COL].values
        for i in range(n):
            a, b = self._lookup_cell_params(
                brand_vals[i], flavor_vals[i], pack_vals[i]
            )
            alphas[i] = a
            betas[i] = b

        log_price = df["log_price_per_litre"].to_numpy()
        promo = df["promotion_indicator"].to_numpy()
        wsin = df["week_sin"].to_numpy()
        wcos = df["week_cos"].to_numpy()
        log_units = (
            np.log1p(df[UNITS_COL].to_numpy().astype(float))
            if UNITS_COL in df.columns else np.zeros(n)
        )
        size_arr = (
            df[SIZE_COL].to_numpy().astype(float)
            if SIZE_COL in df.columns else np.ones(n)
        )
        log_size = np.log(np.maximum(size_arr, 1e-6))

        pt_dum = self._encode_pack_type_dummies(df)
        if pt_dum.shape[1] > 0 and self._theta_packtype_mean_.shape[0] == pt_dum.shape[1]:
            pt_contrib = pt_dum @ self._theta_packtype_mean_
        else:
            pt_contrib = np.zeros(n)

        cust_idx = self._encode_customer_idx(df)
        if self._theta_customer_mean_.shape[0] >= max(len(self._customer_levels_), 1):
            cust_contrib = self._theta_customer_mean_[cust_idx]
        else:
            cust_contrib = np.zeros(n)

        return (
            alphas
            + betas * log_price
            + self._gamma_mean_ * promo
            + self._delta_sin_mean_ * wsin
            + self._delta_cos_mean_ * wcos
            + self._theta_units_mean_ * log_units
            + self._theta_size_mean_ * log_size
            + pt_contrib
            + cust_contrib
        )

    # ------------------------------------------------------------------
    # Elasticity + diagnostics
    # ------------------------------------------------------------------
    def elasticity_posterior(self) -> pd.DataFrame:
        """Per-cell posterior summary of beta_cell (own-price elasticity on log scale).

        Columns: top_brand, flavor_internal, pack_tier, n_skus_in_cell,
                 beta_mean, beta_median, beta_hdi_low, beta_hdi_high, beta_std.
        """
        import arviz as az

        assert self.samples_ is not None, "call fit() first"
        beta = self.samples_["beta_cell"]  # (draws, n_cell)
        rows = []
        for (b, f, p), idx in self._cell_codes_.items():
            post = beta[:, idx]
            hdi = az.hdi(post, hdi_prob=0.95)
            rows.append(
                {
                    "top_brand": b,
                    "flavor_internal": f,
                    "pack_tier": p,
                    "n_skus_in_cell": int(self._cell_n_skus_.get((b, f, p), 0)),
                    "beta_mean": float(post.mean()),
                    "beta_median": float(np.median(post)),
                    "beta_hdi_low": float(hdi[0]),
                    "beta_hdi_high": float(hdi[1]),
                    "beta_std": float(post.std()),
                }
            )
        return pd.DataFrame(rows)

    def convergence_summary(self) -> pd.DataFrame:
        """Return az.summary for the key scalar / vector hyperparameters."""
        import arviz as az

        assert self.idata_ is not None, "call fit() first"
        var_names = [
            "mu_global",
            "mu_brand",
            "alpha_flavor",
            "alpha_pack",
            "sigma_brand",
            "sigma_flavor",
            "sigma_pack",
            "sigma_cell",
            "sigma_alpha_brand",
            "sigma_alpha_cell",
            "mu_alpha_global",
            "gamma_promo",
            "delta_sin",
            "delta_cos",
            "theta_units",
            "theta_size",
            "sigma_y",
        ]
        available = [
            v for v in var_names if v in self.idata_.posterior.data_vars
        ]
        return az.summary(self.idata_, var_names=available)

    def divergences(self) -> int:
        assert self.idata_ is not None, "call fit() first"
        return int(self.idata_.sample_stats.diverging.sum())
