"""Hierarchical Bayesian demand model (NumPyro).

Partial pooling of own-price elasticity:
    brand-level mean mu_b <- Normal(-2.0, 1.0)   (soft prior on soft drinks elasticity)
    sku elasticity beta_sku <- Normal(mu_b, sigma_b)
    log_nielsen_total_volume ~ Normal(alpha_sku + beta_sku * log_price + gamma * promo + season, sigma_y)

Prior sensitivity analysis is done via the ``prior_scale`` argument:
    'weak'     -> Normal(0, 5)  on brand mean, Exponential(1) on sigma_b
    'moderate' -> Normal(-2, 1) on brand mean, Exponential(2) on sigma_b  (default)
    'strong'   -> Normal(-2, 0.5) on brand mean, Exponential(5) on sigma_b

Heaviest file to run: expect 4-6 h on CPU for 923 panel units x 2 chains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


PRIOR_CONFIGS: dict[str, dict] = {
    "weak":     {"mu_loc": 0.0,  "mu_scale": 5.0,  "sigma_rate": 1.0},
    "moderate": {"mu_loc": -2.0, "mu_scale": 1.0,  "sigma_rate": 2.0},
    "strong":   {"mu_loc": -2.0, "mu_scale": 0.5,  "sigma_rate": 5.0},
}


@dataclass
class BayesianHierModel:
    prior_scale: Literal["weak", "moderate", "strong"] = "moderate"
    num_warmup: int = 1000
    num_samples: int = 2000
    num_chains: int = 2
    random_state: int = 42

    def __post_init__(self):
        self.samples_: dict | None = None
        self._brand_codes_: dict | None = None
        self._sku_codes_: dict | None = None
        self._sku_to_brand_: np.ndarray | None = None

    def _encode_indices(self, df: pd.DataFrame):
        brand_cat = pd.Categorical(df["top_brand"])
        sku_cat = pd.Categorical(df["product_sku_code"])
        self._brand_codes_ = {c: i for i, c in enumerate(brand_cat.categories)}
        self._sku_codes_ = {c: i for i, c in enumerate(sku_cat.categories)}

        sku_to_brand = (
            df.groupby("product_sku_code")["top_brand"].first().reindex(sku_cat.categories)
        )
        self._sku_to_brand_ = np.array(
            [self._brand_codes_[b] for b in sku_to_brand.values]
        )
        return brand_cat.codes, sku_cat.codes

    def _model(self, sku_idx, log_price, promo, week_sin, week_cos, y=None):
        import numpyro
        import numpyro.distributions as dist
        import jax.numpy as jnp

        pc = PRIOR_CONFIGS[self.prior_scale]
        n_brand = len(self._brand_codes_)
        n_sku = len(self._sku_codes_)
        sku_to_brand = jnp.array(self._sku_to_brand_)

        # brand-level mean elasticity
        mu_brand = numpyro.sample(
            "mu_brand",
            dist.Normal(pc["mu_loc"], pc["mu_scale"]).expand([n_brand]).to_event(1),
        )
        sigma_brand = numpyro.sample(
            "sigma_brand", dist.Exponential(pc["sigma_rate"])
        )
        # SKU elasticities with partial pooling
        beta_sku = numpyro.sample(
            "beta_sku",
            dist.Normal(mu_brand[sku_to_brand], sigma_brand).to_event(1),
        )
        # SKU intercepts (baseline log-volume)
        alpha_sku = numpyro.sample(
            "alpha_sku",
            dist.Normal(0.0, 5.0).expand([n_sku]).to_event(1),
        )
        gamma_promo = numpyro.sample("gamma_promo", dist.Normal(0.0, 1.0))
        delta_sin = numpyro.sample("delta_sin", dist.Normal(0.0, 1.0))
        delta_cos = numpyro.sample("delta_cos", dist.Normal(0.0, 1.0))
        sigma_y = numpyro.sample("sigma_y", dist.Exponential(1.0))

        mu = (
            alpha_sku[sku_idx]
            + beta_sku[sku_idx] * log_price
            + gamma_promo * promo
            + delta_sin * week_sin
            + delta_cos * week_cos
        )
        numpyro.sample("obs", dist.Normal(mu, sigma_y), obs=y)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "BayesianHierModel":
        import numpyro
        from numpyro.infer import MCMC, NUTS
        import jax
        import jax.numpy as jnp
        import jax.random as jrand

        _, sku_idx = self._encode_indices(df)

        log_price = jnp.array(df["log_price_per_litre"].to_numpy(), dtype=jnp.float32)
        promo = jnp.array(df["promotion_indicator"].to_numpy(), dtype=jnp.float32)
        wsin = jnp.array(df["week_sin"].to_numpy(), dtype=jnp.float32)
        wcos = jnp.array(df["week_cos"].to_numpy(), dtype=jnp.float32)
        y_arr = jnp.array(np.asarray(y), dtype=jnp.float32)
        sku_idx = jnp.array(np.asarray(sku_idx), dtype=jnp.int32)

        kernel = NUTS(self._model, target_accept_prob=0.9)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=True,
        )
        key = jrand.PRNGKey(self.random_state)
        mcmc.run(key, sku_idx, log_price, promo, wsin, wcos, y_arr)
        self.samples_ = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

        import arviz as az
        self.idata_ = az.from_numpyro(
            mcmc,
            coords={
                "sku": list(self._sku_codes_.keys()),
                "brand": list(self._brand_codes_.keys()),
            },
            dims={
                "beta_sku": ["sku"],
                "alpha_sku": ["sku"],
                "mu_brand": ["brand"],
                "sigma_brand": ["brand"],
            },
        )
        return self

    def elasticity_posterior(self) -> pd.DataFrame:
        """Return per-SKU elasticity: mean, median, 95% HDI low/high."""
        import arviz as az

        assert self.samples_ is not None, "call fit() first"
        beta = self.samples_["beta_sku"]  # shape (draws, n_sku)
        rows = []
        for sku, idx in self._sku_codes_.items():
            post = beta[:, idx]
            hdi = az.hdi(post, hdi_prob=0.95)
            rows.append(
                {
                    "product_sku_code": sku,
                    "beta_mean": float(post.mean()),
                    "beta_median": float(np.median(post)),
                    "beta_hdi_low": float(hdi[0]),
                    "beta_hdi_high": float(hdi[1]),
                    "beta_std": float(post.std()),
                }
            )
        return pd.DataFrame(rows)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Posterior predictive mean on log scale."""
        assert self.samples_ is not None, "call fit() first"
        n_sku = len(self._sku_codes_)
        sku_idx = np.array(
            [
                self._sku_codes_.get(s, 0)
                for s in df["product_sku_code"].values
            ]
        )
        log_price = df["log_price_per_litre"].to_numpy()
        promo = df["promotion_indicator"].to_numpy()
        wsin = df["week_sin"].to_numpy()
        wcos = df["week_cos"].to_numpy()

        alpha = self.samples_["alpha_sku"].mean(axis=0)  # (n_sku,)
        beta = self.samples_["beta_sku"].mean(axis=0)
        gamma = float(self.samples_["gamma_promo"].mean())
        d_sin = float(self.samples_["delta_sin"].mean())
        d_cos = float(self.samples_["delta_cos"].mean())

        return (
            alpha[sku_idx]
            + beta[sku_idx] * log_price
            + gamma * promo
            + d_sin * wsin
            + d_cos * wcos
        )
