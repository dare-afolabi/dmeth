#!/usr/bin/env python
# coding: utf-8


"""
Statistical engines for differential methylation analysis.

This module implements empirical Bayes moderated statistics following
Smyth (2004) limma methodology, with extensions for paired designs,
robust estimation, and Numba-accelerated computation. It provides both
full-matrix and chunked processing for memory-constrained environments.

Features
--------
- Empirical Bayes variance shrinkage (Smyth method)
- Robust variance estimation via winsorization
- Numba JIT compilation for 10-100 x speedup
- Memory-efficient chunked processing
- Paired and multi-group designs
- F-tests for multi-coefficient contrasts
- Automatic handling of missing data
- Group mean computation for interpretability
"""


from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.optimize import root_scalar
from scipy.special import digamma, polygamma
from statsmodels.stats.multitest import multipletests

from dmeth.core.analysis.validation import check_analysis_memory
from dmeth.utils.logger import logger

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not installed. Feature fitting will be slower.")


def _winsorize_array(
    arr: Union[np.ndarray, Sequence[float]], lower: float = 0.05, upper: float = 0.95
) -> np.ndarray:
    """
    Apply winsorization to a numeric array by clamping extreme values to \
    specified quantiles.

    Operates robustly in the presence of NaN/inf values: non-finite entries are \
    ignored during quantile computation but preserved in the output.

    Parameters
    ----------
    arr : np.ndarray or sequence of float
        Input data (1-D or 2-D).
    lower : float, default 0.05
        Lower quantile (e.g., 5th percentile) — values below are raised to this level.
    upper : float, default 0.95
        Upper quantile (e.g., 95th percentile) — values above are lowered to this level.

    Returns
    -------
    np.ndarray
        Winsorized array of identical shape and dtype, with NaNs preserved.

    Raises
    ------
    ValueError
        If ``0 ≤ lower < upper ≤ 1`` is violated.
    """
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("`lower` and `upper` must satisfy 0 <= lower < upper <= 1")

    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return arr.copy()

    # Compute quantiles ignoring NaNs
    lo = np.nanquantile(arr, lower)
    hi = np.nanquantile(arr, upper)

    # Clip values while preserving NaNs
    return np.where(np.isnan(arr), arr, np.clip(arr, lo, hi))


@dataclass
class SmythPrior:
    df_prior: float
    var_prior: float


def _estimate_smyth_prior(
    variances: np.ndarray, robust: bool = True, fallback_df: float = 1e6
) -> SmythPrior:
    """
    Estimate empirical Bayes prior degrees of freedom (d₀) and variance scale (s₀²) \
    using Smyth’s moment-matching method on log-variances.

    Closely follows limma’s implementation (Smyth, 2004). Uses root-finding on \
    the trigamma function; falls back to a large prior df if convergence fails.

    Parameters
    ----------
    variances : np.ndarray
        Positive, finite per-feature sample variances.
    robust : bool, default True
        Use median of log-variances instead of mean to reduce outlier influence.
    fallback_df : float, default 1e6
        Prior df returned when variance of log-variances is near zero.

    Returns
    -------
    SmythPrior
        Dataclass containing ``df_prior`` and ``var_prior``.
    """
    x = np.asarray(variances, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        raise ValueError("All variances are zero or non-finite.")
    if x.size == 1:
        return SmythPrior(df_prior=fallback_df, var_prior=float(x[0]))

    logx = np.log(x)
    mean_log = np.median(logx) if robust else np.mean(logx)
    var_log = np.var(logx, ddof=1)
    geom_mean = np.exp(mean_log)

    if var_log <= 0:
        return SmythPrior(df_prior=fallback_df, var_prior=float(geom_mean))

    def objective(t: float) -> float:
        return polygamma(1, t) - var_log

    try:
        sol = root_scalar(objective, bracket=[1e-8, 1e6], method="brentq")
        if not sol.converged:
            raise RuntimeError("Root finding failed to converge.")
        t = sol.root
    except Exception:
        return SmythPrior(df_prior=fallback_df, var_prior=float(geom_mean))

    df_prior = float(2 * t)
    var_prior = float(np.exp(mean_log - digamma(t) + np.log(t)))

    return SmythPrior(df_prior=df_prior, var_prior=var_prior)


def _moderated_variance(
    sample_var: np.ndarray, df_residual: int, df_prior: float, var_prior: float
) -> np.ndarray:
    """
    Compute moderated (shrunk) variances via empirical Bayes blending.

    Formula:
        s₂_post = (d₀·s₀² + df_resid·s²) / (d₀ + df_resid)

    Values are clamped to a tiny positive constant to avoid numerical instability.

    Parameters
    ----------
    sample_var : np.ndarray
        Observed per-feature variances.
    df_residual : int or np.ndarray
        Residual degrees of freedom per feature.
    df_prior, var_prior : float
        Empirical Bayes prior parameters.

    Returns
    -------
    np.ndarray
        Moderated variances (same shape as ``sample_var``).
    """
    sample_var = np.asarray(sample_var, dtype=float)
    df_residual = np.asarray(df_residual, dtype=float)  # allow per-feature df

    # df_residual must be positive for all features
    if np.any(df_residual <= 0):
        raise ValueError("df_residual must be > 0 for all features.")

    # df_prior is a scalar
    if not np.isfinite(df_prior) or df_prior <= 0:
        raise ValueError(
            "df_prior must be a positive finite scalar (check prior estimation)"
        )

    if not np.isfinite(var_prior) or var_prior <= 0:
        raise ValueError("var_prior must be a positive finite scalar")

    moderated = (df_prior * var_prior + df_residual * sample_var) / (
        df_prior + df_residual
    )

    # Clamp to small positive value to prevent zero/negative variance
    return np.maximum(moderated, 1e-12)


def _add_group_means(
    data: pd.DataFrame, group_labels: pd.Series, value_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Append group-wise mean columns (``mean_{group}``) to a wide or long DataFrame.

    Handles both wide-format methylation matrices (features × samples) \
    and long-format tidy data.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    group_labels : pd.Series
        Group membership aligned with samples (wide) or rows (long).
    value_column : str or None
        Required only for long-format data; specifies the value column.

    Returns
    -------
    pd.DataFrame
        Original data with additional ``mean_{group}`` columns.
    """
    group_labels = group_labels.astype("category")

    if value_column is None:
        # WIDE FORMAT
        if data.shape[1] != len(group_labels):
            return data

        if not data.columns.equals(group_labels.index):
            raise ValueError("wide: group_labels.index must match data.columns")

        group_means = {
            f"mean_{g}": data.loc[:, group_labels[group_labels == g].index].mean(axis=1)
            for g in group_labels.cat.categories
        }
        group_means_df = pd.DataFrame(group_means, index=data.index)

    else:
        # LONG FORMAT
        if value_column not in data.columns:
            raise ValueError(f"value_column '{value_column}' not in data")
        if data.shape[0] != len(group_labels):
            raise ValueError(f"long: {data.shape[0]=} != {len(group_labels)=}")
        if not data.index.equals(group_labels.index):
            raise ValueError("long: group_labels.index must match data.index")

        means_series = (
            data[value_column].groupby(group_labels, observed=True).transform("mean")
        )
        group_means_df = pd.DataFrame(
            {
                f"mean_{g}": means_series.where(group_labels == g)
                for g in group_labels.cat.categories
            },
            index=data.index,
        )

    return pd.concat([data, group_means_df], axis=1)


# If numba available, use JIT-compiled helper
if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, cache=True)
    def _fit_features_numba(Y, X, min_count, winsor_lower, winsor_upper, robust):
        """
        JIT-compiled feature-wise linear regression.

        Parameters
        ----------
        Y : np.ndarray (n_features, n_samples)
            Response matrix
        X : np.ndarray (n_samples, n_coef)
            Design matrix
        min_count : int
            Minimum observations per feature
        winsor_lower : float
            Lower quantile for winsorization
        winsor_upper : float
            Upper quantile for winsorization
        robust : bool
            Whether to winsorize

        Returns
        -------
        - beta_hat : np.ndarray (n_coef, n_features)
        - s2 : np.ndarray (n_features,)
        - df_obs : np.ndarray (n_features,)
        - n_obs : np.ndarray (n_features,)
        """
        n_features, n_samples = Y.shape
        n_coef = X.shape[1]

        beta_hat = np.full((n_coef, n_features), np.nan)
        s2 = np.full(n_features, np.nan)
        df_obs = np.full(n_features, np.nan)
        n_obs = np.zeros(n_features, dtype=np.int64)

        # Precompute (X^T X)^{-1}
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.pinv(XtX)
        except np.linalg.LinAlgError as e:
            logger.warning(f"Failed to compute (X^T X)^(-1): {e}")
            return beta_hat, s2, df_obs, n_obs

        for i in prange(n_features):
            y = Y[i, :]

            # Find non-missing values
            mask = ~np.isnan(y)
            n_present = np.sum(mask)

            if n_present < min_count:
                continue

            # Extract observed data
            X_obs = X[mask, :]
            y_obs = y[mask]

            # Winsorize if requested
            if robust and n_present > 2:
                lo = np.percentile(y_obs, winsor_lower * 100)
                hi = np.percentile(y_obs, winsor_upper * 100)
                y_obs = np.clip(y_obs, lo, hi)

            # Check variance
            if np.var(y_obs) < 1e-12:
                continue

            # Fit model
            try:
                rank_obs = np.linalg.matrix_rank(X_obs)
                if rank_obs < n_coef:
                    XtX_obs = X_obs.T @ X_obs
                    XtX_inv_obs = np.linalg.pinv(XtX_obs)
                    beta = XtX_inv_obs @ (X_obs.T @ y_obs)
                else:
                    beta = XtX_inv @ (X.T @ y)  # Use precomputed if full rank

                beta_hat[:, i] = beta

                # Compute residuals
                fitted = X_obs @ beta
                resid = y_obs - fitted

                df_i = n_present - rank_obs
                if df_i > 0:
                    s2[i] = np.sum(resid**2) / df_i
                    df_obs[i] = df_i
                    n_obs[i] = n_present
            except (np.linalg.LinAlgError, ValueError) as e:
                # Optional: log the failure
                logger.warning(f"Sample {i} failed in regression: {e}")
                continue

        return beta_hat, s2, df_obs, n_obs


def fit_differential(
    M: pd.DataFrame,
    design: pd.DataFrame,
    contrast: Optional[np.ndarray] = None,
    contrast_matrix: Optional[np.ndarray] = None,
    shrink: Union[str, float] = "auto",
    robust: bool = True,
    eps: float = 1e-8,
    return_residuals: bool = False,
    min_count: int = 3,
    max_d0: float = 50.0,
    winsor_lower: float = 0.05,
    winsor_upper: float = 0.95,
    group_labels: Optional[pd.Series] = None,
    use_numba: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Core limma-style differential methylation analysis with empirical Bayes moderation.

    Features:

    - Numba-accelerated feature-wise fitting (10–100× speedup when no missing data)
    - Automatic handling of missing values
    - Robust winsorization of residuals
    - Multiple shrinkage strategies (Smyth, median, fixed d₀, none)
    - Single-contrast t-tests or multi-coefficient F-tests
    - Optional group mean beta/M-value columns for interpretability
    - Optional residuals return for diagnostics

    Parameters
    ----------
    M : pd.DataFrame
        CpG × sample matrix of M-values (or beta values if already on log-odds scale).
    design : pd.DataFrame
        Sample × covariate design matrix (output from ``patsy`` or manual construction).
    contrast / contrast_matrix : np.ndarray or None
        Contrast vector (t-test) or matrix (F-test).
    shrink : {"auto", "smyth", "median", "none"} or float
        Variance shrinkage method or fixed prior df.
    robust : bool, default True
        Winsorize residuals per feature.
    group_labels : pd.Series, optional
        Explicit sample grouping for mean columns; otherwise heuristically detected.
    use_numba : bool, default True
        Enable JIT-compiled loop when possible.

    Returns
    -------
    pd.DataFrame
        Results table with columns including ``logFC``, ``t``, ``pval``, ``padj`` \
        (or ``F``, ``df1``, ``df2``), moderated variances (``s2_post``), prior df \
        (``d0``), and group means.
    (pd.DataFrame, pd.DataFrame), optional
        Second element contains per-feature residuals if ``return_residuals=True``.
    """
    check_analysis_memory(M)

    if not isinstance(M, pd.DataFrame) or not isinstance(design, pd.DataFrame):
        raise TypeError("M and design must be pandas DataFrames")

    Y = M.values.astype(float)
    X = design.values.astype(float)
    n_samples, p_full = X.shape
    n_features = Y.shape[0]

    if n_samples != Y.shape[1]:
        raise ValueError("M columns != design rows")

    p = np.linalg.matrix_rank(X)
    if p >= n_samples:
        raise ValueError(
            f"design is singular or overparameterized (rank={p}, n={n_samples})"
        )

    cond = np.linalg.cond(X)
    if cond > 1e10:
        logger.warning(f"Design matrix is ill-conditioned (κ={cond:.2e})")

    min_count = max(min_count, p + 1)

    # FAST PATH: Use Numba if available and no missing data
    if use_numba and NUMBA_AVAILABLE and not np.any(np.isnan(Y)):
        logger.progress("Using Numba-accelerated fitting...")
        beta_hat, s2, df_obs, n_obs = _fit_features_numba(
            Y, X, min_count, winsor_lower, winsor_upper, robust
        )
        residuals = None  # Skip residuals for speed unless requested

    # SLOW PATH: Python loop
    else:
        if np.any(np.isnan(Y)):
            logger.progress("Missing data detected, using slower Python loop...")
        else:
            logger.progress("Numba unavailable, using slower Python loop...")

        XtX_inv_full = linalg.pinv(X.T @ X)
        beta_hat = np.full((p, n_features), np.nan)
        s2 = np.full(n_features, np.nan)
        df_obs = np.full(n_features, np.nan)
        n_obs = np.zeros(n_features, dtype=int)
        residuals = np.full_like(Y, np.nan) if return_residuals else None

        for i in range(n_features):
            y = Y[i, :]
            mask = ~np.isnan(y)
            n_present = mask.sum()

            if n_present < n_samples * 0.5:
                logger.warning(f"Feature {i} has only {n_present}/{n_samples} samples")
            if n_present < min_count:
                continue

            X_obs = X[mask, :]
            y_obs = y[mask]

            if robust:
                y_obs = _winsorize_array(y_obs, winsor_lower, winsor_upper)

            if np.var(y_obs) < 1e-12:
                continue

            try:
                rank_obs = np.linalg.matrix_rank(X_obs)
                XtX_inv = linalg.pinv(X_obs.T @ X_obs) if rank_obs < p else XtX_inv_full

                beta = XtX_inv @ (X_obs.T @ y_obs)
                beta_hat[:, i] = beta

                fitted = X_obs @ beta
                resid = y_obs - fitted

                if return_residuals:
                    residuals[i, mask] = resid

                df_i = n_present - rank_obs
                if df_i > 0:
                    s2[i] = np.sum(resid**2) / df_i
                    df_obs[i] = df_i
                    n_obs[i] = n_present
            except Exception as e:
                logger.warning(f"Fitting failed for sample {i}: {e}")

    logger.info("Fitting complete, applying shrinkage...")

    # Continue with variance shrinkage
    valid = np.isfinite(s2) & (df_obs > 0)
    if not valid.any():
        raise ValueError("No features passed fitting criteria")

    beta_hat = beta_hat[:, valid]
    s2 = s2[valid]
    df_obs = df_obs[valid]
    M_valid = M.iloc[valid]
    # residuals_valid = residuals[valid, :]

    # Shrinkage
    if shrink == "auto":
        if n_samples < 10:
            shrink = "none"
        elif n_samples < 30:
            log_s2 = np.log(np.maximum(s2, 1e-12))
            shrink = 10.0 if np.var(log_s2, ddof=1) < 0.05 else "smyth"
        else:
            shrink = "smyth"

    if isinstance(shrink, (int, float)) and shrink > 0:
        d0 = min(float(shrink), max_d0)
        s2_win = _winsorize_array(s2, winsor_lower, winsor_upper) if robust else s2
        s0sq = np.median(s2_win)
        s2_post = _moderated_variance(s2, df_obs, d0, s0sq)
        df_total = df_obs + d0
    elif shrink == "median":
        s2_win = _winsorize_array(s2, winsor_lower, winsor_upper) if robust else s2
        s0sq = np.median(s2_win)
        d0 = min(max(2.0, n_samples / 2.0), max_d0)
        s2_post = _moderated_variance(s2, df_obs, d0, s0sq)
        df_total = df_obs + d0
    elif shrink == "smyth":
        prior = _estimate_smyth_prior(s2, robust=robust)
        d0 = min(prior.df_prior, max_d0)
        s2_post = _moderated_variance(s2, df_obs, d0, prior.var_prior)
        df_total = df_obs + d0
    elif shrink == "none":
        d0 = 0.0
        s2_post = s2.copy()
        df_total = df_obs
    else:
        raise ValueError(f"Invalid shrink: {shrink}")

    # Contrast handling
    if contrast is not None and contrast_matrix is not None:
        raise ValueError("Specify either contrast or contrast_matrix, not both")

    res_dict = {
        "s2": s2,
        "s2_post": s2_post,
        "d0": np.full_like(s2, d0),
        "n_obs": n_obs[valid],
        "df_resid": df_obs,
        "df_total": df_total,
    }

    if contrast is not None:
        c = np.asarray(contrast, dtype=float).ravel()
        logFC = c @ beta_hat
        cc = c @ XtX_inv_full @ c
        se_post = np.sqrt(np.maximum(cc * s2_post, eps))
        t_stat = np.where(se_post < 1e-3, 0, logFC / se_post)
        pvals = 2 * stats.t.sf(np.abs(t_stat), df_total)
        _, padj, _, _ = multipletests(pvals, method="fdr_bh")
        res_dict.update(
            {
                "logFC": logFC,
                "se": se_post,
                "t": t_stat,
                "pval": pvals,
                "padj": padj,
            }
        )

    elif contrast_matrix is not None:
        R = np.asarray(contrast_matrix, dtype=float)
        r = np.linalg.matrix_rank(R)
        RVR_inv = linalg.pinv(R @ XtX_inv_full @ R.T)
        CB = R @ beta_hat
        numer = np.sum(CB * (RVR_inv @ CB), axis=0)
        F_stat = numer / r / np.maximum(s2_post, eps)
        pvals = stats.f.sf(F_stat, r, df_total)
        _, padj, _, _ = multipletests(pvals, method="fdr_bh")
        res_dict.update(
            {
                "F": F_stat,
                "pval": pvals,
                "padj": padj,
                "df1": np.full_like(F_stat, r),
                "df2": df_total,
            }
        )

    res = pd.DataFrame(res_dict, index=M_valid.index)
    if "pval" in res.columns:
        res = res.sort_values("pval")

    # Group means
    if group_labels is None:
        # Heuristic: pick first 2-level column
        for col in design.columns:
            if design[col].nunique() == 2:
                group_labels = design[col]
                break

    if group_labels is not None:
        group_labels = group_labels.reindex(M_valid.columns)
        if group_labels.isnull().any():
            raise ValueError("group_labels must cover all samples in M_valid.columns")

        group_means_full = _add_group_means(M_valid, group_labels)
        mean_cols = [c for c in group_means_full.columns if str(c).startswith("mean_")]

        if mean_cols:
            res = pd.concat([res, group_means_full[mean_cols]], axis=1)

    if return_residuals:
        resid_df = pd.DataFrame(residuals, index=M.index, columns=M.columns)
        return res, resid_df
    return res


def fit_differential_chunked(
    M: pd.DataFrame,
    design: pd.DataFrame,
    chunk_size: int = 10000,
    verbose: bool = True,
    group_labels: Optional[pd.Series] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Memory-efficient version of ``fit_differential`` that processes features in chunks.

    Useful for >1M probe arrays on limited RAM. P-values are combined \
    globally and FDR-corrected once at the end.

    Parameters
    ----------
    chunk_size : int, default 10000
        Number of CpGs per processing chunk.
    **kwargs
        All additional arguments forwarded to ``fit_differential``.

    Returns
    -------
    pd.DataFrame
        Complete results table with globally adjusted p-values and group means.
    """
    check_analysis_memory(M)

    chunk_size = int(chunk_size)
    n_features = len(M)
    n_chunks = int(np.ceil(n_features / chunk_size))
    results = []

    if verbose:
        logger.info(f"Processing {n_features:,} features in {n_chunks} chunks...")

    start = time.time()
    for i in range(n_chunks):
        idx = slice(i * chunk_size, (i + 1) * chunk_size)
        try:
            res_chunk = fit_differential(
                M.iloc[idx], design, group_labels=group_labels, **kwargs
            )
            results.append(res_chunk)
        except Exception as e:
            logger.warning(f"Chunk {i+1} failed: {e}")

        if verbose and (i + 1) % max(1, n_chunks // 10) == 0:
            elapsed = time.time() - start
            logger.info(f"  {i+1}/{n_chunks} | {elapsed:.1f}s")

    if not results:
        raise RuntimeError("All chunks failed")

    combined = pd.concat(results)
    if "pval" in combined.columns:
        combined["padj"] = multipletests(combined["pval"], method="fdr_bh")[1]
        combined = combined.sort_values("pval")

    if group_labels is not None:
        combined = _add_group_means(combined, group_labels)

    if verbose:
        logger.info(f"Finished in {time.time() - start:.1f}s")
    return combined
