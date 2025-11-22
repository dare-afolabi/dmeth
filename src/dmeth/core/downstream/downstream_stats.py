#!/usr/bin/env python
# coding: utf-8


"""
Core statistical utilities for DNA methylation analysis downstream of \
differential testing.

- This module provides a comprehensive suite of high-performance, publication-grade \
functions for p-value adjustment, effect-size computation, delta-beta calculation, \
reproducible DMR calling via sliding-window clustering, regional summarization, \
and cross-dataset reproducibility assessment.
- Designed for seamless integration into \
DMeth pipelines, all functions operate efficiently on large epigenome-wide \
datasets while preserving genomic coordinates and CpG identifiers.

Features
--------
- Multiple p-value correction methods via statsmodels with robust NaN handling
- Stouffer’s Z-score method for meta-analysis across studies or batches
- Cohen’s d and Hedges’ g effect size estimation with proper small-sample correction
- Flexible delta-beta computation with optional absolute values and index alignment
- Threshold-based filtering of DMS results supporting logFC, delta-beta, \
and directional constraints
- Fast, vectorized sliding-window DMR discovery with configurable gap merging \
and minimum CpG requirements
- Comprehensive DMR summarization and cross-dataset reproducibility metrics \
(Jaccard, concordance, Spearman correlation)
- Full preservation of CpG identifiers and genomic coordinates throughout all operations
"""


from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest

from dmeth.core.downstream.helpers import _clean_chr


def adjust_pvalues(
    pvals: Union[pd.Series, np.ndarray, List[float]],
    method: str = "fdr_bh",
) -> pd.Series:
    """
    Apply multiple-testing correction to raw p-values using statsmodels.

    Handles NaN values robustly (treated as non-significant) and preserves the \
    original index when input is a pandas Series.

    Parameters
    ----------
    pvals : array-like
        Raw p-values (0 ≤ p ≤ 1).
    method : str, default "fdr_bh"
        Correction method passed to ``statsmodels.stats.multitest.multipletests``.
        Supported: ``"bonferroni"``, ``"holm"``, ``"fdr_bh"``, ``"fdr_by"``, \
        ``"sidak"``, ``"none"``.

    Returns
    -------
    pd.Series
        Adjusted p-values with identical index/order as input.
    """
    if pvals is None:
        raise ValueError("pvals cannot be None")
    is_series = isinstance(pvals, pd.Series)
    idx = pvals.index if is_series else None
    arr = np.asarray(pvals, dtype=float)
    arr[np.isnan(arr)] = 1.0
    if method == "none":
        adj = arr.copy()
    else:
        try:
            rej, adj, _, _ = multitest.multipletests(arr, method=method)
        except Exception as e:
            raise ValueError(f"P-value adjustment failed: {e}")
    result = pd.Series(adj, index=idx) if is_series else pd.Series(adj)
    return result


def stouffer_combined_pvalue(pvals):
    """
    Combine independent p-values across studies or replicates using Stouffer’s \
    Z-score method (equal weighting).

    Parameters
    ----------
    pvals : array-like
        List or array of p-values to combine.

    Returns
    -------
    float
        Single combined p-value.

    Raises
    ------
    ValueError
        If input is empty or contains values outside [0, 1].
    """
    pvals = np.asarray(pvals, dtype=float)
    if len(pvals) == 0:
        raise ValueError("pvals cannot be empty")
    if np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("pvals must be between 0 and 1")

    z_scores = stats.norm.ppf(1 - pvals)
    combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
    return stats.norm.sf(combined_z)


def compute_delta_beta(
    mean_beta_group1: Union[pd.Series, np.ndarray],
    mean_beta_group2: Union[pd.Series, np.ndarray],
    as_abs: bool = False,
) -> pd.Series:
    """
    Calculate per-CpG difference in mean beta values between two groups.

    Automatically aligns inputs by index; supports absolute differences.

    Parameters
    ----------
    mean_beta_group1, mean_beta_group2 : Series-like
        Group-wise mean beta values.
    as_abs : bool, default False
        Return absolute delta-beta if True.

    Returns
    -------
    pd.Series
        Delta-beta values (positive = group1 > group2).
    """
    if mean_beta_group1 is None or mean_beta_group2 is None:
        raise ValueError("Both group means must be provided")
    s1 = pd.Series(mean_beta_group1)
    s2 = pd.Series(mean_beta_group2)
    if not s1.index.equals(s2.index):
        # try align by index labels; if not possible, align by position
        try:
            s2 = s2.reindex(index=s1.index)
        except Exception:
            s2 = pd.Series(np.asarray(s2), index=s1.index)
    delta = s1 - s2
    if as_abs:
        delta = delta.abs()
    return delta


def compute_effect_size(
    beta_group1: pd.DataFrame, beta_group2: pd.DataFrame, method: str = "cohens_d"
) -> pd.Series:
    """
    Compute standardized effect size (Cohen’s d or Hedges’ g) for each CpG \
    between two groups.

    Uses pooled standard deviation with proper small-sample correction for Hedges’ g.

    Parameters
    ----------
    beta_group1, beta_group2 : pd.DataFrame
        Beta matrices (CpGs × samples) for each group.
    method : {"cohens_d", "hedges_g"}, default "cohens_d"

    Returns
    -------
    pd.Series
        Effect size per CpG (positive = higher in group1).
    """
    if (
        beta_group1 is None
        or beta_group2 is None
        or beta_group1.empty
        or beta_group2.empty
    ):
        return pd.Series(dtype=float)

    m1 = beta_group1.mean(axis=1)
    m2 = beta_group2.mean(axis=1)
    s1 = beta_group1.std(axis=1, ddof=1)
    s2 = beta_group2.std(axis=1, ddof=1)
    n1 = beta_group1.shape[1]
    n2 = beta_group2.shape[1]

    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(n1 + n2 - 2, 1))
    pooled_sd = pooled_sd.replace(0, np.nan)
    d = (m1 - m2) / pooled_sd

    if method.lower() == "hedges_g":
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        d *= correction

    return d.fillna(0.0)


def filter_dms(
    res: pd.DataFrame,
    lfc_col: str = "logFC",
    pval_col: str = "padj",
    delta_beta_col: Optional[str] = None,
    pval_thresh: float = 0.05,
    lfc_thresh: float = 0.0,
    delta_beta_thresh: Optional[float] = None,
    direction: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply multi-criterion filtering to differential methylation results.

    Supports:

    - Adjusted p-value threshold
    - Minimum |logFC|
    - Minimum |Δβ|
    - Directional filtering (“hyper” or “hypo”)

    Parameters
    ----------
    res : pd.DataFrame
        Differential results table.
    lfc_col, pval_col, delta_beta_col : str
        Column names (defaults: "logFC", "padj", optional).
    pval_thresh, lfc_thresh, delta_beta_thresh
        Numeric thresholds.
    direction : {"hyper", "hypo", None}

    Returns
    -------
    pd.DataFrame
        Subset of rows passing all specified criteria.
    """
    if res is None or res.empty:
        return res.copy() if isinstance(res, pd.DataFrame) else pd.DataFrame()
    df = res.copy()
    if pval_col not in df.columns:
        raise KeyError(f"{pval_col} not in results")
    mask = df[pval_col].fillna(1.0) <= pval_thresh

    if lfc_col in df.columns:
        if direction == "hyper":
            mask &= df[lfc_col] >= lfc_thresh
        elif direction == "hypo":
            mask &= df[lfc_col] <= -lfc_thresh
        else:
            mask &= df[lfc_col].abs() >= lfc_thresh

    if (
        delta_beta_col
        and delta_beta_col in df.columns
        and delta_beta_thresh is not None
    ):
        mask &= df[delta_beta_col].abs() >= float(delta_beta_thresh)

    return df.loc[mask]


def find_dmrs_by_sliding_window(
    dms: pd.DataFrame,
    annotation: pd.DataFrame,
    chr_col: str = "chr",
    pos_col: str = "pos",
    pval_col: str = "padj",
    pval_thresh: float = 0.05,
    delta_beta_col: Optional[str] = "delta_beta",
    max_gap: int = 500,
    min_cpgs: int = 3,
    merge_distance: Optional[int] = None,
    use_intervaltree: bool = True,
) -> pd.DataFrame:
    """
    Identify differentially methylated regions (DMRs) by clustering spatially \
    proximate significant CpGs.

    Uses a fast sliding-window/gap-merging approach per chromosome.

    Parameters
    ----------
    dms : pd.DataFrame
        Significant DMS results (after filtering).
    annotation : pd.DataFrame
        CpG annotation with ``chr`` and ``pos`` columns.
    max_gap : int, default 500
        Maximum distance (bp) to bridge adjacent significant CpGs.
    min_cpgs : int, default 3
        Minimum number of significant CpGs required to call a DMR.
    merge_distance : int or None
        If set, merge DMRs closer than this distance.

    Returns
    -------
    pd.DataFrame
        One row per DMR with columns:
        ``chr``, ``start``, ``end``, ``n_cpgs``, ``mean_delta_beta``, \
        ``mean_logFC``, ``min_padj``, ``cpgs`` (list).
    """
    if dms is None or dms.empty:
        return pd.DataFrame(
            columns=[
                "chr",
                "start",
                "end",
                "n_cpgs",
                "mean_delta_beta",
                "mean_logFC",
                "min_padj",
                "cpgs",
            ]
        )

    # join annotation
    ann = annotation.copy()
    if ann.index.dtype != object:
        ann.index = ann.index.astype(str)
    joined = dms.join(ann[[chr_col, pos_col]], how="inner")
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "chr",
                "start",
                "end",
                "n_cpgs",
                "mean_delta_beta",
                "mean_logFC",
                "min_padj",
                "cpgs",
            ]
        )

    # filter significant
    sig = joined[joined[pval_col].fillna(1.0) <= pval_thresh].copy()
    if sig.empty:
        return pd.DataFrame(
            columns=[
                "chr",
                "start",
                "end",
                "n_cpgs",
                "mean_delta_beta",
                "mean_logFC",
                "min_padj",
                "cpgs",
            ]
        )

    sig[chr_col] = sig[chr_col].apply(_clean_chr)
    regions = []
    for chrom, grp in sig.groupby(chr_col):
        grp_sorted = grp.sort_values(pos_col)
        positions = grp_sorted[pos_col].values.astype(int)
        cpg_ids = grp_sorted.index.to_numpy()
        # form clusters by gaps
        breaks = np.where(np.diff(positions) > max_gap)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(positions) - 1]))
        for s, e in zip(starts, ends):
            idxs = slice(s, e + 1)
            block_ids = list(cpg_ids[idxs])
            if len(block_ids) < min_cpgs:
                continue
            subset = grp_sorted.iloc[idxs]
            start_pos = int(subset[pos_col].min())
            end_pos = int(subset[pos_col].max())
            mean_db = (
                float(subset[delta_beta_col].mean())
                if delta_beta_col in subset.columns
                else np.nan
            )
            mean_lfc = (
                float(subset["logFC"].mean()) if "logFC" in subset.columns else np.nan
            )
            min_p = float(subset[pval_col].min())
            regions.append(
                {
                    "chr": chrom,
                    "start": start_pos,
                    "end": end_pos,
                    "n_cpgs": len(block_ids),
                    "mean_delta_beta": mean_db,
                    "mean_logFC": mean_lfc,
                    "min_padj": min_p,
                    "cpgs": block_ids,
                }
            )

    # optional merge proximate regions
    if merge_distance and regions:
        regions_df = pd.DataFrame(regions).sort_values(["chr", "start"])
        merged = []
        cur = regions_df.iloc[0].to_dict()
        for r in regions_df.iloc[1:].to_dict(orient="records"):
            if r["chr"] == cur["chr"] and r["start"] - cur["end"] <= merge_distance:
                # merge
                cur["end"] = max(cur["end"], r["end"])
                cur["n_cpgs"] += r["n_cpgs"]
                cur["mean_delta_beta"] = np.nanmean(
                    [cur["mean_delta_beta"], r["mean_delta_beta"]]
                )
                cur["mean_logFC"] = np.nanmean([cur["mean_logFC"], r["mean_logFC"]])
                cur["min_padj"] = min(cur["min_padj"], r["min_padj"])
                cur["cpgs"].extend(r["cpgs"])
            else:
                merged.append(cur)
                cur = r.copy()
        merged.append(cur)
        regions = merged

    return pd.DataFrame(regions)


def summarize_regions(
    dmrs: pd.DataFrame, summary_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate a concise summary table of discovered DMRs.

    Reports total count, median length, average CpG density, mean delta-beta, \
    and strongest significance.

    Parameters
    ----------
    dmrs : pd.DataFrame
        Output from ``find_dmrs_by_sliding_window``.
    summary_cols : list[str] or None
        Columns to summarize (defaults to key metrics).

    Returns
    -------
    pd.DataFrame
        Single-row summary suitable for manuscript tables.
    """
    if dmrs is None or dmrs.empty:
        return pd.DataFrame(
            {
                "n_regions": [0],
                "median_length": [0],
                "mean_n_cpgs": [0.0],
                "mean_delta_beta": [0.0],
                "min_padj": [np.nan],
            }
        )

    summary_cols = summary_cols or ["delta_beta", "logFC", "padj"]
    lengths = dmrs["end"].astype(int) - dmrs["start"].astype(int)
    n_cpgs_col = "n_cpgs" if "n_cpgs" in dmrs.columns else None
    mean_delta_beta_col = (
        "mean_delta_beta" if "mean_delta_beta" in dmrs.columns else None
    )
    min_padj_col = "min_padj" if "min_padj" in dmrs.columns else None

    return pd.DataFrame(
        [
            {
                "n_regions": len(dmrs),
                "median_length": int(np.median(lengths)),
                "mean_n_cpgs": float(dmrs[n_cpgs_col].mean()) if n_cpgs_col else np.nan,
                "mean_delta_beta": (
                    float(dmrs[mean_delta_beta_col].mean())
                    if mean_delta_beta_col
                    else np.nan
                ),
                "min_padj": float(dmrs[min_padj_col].min()) if min_padj_col else np.nan,
            }
        ]
    )


def compute_dms_reproducibility(
    res1: pd.DataFrame,
    res2: pd.DataFrame,
    id_col: Optional[str] = None,
    effect_col: str = "logFC",
    pval_col: str = "padj",
    pval_thresh: float = 0.05,
) -> Dict[str, Any]:
    """
    Quantify reproducibility of differential methylation signals across \
    two independent analyses or cohorts.

    Metrics include:

    - Total/overlap feature counts
    - Jaccard index
    - Number of overlapping significant CpGs
    - Directional concordance
    - Spearman correlation of effect sizes

    Parameters
    ----------
    res1, res2 : pd.DataFrame
        Two differential result tables (same CpG index preferred).
    effect_col, pval_col : str
        Columns for effect size and adjusted p-value.
    pval_thresh : float, default 0.05

    Returns
    -------
    dict
        Comprehensive reproducibility statistics.
    """
    if (res1 is None or res1.empty) or (res2 is None or res2.empty):
        return {
            "overlap": 0,
            "jaccard": 0.0,
            "concordance": np.nan,
            "effect_corr": np.nan,
        }

    idx1 = set(res1.index.astype(str))
    idx2 = set(res2.index.astype(str))
    overlap = idx1 & idx2
    jaccard = float(len(overlap) / len(idx1 | idx2)) if (len(idx1 | idx2) > 0) else 0.0

    sig1 = set(res1[res1.get(pval_col, 1.0) < pval_thresh].index.astype(str))
    sig2 = set(res2[res2.get(pval_col, 1.0) < pval_thresh].index.astype(str))
    sig_overlap = sig1 & sig2

    # concordance: direction agreement among overlapping significant
    if sig_overlap:
        e1 = res1.loc[sorted(sig_overlap)][effect_col].values
        e2 = res2.loc[sorted(sig_overlap)][effect_col].values
        concord = np.mean(np.sign(e1) == np.sign(e2))
        corr = (
            float(stats.spearmanr(e1, e2).correlation) if len(e1) > 2 else float(np.nan)
        )
    else:
        concord = np.nan
        corr = np.nan

    return {
        "n_features_res1": len(idx1),
        "n_features_res2": len(idx2),
        "n_overlap": len(overlap),
        "jaccard": jaccard,
        "n_sig_overlap": len(sig_overlap),
        "concordance": concord,
        "effect_spearman_r": corr,
    }
