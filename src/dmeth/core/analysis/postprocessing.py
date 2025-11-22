#!/usr/bin/env python
# coding: utf-8


"""
Postprocessing and interpretation utilities for differential methylation \
analysis results.

This module provides robust, publication-ready tools for summarizing, filtering, \
and extracting biologically meaningful differentially methylated CpGs from \
statistical output. It supports flexible significance criteria, effect-size \
thresholding, directional filtering, and delta-beta constraints, with comprehensive \
summary statistics for reporting.

Features
--------
- Comprehensive summary statistics including significance counts, directionality, \
effect sizes, and empirical Bayes shrinkage metrics
- Flexible extraction of significant CpGs with adjustable logFC, adjusted p-value, \
and delta-beta thresholds
- Directional filtering for hyper- and hypomethylated sites
- Automatic detection or explicit specification of group mean beta-value columns \
for delta-beta filtering
- Graceful handling of missing columns and empty result sets with informative warnings
- Optional detailed summary dictionaries for downstream reporting or visualization
"""


from typing import Dict, List, Optional, Union

import pandas as pd

from dmeth.utils.logger import logger


def summarize_differential_results(
    res: pd.DataFrame, pval_thresh: float = 0.05, verbose: bool = True
) -> Dict[str, Union[int, float]]:
    """
    Produce a comprehensive, publication-ready summary of differential \
    methylation results.

    Handles missing columns gracefully and returns key metrics for reporting:
    significance counts, directionality, effect-size statistics, and \
    variance-shrinkage diagnostics.

    Parameters
    ----------
    res : pd.DataFrame
        Differential methylation results containing at minimum ``logFC`` and ``padj``.
        May also include ``pval``, ``s2``, ``s2_post``, and ``d0`` \
        (prior degrees of freedom).
    pval_thresh : float, default 0.05
        Adjusted p-value threshold defining statistical significance.
    verbose : bool, default True
        Emit warnings for empty input or missing optional columns.

    Returns
    -------
    dict
        Summary dictionary with the following keys:
        ``total_tested``: total CpGs tested
        ``significant``: number of significant CpGs (padj < threshold)
        ``pct_significant``: percentage of significant CpGs
        ``hypermethylated`` / ``hypomethylated``: directional counts
        ``mean_abs_logFC_sig`` / ``median_abs_logFC_sig``: \
        effect size summaries among significant sites
        ``max_abs_logFC``: largest absolute log fold change
        ``min_pval``: smallest raw p-value
        ``shrinkage_factor``: median ratio of moderated to original variance
        ``d0``: median prior degrees of freedom (empirical Bayes)

    Notes
    -----
    Missing optional columns are safely ignored with fallback values.
    Empty input returns a zero-filled summary.
    """
    if not (0 < pval_thresh <= 1):
        raise ValueError("pval_thresh must be in (0, 1]")

    if res.empty:
        if verbose:
            logger.warning("Input DataFrame is empty; returning zeros")
        return {
            "total_tested": 0,
            "significant": 0,
            "pct_significant": 0.0,
            "hypermethylated": 0,
            "hypomethylated": 0,
            "mean_abs_logFC_sig": 0.0,
            "median_abs_logFC_sig": 0.0,
            "max_abs_logFC": 0.0,
            "min_pval": 1.0,
            "shrinkage_factor": 1.0,
            "d0": 0.0,
        }

    # Column-safe extraction
    padj = (
        res["padj"].fillna(1.0)
        if "padj" in res.columns
        else pd.Series(1.0, index=res.index)
    )
    sig = res[padj < pval_thresh]

    logFC = (
        res["logFC"].fillna(0.0)
        if "logFC" in res.columns
        else pd.Series(0.0, index=res.index)
    )
    s2 = (
        res["s2"].fillna(1.0)
        if "s2" in res.columns
        else pd.Series(1.0, index=res.index)
    )
    s2_post = res["s2_post"].fillna(s2) if "s2_post" in res.columns else s2
    d0 = (
        res["d0"].fillna(0.0)
        if "d0" in res.columns
        else pd.Series(0.0, index=res.index)
    )

    s2_med = s2.median()
    shrinkage = s2_post.median() / s2_med if s2_med > 0 else 1.0

    summary = {
        "total_tested": len(res),
        "significant": len(sig),
        "pct_significant": (len(sig) / len(res) * 100.0) if len(res) > 0 else 0.0,
        "hypermethylated": sig[logFC > 0].shape[0] if not sig.empty else 0,
        "hypomethylated": sig[logFC < 0].shape[0] if not sig.empty else 0,
        "mean_abs_logFC_sig": sig["logFC"].abs().mean() if not sig.empty else 0.0,
        "median_abs_logFC_sig": sig["logFC"].abs().median() if not sig.empty else 0.0,
        "max_abs_logFC": logFC.abs().max(),
        "min_pval": res["pval"].min() if "pval" in res.columns else 1.0,
        "shrinkage_factor": shrinkage,
        "d0": float(d0.median()),
    }

    return summary


def get_significant_cpgs(
    res: pd.DataFrame,
    lfc_col: str = "logFC",
    pval_col: str = "padj",
    lfc_thresh: float = 0.0,
    pval_thresh: float = 0.05,
    delta_beta_thresh: Optional[float] = None,
    direction: Optional[str] = None,
    delta_beta_cols: Optional[List[str]] = None,
    return_summary: bool = False,
    verbose: bool = True,
) -> Union[List[str], Dict]:
    """
    Extract biologically meaningful significant CpGs using flexible, \
    multi-layer filtering.

    - Combines adjusted p-value, log fold-change, directionality, and optional \
    absolute delta-beta criteria.
    - Ideal for generating final candidate lists for downstream validation or \
    pathway analysis.

    Parameters
    ----------
    res : pd.DataFrame
        Full differential methylation results table.
    lfc_col : str, default "logFC"
        Column containing log₂ fold change.
    pval_col : str, default "padj"
        Column containing adjusted p-values.
    lfc_thresh : float, default 0.0
        Minimum absolute |logFC| required (0 means no LFC filtering).
    pval_thresh : float, default 0.05
        Maximum adjusted p-value for significance.
    delta_beta_thresh : float or None, optional
        Minimum absolute difference in mean beta values between groups.
    direction : {"hyper", "hypo", None}, optional
        Restrict to hypermethylated (logFC > 0), hypomethylated (logFC < 0), or both.
    delta_beta_cols : list[str] or None, optional
        Exactly two column names containing group mean beta values (e.g. \
        ``["meanB_case", "meanB_control"]``).
        If ``None`` and delta-beta filtering is requested, automatically detects \
        columns starting with ``meanB_``.
    return_summary : bool, default False
        If True, return a detailed dictionary instead of just the CpG list.
    verbose : bool, default True
        Warn when no significant CpGs are found.

    Returns
    -------
    list[str] or dict
        List of significant CpG IDs (default)
        Or a summary dictionary containing counts, directional breakdown, \
        mean |logFC|, and the CpG list

    Raises
    ------
    KeyError
        If required columns (``lfc_col`` or ``pval_col``) are missing.
    ValueError
        If ``delta_beta_thresh`` is used but exactly two mean-beta columns \
        cannot be identified.
    """
    if res.empty:
        if verbose:
            logger.warning("Input DataFrame is empty; returning empty results")
        empty = (
            []
            if not return_summary
            else {
                "n_significant": 0,
                "n_hyper": 0,
                "n_hypo": 0,
                "mean_abs_lfc": 0.0,
                "cpgs": [],
            }
        )
        return empty

    # Column validation
    for col in (lfc_col, pval_col):
        if col not in res.columns:
            raise KeyError(f"Column '{col}' not in results DataFrame")

    sig = res[res[pval_col].fillna(1.0) < pval_thresh].copy()
    lfc = sig[lfc_col].fillna(0.0)

    # Apply direction + LFC threshold
    if direction == "hyper":
        sig = sig[lfc >= lfc_thresh]
    elif direction == "hypo":
        sig = sig[lfc <= -lfc_thresh]
    else:
        sig = sig[abs(lfc) >= lfc_thresh]

    # Delta-beta filtering
    if delta_beta_thresh is not None:
        if delta_beta_cols is None:
            delta_beta_cols = [c for c in sig.columns if c.startswith("meanB_")]
        if len(delta_beta_cols) != 2:
            raise ValueError(
                "Exactly 2 columns must be specified for delta-beta computation"
            )
        delta = abs(sig[delta_beta_cols[0]] - sig[delta_beta_cols[1]])
        sig = sig[delta >= delta_beta_thresh]

    cpgs = sig.index.tolist()

    if return_summary:
        return {
            "n_significant": len(cpgs),
            "n_hyper": (sig[lfc_col] > 0).sum(),
            "n_hypo": (sig[lfc_col] < 0).sum(),
            "mean_abs_lfc": sig[lfc_col].abs().mean() if not sig.empty else 0.0,
            "cpgs": cpgs,
        }
    return cpgs
