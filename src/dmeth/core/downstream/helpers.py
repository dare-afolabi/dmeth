#!/usr/bin/env python
# coding: utf-8


"""
General-purpose utilities for DNA methylation analysis workflows.

This lightweight module contains essential helper functions used across the DMeth \
package for consistent data handling and summarization. It provides robust \
chromosome string normalization and flexible group-wise aggregation of beta-value \
matrices, ensuring compatibility and reproducibility in downstream \
statistical and annotation steps.

Features
--------
- Robust chromosome identifier normalization with automatic 'chr' prefix \
addition and preservation of X/Y conventions
- Safe handling of mixed chromosome formats (e.g., '1', 'chr1', 'CHR1' → 'chr1', 'chrX')
- Efficient group-wise summarization of per-sample beta values with support \
for arbitrary aggregation functions
- Automatic computation of both mean and variance per group for \
interpretability and modeling
- Graceful handling of missing groups, empty inputs, and non-overlapping sample sets
- Index-preserving operations fully compatible with CpG × sample matrix conventions
"""


from typing import Callable

import numpy as np
import pandas as pd


def _clean_chr(ch: str) -> str:
    """
    Standardize chromosome identifiers to the canonical UCSC format (e.g., \
    ``"chr1"``, ``"chrX"``, ``"chrY"``).

    Robustly handles a wide variety of input formats including:

    - ``"1"``, ``"chr1"``, ``"CHR1"`` → ``"chr1"``
    - ``"X"``, ``"chrx"``, ``"ChrX"`` → ``"chrX"``
    - ``"Y"`` → ``"chrY"``

    Parameters
    ----------
    ch : str or None
        Input chromosome identifier.

    Returns
    -------
    str or None
        Normalized chromosome string with lowercase ``"chr"`` prefix and \
        preserved uppercase ``X``/``Y``.
        Returns ``None`` unchanged if input is ``None``.
    """
    if ch is None:
        return ch
    ch = str(ch).strip()
    if not ch.lower().startswith("chr"):
        ch = "chr" + ch

    # Preserve X/Y uppercase
    if ch.lower() in {"chrx", "chry"}:
        return ch[:3] + ch[3:].upper()
    return ch.lower()


def summarize_groups(
    beta: pd.DataFrame, groups: pd.Series, summary_func: Callable = np.mean
) -> pd.DataFrame:
    """
    Compute group-wise summary statistics (mean and variance) across samples \
    for a beta-value matrix.

    Designed for rapid interpretation of methylation levels across \
    experimental conditions or cell types.

    Parameters
    ----------
    beta : pd.DataFrame
        Beta-value matrix with CpGs as rows (index) and samples as columns.
    groups : pd.Series
        Sample-to-group mapping. Index must align with ``beta.columns``.
    summary_func : callable, default ``np.mean``
        Aggregation function applied per group (e.g., ``np.mean``, ``np.median``).

    Returns
    -------
    pd.DataFrame
        New DataFrame with columns:
        ``mean_{group}`` – group-wise summary using ``summary_func``
        ``var_{group}``  – sample variance within each group (ddof=1)

        Missing groups or empty intersections yield ``NaN`` columns.

    Notes
    -----
    - Fully preserves CpG index.
    - Handles partial or missing group overlap gracefully.
    - Ideal for generating input tables for delta-beta calculation, \
    visualization, or reporting.
    """
    if beta is None or beta.empty:
        return pd.DataFrame()

    groups = groups.astype(str)
    result = {}
    for g in groups.unique():
        cols = groups[groups == g].index.intersection(beta.columns)
        if len(cols) == 0:
            result[f"mean_{g}"] = pd.Series(np.nan, index=beta.index)
            result[f"var_{g}"] = pd.Series(np.nan, index=beta.index)
            continue
        result[f"mean_{g}"] = beta[cols].apply(summary_func, axis=1)
        result[f"var_{g}"] = beta[cols].var(axis=1, ddof=1)
    return pd.DataFrame(result)
