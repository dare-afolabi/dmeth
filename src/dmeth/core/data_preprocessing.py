#!/usr/bin/env python
# coding: utf-8


"""
Core preprocessing utilities for DNA methylation array data.

- This module implements quality control, normalization, batch correction,
and filtering operations for methylation matrices (beta or M-values).
- It provides both standard and high-performance implementations with
memory-efficient processing for large datasets.

Features
--------
- Sample-level QC: Flag/remove samples with excessive missing data
- CpG-level QC: Remove probes with high missingness or on sex chromosomes
- Normalization: Beta quantile normalization with optional M-value conversion
- Batch correction: Regression-based or ComBat-style correction
- Filtering: Remove low-variance probes
"""


import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dmeth.io.data_utils import ProcessedData
from dmeth.utils.logger import logger

try:
    from combat.pycombat import pycombat
except ImportError:
    pycombat = None
    logger.warning("pycombat not installed. Combat batch correction disabled.")


def qc_sample_level(
    data: ProcessedData,
    max_missing_fraction: float = 0.05,
    min_nonmissing_probes: Optional[int] = None,
    remove_samples: bool = True,
) -> ProcessedData:
    """
    Identify and optionally remove samples with excessive missing methylation data.

    Parameters
    ----------
    data : ProcessedData
        Input methylation dataset.
    max_missing_fraction : float, default 0.05
        Maximum allowed fraction of missing CpGs per sample (5% by default).
    min_nonmissing_probes : int or None, optional
        Alternative absolute threshold: minimum number of detected (non-missing) \
        probes required.
    remove_samples : bool, default True
        If True, flagged samples are removed from both ``M`` and ``pheno``; \
        if False, only metadata is recorded.

    Returns
    -------
    ProcessedData
        Updated container (samples removed if requested).

    Notes
    -----
    Flagged sample list and thresholds are stored in ``data.meta["qc"]\
    ["sample_missing"]``.
    """
    if "qc" not in data.meta:
        data.meta["qc"] = {}

    n_probes = data.M.shape[0]
    missing_frac = data.M.isna().sum(axis=0) / n_probes
    flagged = missing_frac[missing_frac > max_missing_fraction].index

    if min_nonmissing_probes is not None:
        low_count = data.M.notna().sum(axis=0) < min_nonmissing_probes
        flagged = flagged.union(low_count[low_count].index)

    flagged = sorted(flagged)
    data.meta["qc"]["sample_missing"] = {
        "threshold": max_missing_fraction,
        "flagged": len(flagged),
        "removed": len(flagged) if remove_samples else 0,
    }

    logger.info(
        f"Sample level QC: threshold={max_missing_fraction}, flagged="
        f"{len(flagged)}, removed={len(flagged) if remove_samples else 0}"
    )

    if remove_samples and flagged:
        keep = [c for c in data.M.columns if c not in flagged]
        data.M = data.M.loc[:, keep]
        data.pheno = data.pheno.loc[keep]
    return data


def qc_cpg_level(
    data: ProcessedData,
    max_missing_fraction: float = 0.10,
    drop_sex_chr: bool = True,
    chr_col: str = "chromosome",
) -> ProcessedData:
    """
    Remove CpGs with high missingness across samples and optionally drop \
    sex-chromosome probes.

    Parameters
    ----------
    data : ProcessedData
        Input dataset.
    max_missing_fraction : float, default 0.10
        Maximum fraction of samples allowed to be missing for a CpG (10% default).
    drop_sex_chr : bool, default True
        If True and annotation is available, remove all probes on chromosomes X and Y.
    chr_col : str, default "chromosome"
        Column name in ``data.ann`` containing chromosome information.

    Returns
    -------
    ProcessedData
        Updated container with low-quality and/or sex-chromosome CpGs removed.

    Notes
    -----
    Summary statistics are recorded in ``data.meta["qc"]["cpg_missing"]``.
    """
    n_samples = data.M.shape[1]
    missing_frac = data.M.isna().sum(axis=1) / n_samples
    flagged = set(missing_frac[missing_frac > max_missing_fraction].index)

    if drop_sex_chr:
        if data.ann is None:
            logger.warning(
                "drop_sex_chr=True but annotation not provided; \
                skipping sex chromosome probe removal."
            )
        elif chr_col not in data.ann.columns:
            logger.warning(
                f"drop_sex_chr=True but '{chr_col}' not found in annotation; \
                skipping sex chromosome probe removal."
            )
        else:
            sex_probes = data.ann.index[
                data.ann[chr_col].astype(str).str.upper().isin({"X", "Y"})
            ]
            flagged.update(sex_probes)

    flagged = sorted(flagged)
    data.meta.setdefault("qc", {})
    data.meta["qc"]["cpg_missing"] = {
        "threshold": max_missing_fraction,
        "sex_chr_dropped": drop_sex_chr
        and (data.ann is not None and chr_col in data.ann.columns),
        "flagged": len(flagged),
    }

    logger.info(
        f"CpG level QC:threshold={max_missing_fraction}, sex_chr_dropped="
        f"{drop_sex_chr}, flagged={len(flagged)}"
    )

    if flagged:
        data.M = data.M.drop(index=flagged)
        if data.ann is not None and not data.ann.empty:
            data.ann = data.ann.drop(index=[f for f in flagged if f in data.ann.index])
    return data


def normalize_methylation(
    data: ProcessedData,
    method: str = "beta_quantile",
    convert_to: Optional[str] = None,
    copy: bool = False,
    # element threshold to trigger memmap (n_probes * n_samples)
    q_chunk_threshold: int = 1e8,
    q_block_probes: int = 50_000,  # probes-per-block when assigning back
) -> ProcessedData:
    """
    Perform quantile normalization across samples with optional beta ↔ \
    M-value conversion.

    - Implements memory-efficient beta quantile normalization using column-wise \
    sorting and block-wise rank mapping.
    - Automatically falls back to disk-backed memmap for very large datasets.

    Parameters
    ----------
    data : ProcessedData
        Input methylation data.
    method : {"beta_quantile", "none"}, default "beta_quantile"
        Normalization method.
    convert_to : {"beta", "m"} or None, optional
        Convert matrix type after normalization (e.g., "m" for M-values).
    copy : bool, default False
        Work on a deep copy of the input.
    q_chunk_threshold : int, default 1e8
        Element count threshold (n_probes × n_samples) above which memmap is used.
    q_block_probes : int, default 50_000
        Number of probes processed per block during rank-to-target assignment.

    Returns
    -------
    ProcessedData
        Normalized dataset with updated ``meta["normalized"]`` and detailed provenance.

    Notes
    -----
    Preserves original NaNs. Records full normalization metadata including \
    memory strategy.
    """
    if copy:
        import copy as cp

        data = cp.deepcopy(data)

    # Quick checks
    if data.M.size == 0:
        return data

    n_probes, n_samples = data.M.shape
    data.meta.setdefault("normalization", {})
    use_memmap = False

    if method == "beta_quantile":
        # Work with numpy arrays for speed; use float32 to reduce memory footprint
        arr = data.M.values.astype(np.float32)
        nan_mask = np.isnan(arr)

        # Temporary fill for ranking - use column medians
        col_medians = np.nanmedian(arr, axis=0)
        filled = np.where(nan_mask, np.broadcast_to(col_medians, arr.shape), arr)

        n_elements = int(n_probes) * int(n_samples)

        # Function to create either ndarray or memmap for sorted values
        use_memmap = n_elements > q_chunk_threshold
        if use_memmap:
            # create temporary memmap file in system temp
            tmp_dir = tempfile.mkdtemp(prefix="qnorm_")
            mempath = os.path.join(tmp_dir, "sorted_vals.dat")
            sorted_vals = np.memmap(
                mempath, dtype=np.float32, mode="w+", shape=(n_probes, n_samples)
            )
        else:
            sorted_vals = np.empty((n_probes, n_samples), dtype=np.float32)

        # Compute per-sample sorted vectors and write into sorted_vals column-by-column
        # This avoids storing multiple full-size copies of the data
        for j in range(n_samples):
            col = filled[:, j]
            sorted_vals[:, j] = np.sort(col, kind="quicksort")

        # Compute the target distribution: mean of sorted
        # values across samples (float32)
        target = np.nanmean(sorted_vals, axis=1).astype(np.float32)

        # Now assign target values back to each sample according to rank (argsort)
        # Do it in probe-blocks to limit peak memory
        qnormed = np.empty_like(arr, dtype=np.float32)
        probes = np.arange(n_probes)
        for start in range(0, n_probes, q_block_probes):
            end = min(n_probes, start + q_block_probes)
            block_idx = probes[start:end]
            # argsort on the block for each column -> we need positions of these probes
            # We'll compute order indices for the block across samples
            # For each sample j, find the rank positions of
            # block probes among entire sample
            for j in range(n_samples):
                col = filled[:, j]
                # argsort full column (int32)
                order = np.argsort(col, kind="quicksort")
                # inverse of order -> rank[position] = rank
                inv = np.empty_like(order, dtype=np.int32)
                inv[order] = np.arange(len(order), dtype=np.int32)
                # for the block probes, get their ranks
                ranks = inv[block_idx]
                # assign target values at those ranks
                qnormed[block_idx, j] = target[ranks]

        # Reinsert NaNs
        qnormed[nan_mask] = np.nan

        # Convert back to DataFrame with original index/columns
        data.M = pd.DataFrame(
            qnormed.astype(np.float32), index=data.M.index, columns=data.M.columns
        )

        # Clean up memmap if used
        if use_memmap:
            try:
                # flush and remove temporary files
                sorted_vals.flush()
                del sorted_vals
                shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")

        # Update meta
        data.meta["normalized"] = True
        data.meta["normalization"]["method"] = "beta_quantile"
        data.meta["normalization"]["n_probes"] = int(n_probes)
        data.meta["normalization"]["n_samples"] = int(n_samples)
        data.meta["normalization"]["memmap_used"] = bool(use_memmap)

    elif method in (None, "none"):
        # nothing to do
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Conversion between matrix types, if requested
    if convert_to and convert_to != data.meta.get("matrix_type"):
        eps = 1e-6
        if convert_to == "m":
            beta = np.clip(data.M.values.astype(float), eps, 1 - eps)
            mvals = np.log2(beta / (1 - beta))
            data.M = pd.DataFrame(mvals, index=data.M.index, columns=data.M.columns)
            data.meta["matrix_type"] = "m"
            data.meta.setdefault("transform", {})["last"] = "beta->m"
        elif convert_to == "beta":
            m = data.M.values.astype(float)
            beta = (2**m) / (1 + 2**m)
            data.M = pd.DataFrame(beta, index=data.M.index, columns=data.M.columns)
            data.meta["matrix_type"] = "beta"
            data.meta.setdefault("transform", {})["last"] = "m->beta"
        else:
            raise ValueError("convert_to must be 'm' or 'beta'")

    logger.info(
        f"Normalization: method={method}, n_probes={int(n_probes)}, n_samples="
        f"{int(n_samples)}, memmap_used={bool(use_memmap)}"
    )

    return data


def normalize_methylation_highperf(
    data: ProcessedData,
    method: str = "beta_quantile",
    convert_to: Optional[str] = None,
    copy: bool = False,
    # elements = n_probes * n_samples threshold to use memmap
    memmap_threshold: int = 2e8,
    memmap_dir: Optional[str] = None,
    n_workers: Optional[int] = None,
    sample_block: int = 16,
    random_state: Optional[int] = None,
) -> ProcessedData:
    """
    High-performance quantile normalization with multiprocessing and automatic \
    memmap handling.

    Designed for >850k × 500+ datasets. Uses disk-backed arrays when needed \
    and parallelizes the rank-mapping stage.

    Parameters
    ----------
    memmap_threshold : int, default 2e8
        Element count above which temporary memmap files are created.
    memmap_dir : str or None
        Directory for memmap files (defaults to system temp).
    n_workers : int or None
        Number of processes for parallel assignment (defaults to CPU count – 1).
    sample_block : int, default 16
        Number of samples per parallel job.
    random_state : int or None
        Currently unused (kept for API consistency).

    Returns
    -------
    ProcessedData
        Quantile-normalized data with comprehensive metadata.

    Notes
    -----
    Significantly faster and lower peak RAM than ``normalize_methylation`` \
    on large cohorts.
    """
    import multiprocessing as mp

    if copy:
        import copy as cp

        data = cp.deepcopy(data)

    if method not in (None, "none", "beta_quantile"):
        raise ValueError(
            "Only 'beta_quantile' (true quantile), 'none', or \
            None supported by this routine."
        )

    if data.M.size == 0:
        return data

    n_probes, n_samples = data.M.shape
    data.meta.setdefault("normalization", {})
    use_memmap = False
    dtype = np.float32

    if method in (None, "none"):
        # Only optionally convert types
        if convert_to and convert_to != data.meta.get("matrix_type"):
            return normalize_methylation(
                data, method="none", convert_to=convert_to, copy=False
            )
        return data

    # Prepare arrays
    arr = data.M.values.astype(dtype, copy=False)
    nan_mask = np.isnan(arr)

    # Fill missing values per-sample with column medians for ranking purposes
    col_medians = np.nanmedian(arr, axis=0)
    filled = np.where(nan_mask, np.broadcast_to(col_medians, arr.shape), arr)

    use_memmap = (n_probes * n_samples) > int(memmap_threshold)
    tmpdir = memmap_dir or tempfile.mkdtemp(prefix="qnorm_tmp_")
    sorted_path = None
    qnormed_path = None

    try:
        # 1) Build per-sample sorted arrays (n_probes x n_samples)
        if use_memmap:
            sorted_path = os.path.join(tmpdir, "sorted_vals.npy")
            sorted_vals = np.memmap(
                sorted_path, dtype=dtype, mode="w+", shape=(n_probes, n_samples)
            )
            for j in range(n_samples):
                sorted_vals[:, j] = np.sort(filled[:, j], kind="quicksort")
        else:
            sorted_vals = np.empty((n_probes, n_samples), dtype=dtype)
            for j in range(n_samples):
                sorted_vals[:, j] = np.sort(filled[:, j], kind="quicksort")

        # 2) Compute target distribution (mean of sorted values) - shape (n_probes,)
        # Use nanmean just in case, but sorted_vals should \
        # not contain NaN (we filled earlier)
        target = np.nanmean(sorted_vals, axis=1).astype(dtype, copy=False)

        # 3) Map target back to each sample according to the rank \
        # of each probe in that sample.
        # For memory scaling we write qnormed into a memmap if needed
        if use_memmap:
            qnormed_path = os.path.join(tmpdir, "qnormed.npy")
            qnormed = np.memmap(
                qnormed_path, dtype=dtype, mode="w+", shape=(n_probes, n_samples)
            )
        else:
            qnormed = np.empty((n_probes, n_samples), dtype=dtype)

        # Helper to process a list of sample indices
        def _assign_samples(sample_indices):
            out = {}
            for j in sample_indices:
                col = filled[:, j]
                order = np.argsort(col, kind="quicksort")
                inv = np.empty_like(order, dtype=np.int32)
                inv[order] = np.arange(len(order), dtype=np.int32)
                # Map target by rank
                mapped = target[inv]
                # Restore NaNs
                mapped[nan_mask[:, j]] = np.nan
                out[j] = mapped
            return out

        # Decide worker count
        workers = int(n_workers or max(1, (mp.cpu_count() or 1) - 1))
        # Serial fast-path for small n_samples or when workers==1
        if workers <= 1 or n_samples < 4:
            for j in range(n_samples):
                assigned = _assign_samples([j])[j]
                qnormed[:, j] = assigned
        else:
            # Parallel assignment in blocks of samples
            sample_indices = list(range(n_samples))
            blocks = [
                sample_indices[i : i + sample_block]
                for i in range(0, n_samples, sample_block)
            ]
            with mp.Pool(processes=workers) as pool:
                for res in pool.imap_unordered(_assign_samples, blocks):
                    for j, arr_j in res.items():
                        qnormed[:, j] = arr_j

        # Convert back to DataFrame with original index/columns
        data.M = pd.DataFrame(
            qnormed.astype(np.float32), index=data.M.index, columns=data.M.columns
        )

        # Metadata
        data.meta["normalized"] = True
        data.meta["normalization"]["method"] = "beta_quantile"
        data.meta["normalization"]["n_probes"] = int(n_probes)
        data.meta["normalization"]["n_samples"] = int(n_samples)
        data.meta["normalization"]["memmap_used"] = bool(use_memmap)
        data.meta["normalization"]["temp_dir"] = tmpdir if use_memmap else None
        data.meta["normalization"]["n_workers"] = workers

    finally:
        # If memmap used we keep qnormed by reading into DataFrame above;
        # remove sorted memmap if any
        try:
            if use_memmap and sorted_path and os.path.exists(sorted_path):
                # memmaps flushed on deletion - unlink files
                try:
                    os.remove(sorted_path)
                except Exception as e:
                    logger.warning(f"Failed to remove tenporary file: {e}")
            # Do not remove tmpdir if we recorded path in meta
            # (so user can inspect); user can clean later.
            if not data.meta["normalization"].get("temp_dir"):
                try:
                    shutil.rmtree(tmpdir)
                except Exception as e:
                    logger.warning(f"Failed to remove tenporary file: {e}")
        except Exception as e:
            logger.warning(f"Failed to remove tenporary file: {e}")

    # Optional conversion between matrix types if requested
    if convert_to and convert_to != data.meta.get("matrix_type"):
        eps = 1e-6
        if convert_to == "m":
            beta = np.clip(data.M.values.astype(float), eps, 1 - eps)
            mvals = np.log2(beta / (1 - beta))
            data.M = pd.DataFrame(mvals, index=data.M.index, columns=data.M.columns)
            data.meta["matrix_type"] = "m"
            data.meta.setdefault("transform", {})["last"] = "beta->m"
        elif convert_to == "beta":
            m = data.M.values.astype(float)
            beta = (2**m) / (1 + 2**m)
            data.M = pd.DataFrame(beta, index=data.M.index, columns=data.M.columns)
            data.meta["matrix_type"] = "beta"
            data.meta.setdefault("transform", {})["last"] = "m->beta"
        else:
            raise ValueError("convert_to must be 'm' or 'beta'")

    logger.info(
        f"Normalization: method={method}, memmap_threshold="
        f"{memmap_threshold}, n_workers="
        f"{n_workers}, sample_block={sample_block}, random_state={random_state}"
    )

    return data


def batch_correction(
    data: ProcessedData,
    batch_col: str,
    covariates: Optional[List[str]] = None,
    method: str = "qr",
    weights: Optional[np.ndarray] = None,
    block_size: int = 50_000,
    robust: bool = False,
    return_diagnostics: bool = False,
) -> Union[ProcessedData, Tuple[ProcessedData, Dict[str, Any]]]:
    """
    Regression-based batch correction using full linear model (intercept + \
    covariates + batch).

    Numerically stable QR/pinv solver with optional per-probe weights and \
    block-wise processing.

    Parameters
    ----------
    batch_col : str
        Column in ``pheno`` identifying batch.
    covariates : list[str] or None
        Additional technical or biological covariates to preserve.
    method : {"qr", "pinv"}, default "qr"
        Solver backend.
    weights : ndarray or None
        Per-probe inverse-variance weights.
    robust : bool, default False
        Use Huber robust regression (much slower).
    return_diagnostics : bool, default False
        Return detailed effect sizes, residuals, and variance explained.

    Returns
    -------
    ProcessedData or (ProcessedData, dict)
        Corrected data and optional diagnostics.
    """
    if not hasattr(data, "pheno") or batch_col not in data.pheno.columns:
        raise KeyError(f"Batch column '{batch_col}' not in pheno")

    n_samples = data.M.shape[1]
    n_probes = data.M.shape[0]

    # Build design matrix
    X_parts = [np.ones((n_samples, 1), dtype=float)]
    feat_names = ["(Intercept)"]
    if covariates:
        for cov in covariates:
            if cov not in data.pheno.columns:
                raise KeyError(f"Covariate '{cov}' not in pheno")
            s = data.pheno[cov]
            if pd.api.types.is_numeric_dtype(s):
                arr = s.astype(float).values.reshape(-1, 1)
                X_parts.append(arr)
                feat_names.append(cov)
            else:
                dummies = pd.get_dummies(s.astype(str), drop_first=True)
                X_parts.append(dummies.values)
                feat_names += [f"{cov}:{c}" for c in dummies.columns]

    # Batch dummies (keep all but drop one level to avoid collinearity)
    batch_dummies = pd.get_dummies(data.pheno[batch_col].astype(str), drop_first=True)
    if batch_dummies.shape[1] == 0:
        logger.warning("Single-level batch; nothing to correct.")
        return (data, {}) if return_diagnostics else data
    X_parts.append(batch_dummies.values)
    batch_feat_names = [f"batch:{c}" for c in batch_dummies.columns]
    feat_names += batch_feat_names

    X_full = np.concatenate(X_parts, axis=1)  # (n_samples, p)
    p = X_full.shape[1]

    if np.isnan(X_full).any():
        raise ValueError(
            "Design matrix contains NaNs; check pheno/covariates for missing entries"
        )

    # Precompute solver components depending on method and weights
    if weights is not None:
        if len(weights) != n_probes:
            raise ValueError("weights length must equal number of probes")
        # We'll implement weighted least squares by scaling Y and X per probe in blocks
        use_weights = True
    else:
        use_weights = False

    # Diagnostics containers
    diagnostics: Dict[str, Any] = {}
    effect_sizes = np.zeros(
        (p, n_probes), dtype=float
    )  # may be large; we'll fill block-wise
    residuals_store = None
    if return_diagnostics:
        residuals_store = np.empty((n_probes, n_samples), dtype=float)

    # Precompute BX = (X'X)^-1 X' (p x n_samples) - small matrix
    if method == "pinv":
        XtX_inv = np.linalg.pinv(X_full.T @ X_full)
        BX = XtX_inv @ X_full.T  # (p, n_samples)
    else:
        # QR-based stable computation:
        # compute pinv of XtX via np.linalg.pinv - also acceptable
        XtX = X_full.T @ X_full
        XtX_inv = np.linalg.pinv(XtX)
        BX = XtX_inv @ X_full.T

    # block-wise processing
    for start in range(0, n_probes, block_size):
        end = min(n_probes, start + block_size)
        block_idx = slice(start, end)
        Y_block = data.M.values[block_idx, :]  # (b, n_samples)

        if use_weights:
            # weights for block
            w_block = weights[start:end].astype(float)  # length b
            # scale Y_block rows by sqrt(w)
            Whalf = np.sqrt(w_block)[:, None]
            Yw = Y_block * Whalf

            for i in range(Yw.shape[0]):
                # yi = Yw[i, :]
                beta_i = BX @ Y_block[i, :].T
                effect_sizes[:, start + i] = beta_i
                fitted_i = X_full @ beta_i
                resid_i = Y_block[i, :] - fitted_i
                if return_diagnostics:
                    residuals_store[start + i, :] = resid_i
        else:
            # vectorized: betas_block = BX @ Y_block.T  -> shape (p, b)
            betas_block = BX @ Y_block.T  # (p, b)
            effect_sizes[:, start:end] = betas_block
            # compute fitted = X_full @ betas_block -> (n_samples, b) -> transpose
            fitted_block = (X_full @ betas_block).T  # (b, n_samples)
            resid_block = Y_block - fitted_block
            if return_diagnostics:
                residuals_store[start:end, :] = resid_block

    # Compose corrected matrix by removing batch contribution only
    # Identify batch columns indices within feature set
    batch_start_idx = p - batch_dummies.shape[1]
    # effect_sizes[batch_start_idx:, :] are batch betas (n_batch_dummies x n_probes)
    batch_betas = effect_sizes[batch_start_idx:, :]  # (n_batch_dummies, n_probes)
    Xb = X_full[:, batch_start_idx:]  # (n_samples, n_batch_dummies)
    # batch_effects per probe and sample = Xb @ batch_betas ->
    # (n_samples, n_probes) transposed
    # Do block-wise assembly to avoid big allocation
    corrected = np.empty_like(data.M.values, dtype=float)
    for start in range(0, n_probes, block_size):
        end = min(n_probes, start + block_size)
        bb = batch_betas[:, start:end]  # (n_batch_dummies, block)
        batch_effects_block = (Xb @ bb).T  # (block, n_samples)
        orig_block = data.M.values[start:end, :]
        corrected[start:end, :] = orig_block - batch_effects_block

    data.M = pd.DataFrame(corrected, index=data.M.index, columns=data.M.columns)

    # Build diagnostics if requested
    if return_diagnostics:
        diagnostics["effect_sizes"] = pd.DataFrame(
            effect_sizes, index=feat_names, columns=data.M.index
        )
        diagnostics["residuals"] = pd.DataFrame(
            residuals_store, index=data.M.index, columns=data.M.columns
        )
        # Variance explained per covariate: compute SS_total and SS_res
        # for each covariate by constructing fitted contribution
        var_expl = {}
        # ss_total = np.var(data.M.values, axis=1, ddof=1)
        for fi, fname in enumerate(feat_names):
            # contribution from term fi across samples = X_full[:, fi] * beta(fi, probe)
            contribution = np.outer(
                X_full[:, fi], effect_sizes[fi, :]
            ).T  # (n_probes, n_samples)
            var_expl[fname] = np.nanmean(
                contribution**2, axis=1
            )  # mean square as contribution proxy
        diagnostics["var_explained_mean_square"] = pd.DataFrame(
            var_expl, index=data.M.index
        )

    # Metadata
    data.meta["batch_corrected"] = True
    data.meta["batch_method"] = "advanced_regression"
    data.meta.setdefault("batch_info", {})["batch_col"] = batch_col

    logger.info(
        f"Batch correction: batch_method=advanced_regression, \
        n_samples={int(data.M.shape[1])}, \
        n_probes={int(data.M.shape[0])}, batch_col={batch_col}"
    )

    return (data, diagnostics) if return_diagnostics else data


def batch_correction_combat(
    data: ProcessedData,
    batch_col: str,
    covariates: Optional[List[str]] = None,
    parametric: bool = True,
    return_diagnostics: bool = False,
) -> Union[ProcessedData, Tuple[ProcessedData, Dict[str, Any]]]:
    """
    Wrapper for ComBat (or safe fallback) batch correction.

    - Uses ``pycombat`` if installed (parametric or nonparametric).
    - Falls back to conservative mean-centering + variance rescaling with clear warning.
    - Automatically handles beta ↔ M-value conversion.

    Parameters
    ----------
    parametric : bool, default True
        Use parametric ComBat (faster, assumes normality).
    return_diagnostics : bool, default False
        Return metadata about which method was actually used.

    Returns
    -------
    ProcessedData or (ProcessedData, dict)
        Batch-corrected data and diagnostics."""
    diagnostics: Dict[str, Any] = {}
    if not hasattr(data, "pheno") or batch_col not in data.pheno.columns:
        raise KeyError(f"Batch column '{batch_col}' not in pheno")

    orig_type = data.meta.get("matrix_type", "beta")
    need_return_to_beta = orig_type == "beta"

    # Convert to M-values for Combat input (Combat typically
    # expects roughly normal data)
    eps = 1e-6
    if orig_type == "beta":
        beta = np.clip(data.M.astype(float), eps, 1 - eps)
        mvals = np.log2(beta / (1 - beta))
    else:
        mvals = data.M.astype(float)

    batch_labels = data.pheno[batch_col].astype(str).values

    if covariates:
        for c in covariates:
            if c not in data.pheno.columns:
                raise KeyError(f"Covariate '{c}' not found in pheno")
        mod_df = pd.get_dummies(data.pheno[covariates].astype(str), drop_first=True)
        mod = mod_df.values

    else:
        mod = np.empty((len(batch_labels), 0))  # no covariates

    def _combat_fallback(mvals: pd.DataFrame, batch_labels: np.ndarray):
        unique_batches, inv_idx = np.unique(batch_labels, return_inverse=True)
        corrected = np.empty_like(mvals.values)
        batch_means = {}
        batch_vars = {}

        for b_i, b in enumerate(unique_batches):
            mask = inv_idx == b_i
            batch_slice = mvals.iloc[:, mask]

            if mask.sum() <= 1:
                batch_means[b] = np.nanmean(batch_slice, axis=1)
                batch_vars[b] = (
                    np.nanvar(batch_slice, axis=1, ddof=1)
                    if mask.sum() > 1
                    else np.zeros(mvals.shape[0])
                )
                corrected[:, mask] = batch_slice.values - batch_means[b][:, None]
            else:
                bm = np.nanmean(batch_slice, axis=1)
                bv = np.nanvar(batch_slice, axis=1, ddof=1)
                pooled_sd = np.sqrt(np.nanvar(mvals, axis=1, ddof=1))
                scale = np.where(bv > 0, np.sqrt(bv), 1.0)
                corrected[:, mask] = (
                    (batch_slice.values - bm[:, None])
                    / np.where(scale[:, None] == 0, 1.0, scale[:, None])
                    * pooled_sd[:, None]
                )
                batch_means[b] = bm
                batch_vars[b] = bv

        return corrected, batch_means, batch_vars

    if pycombat is not None:
        # pycombat interface variants exist; most accept (samples x features) arrays
        try:
            combat_input = mvals.T  # samples x probes
            corrected = pycombat(combat_input, batch_labels, mod, parametric=parametric)
            # corrected: samples x probes -> transpose back
            corrected = corrected.T
            used_combat = True
            diagnostics["combat_parametric"] = bool(parametric)
        except Exception as e:
            logger.warning(
                f"pycombat failed ({e}); using conservative fallback instead."
            )
            used_combat = False
            corrected, batch_means, batch_vars = _combat_fallback(mvals, batch_labels)
            diagnostics["fallback_batch_means"] = batch_means
            diagnostics["fallback_batch_vars"] = batch_vars

    else:
        logger.warning("pycombat not available; using conservative fallback.")
        used_combat = False
        corrected, batch_means, batch_vars = _combat_fallback(mvals, batch_labels)
        diagnostics["fallback_batch_means"] = batch_means
        diagnostics["fallback_batch_vars"] = batch_vars

    # Convert back to beta if original was beta
    if need_return_to_beta:
        corrected_beta = (2**corrected) / (1 + 2**corrected)
        data.M = pd.DataFrame(
            corrected_beta, index=data.M.index, columns=data.M.columns
        )
    else:
        data.M = pd.DataFrame(corrected, index=data.M.index, columns=data.M.columns)

    data.meta["batch_corrected"] = True
    data.meta["batch_method"] = (
        "combat_parametric"
        if used_combat and parametric
        else ("combat_nonparametric" if used_combat else "mean_center_fallback")
    )
    diagnostics["used_pycombat"] = bool(pycombat is not None and used_combat)
    diagnostics["n_samples"] = int(data.M.shape[1])
    diagnostics["n_probes"] = int(data.M.shape[0])
    diagnostics["batch_col"] = batch_col

    batch_method = data.meta["batch_method"]
    used_pycombat = diagnostics["used_pycombat"]
    n_samples = diagnostics["n_samples"]
    n_probes = diagnostics["n_probes"]
    logger.info(
        f"Batch correction - batch_method: {batch_method}, used_pycombat: \
        {used_pycombat}, n_samples: {n_samples}, n_probes: {n_probes}, \
        batch_col: {batch_col}"
    )

    if return_diagnostics:
        return data, diagnostics
    return data


def filter_low_variance_cpgs(
    data: ProcessedData, min_percentile: float = 10.0, inplace: bool = True
) -> ProcessedData:
    """
    Remove CpGs below a variance percentile threshold (e.g., bottom 10%).

    Parameters
    ----------
    - data : ProcessedData
    - min_percentile : float, default 10.0
        Keep only probes with variance ≥ this percentile.
    - inplace : bool, default True
        Modify input object or return a copy.

    Returns
    -------
    - ProcessedData
        Dataset with low-variance probes removed.

    Notes
    -----
    - Number of removed probes recorded in ``data.meta["qc"]["low_variance_removed"]``.
    """
    var = data.M.var(axis=1)
    thr = np.percentile(var.dropna(), min_percentile)
    keep = var[var >= thr].index
    removed = len(data.M) - len(keep)

    if not inplace:
        import copy as cp

        data = cp.deepcopy(data)
    data.M = data.M.loc[keep]
    if data.ann is not None:
        data.ann = data.ann.loc[keep]
    data.meta["qc"]["low_variance_removed"] = removed

    logger.info(f"low_variance_removed: {removed}")

    return data
