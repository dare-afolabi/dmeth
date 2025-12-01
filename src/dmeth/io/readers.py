#!/usr/bin/env python
# coding: utf-8


"""
Input utilities for loading methylation datasets in the DMeth pipeline.

This module provides a **Python API** for loading
preprocessed or raw methylation data into the standard :class:`ProcessedData` \
container used throughout the package.

Features
--------
- Robust multi-format reading (CSV, TSV, Excel, pickle, HDF5, Parquet, Feather)
- Automatic coercion of methylation matrix to numeric values
- Smart HDF5 key selection (prefers `/M` or `M`)
- Safe sample alignment between methylation matrix and phenotype data
- Two high-level entry points:
    - ``load_processed_data()`` – load a previously saved ``ProcessedData`` object
    - ``load_methylation_data()`` – construct a new ``ProcessedData`` from raw files
"""


from pathlib import Path
from typing import Optional, Union

import pandas as pd

from dmeth.io.data_utils import ProcessedData
from dmeth.utils.logger import logger


def load_processed_data(path: Union[str, Path], trusted: bool = False) -> ProcessedData:
    """
    Load a previously saved ProcessedData object from disk.

    Supports the two canonical DMeth output formats:
    - ``.pkl`` / ``.pickle`` → direct pandas pickle of the ProcessedData instance
    - ``.h5`` / ``.hdf5`` → HDF5 store containing the keys ``M``, \
    ``pheno``, and optionally ``ann``

    Parameters
    ----------
    path : str or Path
        Path to the saved ProcessedData file.
    trusted : bool, default False
        Only allow ``.pkl`` / ``.pickle`` files from trusted sources

    Returns
    -------
    ProcessedData
        Fully reconstructed and validated ProcessedData container.

    Raises
    ------
    ValueError
        If pickle file is from an untrusted source
    ValueError
        If the file extension is not one of the supported formats.
    """
    path = Path(path)
    if path.suffix.lower() in (".pkl", ".pickle"):
        if trusted:
            return pd.read_pickle(path)  # nosec B301
            logger.info(f"ProcessedData object loaded from {path}")
        else:
            ValueError("Pickle files are not supported for untrusted input")
    elif path.suffix.lower() in (".h5", ".hdf5"):
        with pd.HDFStore(path, "r") as store:
            M = store["M"]
            pheno = store["pheno"]
            ann = store.get("ann", pd.DataFrame(index=M.index))
        return ProcessedData(M=M, pheno=pheno, ann=ann)
        logger.info(f"ProcessedData object loaded from {path}")
    else:
        raise ValueError("Unsupported load format; must be .pkl or .h5")


def _read(
    path: Union[str, Path], index_col: Optional[int], trusted: bool = False
) -> pd.DataFrame:
    """
    Internal robust reader for many common tabular formats.

    Supported formats
    -----------------
    .csv, .tsv/.txt, .xlsx/.xls, .pkl/.pickle, .parquet, .feather, .h5/.hdf5

    For HDF5 files:

        Automatically selects the dataset ``/M`` or ``M`` if present.
        Falls back to the only dataset if the file contains exactly one.
        Raises a clear error listing available keys otherwise.

    Parameters
    ----------
    path : str or Path
        File to read.
    index_col : int or None, optional
        Column to use as the row index (passed to pandas readers).
    trusted : bool, default False
        Only allow ``.pkl`` / ``.pickle`` files from trusted sources

    Returns
    -------
    pd.DataFrame
        Loaded table with the requested index set.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If pickle file is from an untrusted source
    ValueError
        If the format is unsupported or (for HDF5) multiple ambiguous keys exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suf = path.suffix.lower()

    if suf == ".csv":
        return pd.read_csv(path, index_col=index_col)
    elif suf in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", index_col=index_col)
    elif suf in (".xlsx", ".xls"):
        return pd.read_excel(path, index_col=index_col)
    elif suf in (".pkl", ".pickle"):
        if trusted:
            return pd.read_pickle(path)  # nosec B301
        else:
            raise ValueError("Pickle files are not supported for untrusted input")
    elif suf in (".parquet",):
        return pd.read_parquet(path)
    elif suf in (".feather",):
        df = pd.read_feather(path)
        if index_col is not None:
            # if index_col is integer select that column as index name
            if isinstance(index_col, int):
                if index_col < 0 or index_col >= len(df.columns):
                    raise IndexError("index_col out of bounds for feather file")
                idx_name = df.columns[index_col]
            else:
                idx_name = index_col
            df = df.set_index(idx_name)
        return df
    elif suf in (".h5", ".hdf5"):
        # safer HDF access: prefer '/M' or 'M', otherwise require explicit choice
        with pd.HDFStore(path, "r") as store:
            keys = [k for k in store.keys()]
            if len(keys) > 1:
                raise ValueError("multiple keys found in HDF5 file")
            # normalize keys (strip leading '/')
            norm_keys = [k.lstrip("/") for k in keys]
            if "/M" in keys or "M" in norm_keys:
                # prefer exact '/M' if present, else first matching 'M'
                sel = "/M" if "/M" in keys else keys[norm_keys.index("M")]
                return store[sel]
            if len(keys) == 1:
                return store[keys[0]]
            # multiple keys present -> explicit required
            raise ValueError(
                f"HDF5 file contains multiple keys: {keys}. Please provide a \
                file with a single dataset or supply a path to the specific key."
            )
    else:
        raise ValueError(f"Unsupported format: {suf}")


def load_methylation_data(
    methylation_input: Union[str, Path, pd.DataFrame],
    pheno_input: Optional[Union[str, Path, pd.DataFrame]] = None,
    ann_input: Optional[Union[str, Path, pd.DataFrame]] = None,
    index_col_probe: Optional[int] = 0,
    index_col_sample: Optional[int] = 0,
) -> ProcessedData:
    """
    High-level constructor that builds a ProcessedData object from \
    raw matrix/metadata files.

    - Accepts file paths, already-loaded DataFrames, or a mix of both.
    - Performs automatic type coercion, sample alignment, and validation.

    Parameters
    ----------
    methylation_input : str | Path | pd.DataFrame
        CpG × sample methylation matrix (beta or M-values).
    pheno_input : str | Path | pd.DataFrame, optional
        Sample phenotype/metadata table. If ``None``, an empty placeholder indexed by
        the matrix columns is created.
    ann_input : str | Path | pd.DataFrame, optional
        CpG probe annotation (e.g., Illumina manifest). If ``None``, \
        annotation is omitted.
    index_col_probe : int, default 0
        Column containing CpG identifiers when reading the methylation matrix.
    index_col_sample : int, default 0
        Column containing sample identifiers when reading the phenotype table.

    Returns
    -------
    ProcessedData
        Fully validated container ready for downstream analysis.

    Notes
    -----
    - Non-numeric entries in the methylation matrix are forcibly coerced to NaN
      (with a warning indicating how many values were affected).
    - The phenotype table is reindexed to exactly match the sample order in ``M``.
    - All indices are converted to strings and alignment is checked by \
    ``ProcessedData.__post_init__``.
    """
    # Read M
    if isinstance(methylation_input, (str, Path)):
        M = _read(Path(methylation_input), index_col=index_col_probe)
    else:
        M = methylation_input.copy()

    # Coerce numeric values in M only (report coerced count)
    orig_total_na = M.isna().sum().sum()
    M = M.apply(pd.to_numeric, errors="coerce")
    new_total_na = M.isna().sum().sum()
    coerced = new_total_na - orig_total_na
    if coerced > 0:
        logger.warning(
            f"Coerced {int(coerced)} values to NaN while parsing methylation matrix."
        )

    # Read pheno
    if pheno_input is None:
        pheno = pd.DataFrame(index=M.columns)
    elif isinstance(pheno_input, (str, Path)):
        pheno = _read(Path(pheno_input), index_col=index_col_sample)
    else:
        pheno = pheno_input.copy()

    # Ensure pheno index matches samples exactly
    if not set(M.columns).issubset(set(pheno.index)):
        missing = set(M.columns) - set(pheno.index)
        raise KeyError(
            f"Pheno missing sample metadata for {len(missing)} sample(s): \
            {sorted(list(missing))[:10]}..."
        )
    # Reindex pheno to sample order
    pheno = pheno.reindex(M.columns)

    # Read annotation (if provided).
    # If not provided, keep ann = None so alignment check skips.
    if ann_input is None:
        # normalize to empty DataFrame with probe index
        ann = pd.DataFrame(index=M.index)
    elif isinstance(ann_input, (str, Path)):
        ann = pd.read_csv(ann_input, index_col=index_col_probe)
    elif isinstance(ann_input, pd.DataFrame):
        ann = ann_input.copy()
    else:
        raise TypeError("ann_input must be None, path, or DataFrame")

    data = ProcessedData(M=M, pheno=pheno, ann=ann)
    logger.info(f"Loaded {data.M.shape[1]} samples, {data.M.shape[0]} CpGs.")
    return data
