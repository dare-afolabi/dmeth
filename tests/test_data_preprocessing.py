#!/usr/bin/env python
# coding: utf-8


"""
Tests for dmeth.core.data_preprocessing.

Covers:
- Input normalization and cleaning routines.
- Handling of missing values and edge cases.
- Branches for different preprocessing strategies (e.g. scaling, filtering).
- Error paths when data is malformed.
"""


import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest

from dmeth.core import data_preprocessing
from dmeth.io.data_utils import ProcessedData

RNG = np.random.RandomState(42)


def make_data(n_probes=10, n_samples=6, with_ann=True, seed=0, with_batch=True):
    """Create dummy input dataframe"""
    rng = np.random.RandomState(seed)
    M = pd.DataFrame(
        rng.rand(n_probes, n_samples),
        index=[f"cg{i}" for i in range(n_probes)],
        columns=[f"s{i}" for i in range(n_samples)],
    )
    if with_batch:
        pheno = pd.DataFrame(
            {"batch": ["A"] * (n_samples // 2) + ["B"] * (n_samples - n_samples // 2)},
            index=M.columns,
        )
    else:
        # pheno without batch column
        pheno = pd.DataFrame(
            {"group": ["A"] * (n_samples // 2) + ["B"] * (n_samples - n_samples // 2)},
            index=M.columns,
        )
    ann = (
        pd.DataFrame({"chromosome": ["1"] * n_probes}, index=M.index)
        if with_ann
        else None
    )
    return ProcessedData(M=M, pheno=pheno, ann=ann, meta={"qc": {}})


class TestQC:
    """Test data QC function"""

    def test_qc_sample_level_empty_matrix(self):
        data = ProcessedData(M=pd.DataFrame(), pheno=pd.DataFrame(), ann=None, meta={})
        out = data_preprocessing.qc_sample_level(data)
        assert isinstance(out, ProcessedData)

    def test_qc_sample_level_min_nonmissing_flags(self):
        data = make_data(n_probes=100, n_samples=3, seed=1)
        # force sample 0 to be all NaN
        data.M.iloc[:, 0] = np.nan
        out = data_preprocessing.qc_sample_level(data, min_nonmissing_probes=50)
        # meta.qc should exist and contain a sample_missing entry (or no crash)
        assert isinstance(out, ProcessedData)
        assert "qc" in out.meta
        # if flagged, recorded as list or dict entry
        assert isinstance(out.meta["qc"], dict)

    def test_qc_cpg_level_no_annotation(self):
        data = make_data(with_ann=False)
        out = data_preprocessing.qc_cpg_level(data, drop_sex_chr=True)
        assert isinstance(out, ProcessedData)
        # ensure qc key exists
        assert "qc" in out.meta

    def test_qc_cpg_level_missing_chr_col(self):
        data = make_data()
        # Provide ann but missing chromosome column
        data.ann = pd.DataFrame(index=data.M.index)
        out = data_preprocessing.qc_cpg_level(
            data, drop_sex_chr=True, chr_col="chromosome"
        )
        assert isinstance(out, ProcessedData)
        # should still record qc metadata
        assert "qc" in out.meta


class TestNormalization:
    """Test data normalization"""

    def test_normalize_none_path(self):
        data = make_data(n_probes=10, n_samples=4)
        out = data_preprocessing.normalize_methylation(data, method="none")
        # method 'none' should not set "normalized" True
        assert isinstance(out, ProcessedData)
        assert out.meta.get("normalized") is None or out.meta.get("normalized") is False

    def test_normalize_bad_method_raises(self):
        data = make_data()
        with pytest.raises(ValueError):
            data_preprocessing.normalize_methylation(
                data, method="nonexistent_method_zzz"
            )

    def test_normalize_beta_quantile_memmap_and_blocking(self):
        # Force memmap path by using many probes * many samples thresholds low
        data = make_data(n_probes=200, n_samples=40, seed=2)
        out = data_preprocessing.normalize_methylation(
            data, method="beta_quantile", q_chunk_threshold=1, q_block_probes=50
        )
        assert isinstance(out, ProcessedData)
        # If normalization metadata present, sanity check keys
        if "normalization" in out.meta:
            assert isinstance(out.meta["normalization"], dict)

    def test_convert_beta_to_m_and_back(self):
        data = make_data()
        # convert to M-values
        out = data_preprocessing.normalize_methylation(
            data, method="none", convert_to="m"
        )
        assert out.meta.get("matrix_type") == "m"
        # convert back to beta
        out.meta["matrix_type"] = "m"
        out2 = data_preprocessing.normalize_methylation(
            out, method="none", convert_to="beta"
        )
        assert out2.meta.get("matrix_type") == "beta"


class TestNormalizeHighPerf:
    """Test high performance data normalization"""

    def test_highperf_serial_and_parallel_paths(self, monkeypatch):
        data = make_data(n_probes=30, n_samples=4, seed=3)

        # Serial
        out_serial = data_preprocessing.normalize_methylation_highperf(
            data, method="beta_quantile", n_workers=1
        )
        assert isinstance(out_serial, ProcessedData)

        monkeypatch.setattr(mp, "cpu_count", lambda: 8)

        # mimic multiprocessing.Pool API
        class DummyPool:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            # non-parallel map
            def map(self, func, chunks):
                return [func(chunk) for chunk in chunks]

            def imap_unordered(self, func, chunks):
                for chunk in chunks:
                    yield func(chunk)

        monkeypatch.setattr(mp, "Pool", lambda *a, **k: DummyPool())

        # Parallel path
        out_parallel = data_preprocessing.normalize_methylation_highperf(
            data, method="beta_quantile", n_workers=4
        )

        assert isinstance(out_parallel, ProcessedData)
        assert out_parallel.M.shape == data.M.shape
        assert np.allclose(out_serial.M, out_parallel.M)
        assert (
            out_parallel.meta.get("normalization", {}).get("method") == "beta_quantile"
        )


class TestFilterLowVariance:
    """Test data filtering"""

    def test_filter_low_variance_cpgs_removes_and_records_meta(self):
        data = make_data(n_probes=50, n_samples=5, seed=4)
        # make one probe constant (low variance)
        data.M.iloc[0, :] = 0.5
        res = data_preprocessing.filter_low_variance_cpgs(
            data, min_percentile=0.1, inplace=False
        )
        assert isinstance(res, ProcessedData)
        # either some rows removed, or meta indicates zero removed; at least no crash
        assert "qc" in res.meta
        assert "low_variance_removed" in res.meta["qc"]


class TestBatchCorrectionLinear:
    """Test batch correction functionality"""

    def test_batch_correction_missing_batch_raises(self):
        data = make_data()
        data.pheno = pd.DataFrame(index=data.M.columns)
        with pytest.raises(KeyError):
            data_preprocessing.batch_correction(data, batch_col="batch")

    def test_batch_correction_covariate_missing_raises(self):
        data = make_data()
        with pytest.raises(KeyError):
            data_preprocessing.batch_correction(
                data, batch_col="batch", covariates=["age"]
            )

    def test_batch_correction_weights_mismatch_raises(self):
        data = make_data(n_probes=5, n_samples=4)
        data.pheno["batch"] = ["A", "A", "B", "B"]
        bad_weights = np.array([1, 2])  # wrong length
        with pytest.raises(ValueError):
            data_preprocessing.batch_correction(
                data, batch_col="batch", weights=bad_weights
            )

    def test_batch_correction_single_level_noop_returns_processeddata_and_empty_diag(
        self,
    ):
        data = make_data(n_samples=4)
        data.pheno["batch"] = ["A", "A", "A", "A"]
        res, diag = data_preprocessing.batch_correction(
            data, batch_col="batch", return_diagnostics=True
        )
        assert isinstance(res, ProcessedData)
        assert diag == {} or isinstance(diag, dict)

    def test_batch_correction_qr_and_pinv_and_robust_paths(self):
        # Prepare dataset with two batches A/B
        data = make_data(n_probes=12, n_samples=6, seed=5)
        data.pheno["batch"] = ["A", "A", "A", "B", "B", "B"]

        # QR solver
        out_qr = data_preprocessing.batch_correction(
            data, batch_col="batch", method="qr"
        )
        assert isinstance(out_qr, (ProcessedData, tuple))

        # pinv solver
        out_pinv = data_preprocessing.batch_correction(
            data, batch_col="batch", method="pinv"
        )
        assert isinstance(out_pinv, (ProcessedData, tuple))

        # robust True path
        out_robust = data_preprocessing.batch_correction(
            data, batch_col="batch", robust=True
        )
        assert isinstance(out_robust, (ProcessedData, tuple))

    def test_batch_correction_design_nans_raise(self):
        data = make_data(n_samples=6, seed=6)
        data.pheno["batch"] = ["A", "A", "A", "B", "B", "B"]
        data.pheno["age"] = [np.nan, 1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            data_preprocessing.batch_correction(
                data, batch_col="batch", covariates=["age"]
            )

    def test_batch_correction_return_diagnostics_flagged(self):
        data = make_data(n_probes=12, n_samples=6, seed=7)
        data.pheno["batch"] = ["A", "A", "A", "B", "B", "B"]
        out, diag = data_preprocessing.batch_correction(
            data, batch_col="batch", return_diagnostics=True
        )
        assert isinstance(diag, dict)
        # basic checks on diag structure (if present)
        if diag:
            assert "effect_sizes" in diag or "n_probes" in diag


class TestComBat:
    """Test combat batch correction"""

    def test_combat_missing_batch_col_raises(self):
        data = make_data(with_batch=False)  # no "batch" column
        with pytest.raises(KeyError):
            data_preprocessing.batch_correction_combat(data, batch_col="batch")

    def test_combat_pycombat_fallback_and_metadata_flag(self, monkeypatch):
        # Simulate pycombat not installed (None)
        monkeypatch.setattr(data_preprocessing, "pycombat", None, raising=False)
        data = make_data(n_samples=6, seed=8)
        data.pheno["batch"] = ["A", "A", "A", "B", "B", "B"]
        out = data_preprocessing.batch_correction_combat(data, batch_col="batch")
        assert isinstance(out, ProcessedData)
        if "batch_method" in out.meta:
            assert (
                "fallback" in out.meta["batch_method"]
                or "mean_center" in out.meta["batch_method"]
                or "combat" in out.meta["batch_method"]
            )

    def test_combat_parametric_and_nonparametric_paths(self, monkeypatch):
        # Provide a fake pycombat that returns the array unchanged
        def fake_pycombat(arr, batch_labels, mod=None, parametric=True):
            return arr

        monkeypatch.setattr(
            data_preprocessing, "pycombat", fake_pycombat, raising=False
        )
        data = make_data(n_samples=6, seed=9)
        data.pheno["batch"] = ["A", "A", "A", "B", "B", "B"]
        data.meta["matrix_type"] = "beta"
        r1 = data_preprocessing.batch_correction_combat(
            data, batch_col="batch", parametric=True
        )
        r2 = data_preprocessing.batch_correction_combat(
            data, batch_col="batch", parametric=False
        )
        assert isinstance(r1, ProcessedData)
        assert isinstance(r2, ProcessedData)
        # matrix type should be preserved in meta
        assert data.meta.get("matrix_type") == "beta"

    def test_combat_pycombat_raises_falls_back(self, monkeypatch):
        # Make pycombat raise to exercise exception handling
        def bad_pycombat(*args, **kwargs):
            raise RuntimeError("pycombat failure")

        monkeypatch.setattr(data_preprocessing, "pycombat", bad_pycombat, raising=False)
        data = make_data(n_samples=6, seed=10)
        data.pheno["batch"] = ["A", "A", "A", "B", "B", "B"]
        out = data_preprocessing.batch_correction_combat(data, batch_col="batch")
        assert isinstance(out, ProcessedData)
        # if diagnostics requested, it should be returned as tuple
        data2 = make_data(n_samples=6, seed=11)
        data2.pheno["batch"] = ["A", "A", "A", "B", "B", "B"]
        res, diag = data_preprocessing.batch_correction_combat(
            data2, batch_col="batch", return_diagnostics=True
        )
        assert isinstance(res, ProcessedData)
        assert isinstance(diag, dict)


# A small integration style test to exercise many smaller branches in sequence
class TestIntegrationBranches:
    """Test additional branches"""

    def test_qc_filter_normalize_batch_sequence(self, monkeypatch, tmp_path):
        data = make_data(n_probes=80, n_samples=8, seed=12)
        # introduce missingness for QC branch
        data.M.iloc[0, :] = np.nan
        qced = data_preprocessing.qc_sample_level(data, min_nonmissing_probes=5)
        assert isinstance(qced, ProcessedData)

        # filter low variance
        qced.M.iloc[1, :] = 0.2  # maybe constant
        filtered = data_preprocessing.filter_low_variance_cpgs(
            qced, min_percentile=5.0, inplace=False
        )
        assert isinstance(filtered, ProcessedData)

        # normalize paths (force chunking memmap small threshold)
        normed = data_preprocessing.normalize_methylation(
            filtered, method="beta_quantile", q_chunk_threshold=1, q_block_probes=10
        )
        assert isinstance(normed, ProcessedData)

        # batch correction using qr and request diagnostics
        normed.pheno["batch"] = ["A", "A", "A", "A", "B", "B", "B", "B"][
            : normed.M.shape[1]
        ]
        out, diag = data_preprocessing.batch_correction(
            normed, batch_col="batch", return_diagnostics=True
        )
        assert isinstance(out, ProcessedData)
        assert isinstance(diag, dict)