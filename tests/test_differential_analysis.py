#!/usr/bin/env python
# coding: utf-8


"""
Tests for dmeth.core.analysis modules.

Covers:
- Core analysis functions and statistical outputs.
- Postprocessing and preparation steps.
- Validation routines for methylation data.
"""


import numpy as np
import pandas as pd
import pytest

import dmeth.core.analysis.core_analysis  # noqa: F401
from dmeth.core.analysis import validation
from dmeth.core.analysis.core_analysis import (
    _add_group_means,
    _estimate_smyth_prior,
    _moderated_variance,
    _winsorize_array,
    fit_differential,
    fit_differential_chunked,
)
from dmeth.core.analysis.postprocessing import (
    get_significant_cpgs,
    summarize_differential_results,
)
from dmeth.core.analysis.preparation import (
    filter_cpgs_by_missingness,
    filter_min_per_group,
    impute_missing_values,
)
from dmeth.core.analysis.validation import (
    build_design,
    check_analysis_memory,
    validate_alignment,
    validate_contrast,
)


class TestCoreAnalysis:
    """Test core differential analysis functions"""

    def test_winsorize_array_basic(self):
        arr = np.array([1, 35, 40, 45, 50, 55, 60, 65, 37, 42, 47, 52, 57, 62, 64, 100])
        result = _winsorize_array(arr, lower=0.1, upper=0.9)
        assert result.max() < 100
        assert result.min() > 1

    def test_winsorize_with_nan(self):
        arr = np.array(
            [1, 35, 40, 45, np.nan, 55, 60, 65, 37, 42, 47, np.nan, 57, 62, 64, 100]
        )
        result = _winsorize_array(arr)

        assert np.isnan(result[4])
        assert np.isnan(result[11])

    def test_winsorize_empty_array(self):
        arr = np.array([])
        result = _winsorize_array(arr)

        assert len(result) == 0

    def test_winsorize_invalid_bounds(self):
        with pytest.raises(ValueError):
            _winsorize_array([1, 2, 3], lower=0.9, upper=0.1)

    def test_winsorize_invalid_bounds_and_empty(self):
        with pytest.raises(ValueError):
            _winsorize_array([1, 2, 3], lower=0.9, upper=0.1)
        arr = np.array([np.nan, np.nan])
        out = _winsorize_array(arr)
        assert np.isnan(out).all()

    def test_estimate_smyth_prior_edge_cases(self):
        with pytest.raises(ValueError):
            _estimate_smyth_prior(np.array([np.nan]))
        # Single variance
        prior = _estimate_smyth_prior(np.array([1.0]))
        assert prior.var_prior == 1.0

    def test_estimate_smyth_prior(self):
        variances = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        prior = _estimate_smyth_prior(variances, robust=True)
        assert prior.df_prior > 0
        assert prior.var_prior > 0

    def test_estimate_smyth_prior_empty(self):
        with pytest.raises(ValueError):
            _estimate_smyth_prior(np.array([np.nan]))

    def test_moderated_variance(self):
        sample_var = np.array([1.0, 2.0, 3.0])
        df_residual = 10
        df_prior = 5.0
        var_prior = 1.5
        result = _moderated_variance(sample_var, df_residual, df_prior, var_prior)
        assert all(result > 0)
        assert len(result) == len(sample_var)

    def test_moderated_variance_invalid(self):
        with pytest.raises(ValueError):
            _moderated_variance([1], df_residual=[0], df_prior=1, var_prior=1)

    def test_moderated_variance_invalids(self):
        with pytest.raises(ValueError):
            _moderated_variance([1], df_residual=[0], df_prior=1, var_prior=1)
        with pytest.raises(ValueError):
            _moderated_variance([1], df_residual=[1], df_prior=0, var_prior=1)
        with pytest.raises(ValueError):
            _moderated_variance([1], df_residual=[1], df_prior=1, var_prior=0)

    def test_add_group_means_errors(self):
        df = pd.DataFrame(np.random.rand(2, 2), columns=["a", "b"])
        groups = pd.Series(["x", "y"], index=["c", "d"])
        with pytest.raises(ValueError):
            _add_group_means(df, groups)

    def test_add_group_means_misaligned(self):
        df = pd.DataFrame(np.random.rand(2, 2), columns=["a", "b"])
        groups = pd.Series(["x", "y"], index=["c", "d"])
        with pytest.raises(ValueError):
            _add_group_means(df, groups)

    def test_fit_differential_basic(self):
        np.random.seed(42)
        M = pd.DataFrame(
            np.random.randn(100, 20), index=[f"CpG{i}" for i in range(100)]
        )

        group_labels = ["control"] * 10 + ["treatment"] * 10
        design = build_design(pd.DataFrame({"group": group_labels}))
        contrast = np.array([0, 1])

        result = fit_differential(M, design, contrast=contrast)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert set(["logFC", "t", "pval", "padj", "s2_post", "d0"]).issubset(
            result.columns
        )
        assert (result["padj"] >= 0).all() and (result["padj"] <= 1).all()
        assert (result["pval"] >= 0).all() and (result["pval"] <= 1).all()
        assert (result["d0"] > 0).all()
        assert (result["s2_post"] > 0).all()
        assert (result["padj"] >= result["pval"]).all()
        assert not result["padj"].isna().any()

    def test_fit_differential_with_missing(self):
        np.random.seed(42)
        M = pd.DataFrame(
            np.random.randn(100, 20), index=[f"CpG{i}" for i in range(100)]
        )
        M.iloc[:9, 5:9] = np.nan
        M.iloc[90:, 15:] = np.nan

        group_labels = ["control"] * 10 + ["treatment"] * 10
        design = build_design(pd.DataFrame({"group": group_labels}))
        contrast = np.array([0, 1])

        result = fit_differential(M, design, contrast=contrast, use_numba=False)

        assert 50 <= len(result) <= 100
        assert not result["padj"].isna().any()
        assert (result["padj"] >= result["pval"]).all()
        assert (result["d0"] > 0).all()

    def test_fit_differential_invalid_inputs(self):
        M = pd.DataFrame(np.random.rand(2, 2))
        design = pd.DataFrame(np.ones((2, 1)))
        with pytest.raises(ValueError):
            fit_differential(M, design)

    def test_fit_differential_invalid_shrink(self):
        M = pd.DataFrame(np.random.rand(2, 2))
        design = pd.DataFrame(np.ones((2, 1)))
        with pytest.raises(ValueError):
            fit_differential(M, design)
        # Invalid shrink
        design = pd.DataFrame({"intercept": [1, 1], "group": [0, 1]})
        with pytest.raises(ValueError):
            fit_differential(M, design, shrink="bad")

    def test_fit_differential_chunked_all_fail(self):
        M = pd.DataFrame(np.random.rand(2, 2))
        design = pd.DataFrame({"intercept": [1, 1], "group": [0, 1]})
        with pytest.raises(RuntimeError):
            fit_differential_chunked(M, design, chunk_size=1, use_numba=False)

    def test_fit_differential_chunked(self):
        np.random.seed(42)
        n_cpgs = 60_000
        M = pd.DataFrame(
            np.random.randn(n_cpgs, 20), index=[f"CpG{i:06d}" for i in range(n_cpgs)]
        )

        group_labels = ["control"] * 10 + ["treatment"] * 10
        design = build_design(pd.DataFrame({"group": group_labels}))
        contrast = np.array([0, 1])

        result = fit_differential_chunked(
            M, design, contrast=contrast, chunk_size=2e4, verbose=True
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6e4
        assert set(["logFC", "t", "pval", "padj", "s2_post", "d0"]).issubset(
            result.columns
        )
        assert (result["padj"] >= result["pval"]).all()

    # mop up
    # _estimate_smyth_prior
    def test_smyth_prior_varlog_le_zero(self):
        variances = np.array([1.0, 1.0])  # identical → var_log=0
        prior = _estimate_smyth_prior(variances)
        assert prior.df_prior > 0

    def test_smyth_prior_root_scalar_failure(self, monkeypatch):
        # Monkeypatch root_scalar to raise
        from dmeth.core.analysis import core_analysis as ca

        monkeypatch.setattr(
            ca,
            "root_scalar",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")),
        )
        prior = _estimate_smyth_prior(np.array([1.0, 2.0]))
        assert prior.df_prior > 0

    # fit_differential slow loop
    def test_fit_differential_slow_loop(self):
        design = pd.DataFrame(
            {"intercept": [1, 1, 1, 1], "group": [0, 0, 1, 1]},
            index=["S1", "S2", "S3", "S4"],
        )
        M = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            columns=["S1", "S2", "S3", "S4"],
        )
        res = fit_differential(M, design, use_numba=False)
        assert "s2_post" in res.columns

    # shrinkage branches
    def test_fit_differential_shrink_median(self):
        design = pd.DataFrame(
            {"intercept": [1, 1, 1, 1], "group": [0, 0, 1, 1]},
            index=["S1", "S2", "S3", "S4"],
        )
        M = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            columns=["S1", "S2", "S3", "S4"],
        )
        res = fit_differential(M, design, shrink="median")
        assert "s2_post" in res.columns

    def test_fit_differential_shrink_smyth(self):
        design = pd.DataFrame(
            {"intercept": [1, 1, 1, 1], "group": [0, 0, 1, 1]},
            index=["S1", "S2", "S3", "S4"],
        )
        M = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            columns=["S1", "S2", "S3", "S4"],
        )
        res = fit_differential(M, design, shrink="smyth")
        assert "s2_post" in res.columns

    def test_fit_differential_shrink_none(self):
        design = pd.DataFrame(
            {"intercept": [1, 1, 1, 1], "group": [0, 0, 1, 1]},
            index=["S1", "S2", "S3", "S4"],
        )
        M = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            columns=["S1", "S2", "S3", "S4"],
        )
        res = fit_differential(M, design, shrink="none")
        assert "s2_post" in res.columns

    def test_fit_differential_shrink_auto(self):
        design = pd.DataFrame(
            {"intercept": [1, 1, 1, 1], "group": [0, 0, 1, 1]},
            index=["S1", "S2", "S3", "S4"],
        )
        M = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            columns=["S1", "S2", "S3", "S4"],
        )
        res = fit_differential(M, design, shrink="auto")
        assert "s2_post" in res.columns

    def test_fit_differential_shrink_invalid(self):
        M = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
        design = pd.DataFrame({"intercept": [1, 1], "group": [0, 1]})
        with pytest.raises(ValueError):
            fit_differential(M, design, shrink="bad")

    # contrast handling
    def test_fit_differential_both_contrast_and_matrix(self):
        M = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
        design = pd.DataFrame({"intercept": [1, 1], "group": [0, 1]})
        with pytest.raises(ValueError):
            fit_differential(
                M, design, contrast=np.array([1, 0]), contrast_matrix=np.eye(1)
            )

    def test_fit_differential_f_test(self):
        design = pd.DataFrame(
            {"intercept": [1, 1, 1, 1], "group": [0, 0, 1, 1]},
            index=["S1", "S2", "S3", "S4"],
        )
        M = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            columns=["S1", "S2", "S3", "S4"],
        )
        res = fit_differential(M, design, contrast_matrix=np.eye(2))
        assert "F" in res.columns

    # group means heuristic
    def test_fit_differential_group_means(self):
        design = pd.DataFrame(
            {"intercept": [1, 1, 1, 1], "group": [0, 0, 1, 1]},
            index=["S1", "S2", "S3", "S4"],
        )
        M = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            columns=["S1", "S2", "S3", "S4"],
        )
        group_labels = pd.Series([0, 0, 1, 1], index=M.columns)
        res = fit_differential(M, design, group_labels=group_labels)
        assert any(col.startswith("mean_") for col in res.columns)

    # residuals path
    def test_fit_differential_return_residuals(self):
        design = pd.DataFrame(
            {"intercept": [1, 1, 1, 1], "group": [0, 0, 1, 1]},
            index=["S1", "S2", "S3", "S4"],
        )
        M = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            columns=["S1", "S2", "S3", "S4"],
        )
        res, resid = fit_differential(M, design, return_residuals=True, use_numba=False)
        assert isinstance(resid, pd.DataFrame)


class TestAnalysisPostprocessing:
    """Test analysis postprocessing functions"""

    def test_summarize_differential_results(self):
        res = pd.DataFrame(
            {
                "logFC": [1.0, -1.5, 0.5, -0.3],
                "pval": [0.001, 0.002, 0.05, 0.3],
                "padj": [0.01, 0.02, 0.1, 0.4],
                "s2": [1.0, 1.2, 0.8, 0.9],
                "s2_post": [0.9, 1.1, 0.7, 0.85],
                "d0": [5.0, 5.0, 5.0, 5.0],
            }
        )

        summary = summarize_differential_results(res, pval_thresh=0.05)

        assert "total_tested" in summary
        assert "significant" in summary
        assert summary["total_tested"] == 4
        assert summary["significant"] == 2

    def test_summarize_invalid_thresh(self):
        with pytest.raises(ValueError):
            summarize_differential_results(pd.DataFrame(), pval_thresh=2)

    def test_summarize_empty_and_invalid_thresh(self):
        df = pd.DataFrame()
        summary = summarize_differential_results(df)
        assert summary["total_tested"] == 0
        with pytest.raises(ValueError):
            summarize_differential_results(df, pval_thresh=2)

    def test_get_significant_cpgs_basic(self):
        res = pd.DataFrame(
            {
                "logFC": [2.0, -2.5, 0.3, -0.2],
                "padj": [0.001, 0.002, 0.05, 0.3],
            },
            index=["CpG1", "CpG2", "CpG3", "CpG4"],
        )

        cpgs = get_significant_cpgs(res, lfc_thresh=1.0, pval_thresh=0.05)
        assert len(cpgs) == 2
        assert "CpG1" in cpgs
        assert "CpG2" in cpgs

    def test_get_significant_cpgs_direction(self):
        res = pd.DataFrame(
            {
                "logFC": [2.0, -2.5, 1.5, -0.2],
                "padj": [0.001, 0.002, 0.003, 0.3],
            },
            index=["CpG1", "CpG2", "CpG3", "CpG4"],
        )

        hyper = get_significant_cpgs(res, direction="hyper", lfc_thresh=1.0)
        assert len(hyper) == 2
        assert "CpG1" in hyper

        hypo = get_significant_cpgs(res, direction="hypo", lfc_thresh=1.0)
        assert len(hypo) == 1
        assert "CpG2" in hypo

    def test_get_significant_cpgs_with_summary(self):
        res = pd.DataFrame(
            {"logFC": [2.0, -2.5], "padj": [0.001, 0.002]},
            index=["CpG1", "CpG2"],
        )

        result = get_significant_cpgs(res, return_summary=True)
        assert isinstance(result, dict)
        assert "n_significant" in result
        assert "cpgs" in result

    def test_get_significant_delta_beta_error(self):
        df = pd.DataFrame({"logFC": [1], "padj": [0.01]}, index=["cpg1"])
        with pytest.raises(ValueError):
            get_significant_cpgs(df, delta_beta_thresh=0.1)

    def test_get_significant_missing_and_delta_beta(self):
        df = pd.DataFrame({"logFC": [1], "padj": [0.01]}, index=["cpg1"])
        # Missing column
        with pytest.raises(KeyError):
            get_significant_cpgs(df, lfc_col="missing")
        # Delta-beta error
        with pytest.raises(ValueError):
            get_significant_cpgs(df, delta_beta_thresh=0.1)


class TestAnalysisPreparation:
    """Test data preparation functions"""

    def test_filter_cpgs_by_missingness(self):
        M = pd.DataFrame(
            {
                "S1": [0.5, np.nan, 0.3, 0.4],
                "S2": [0.6, 0.7, np.nan, 0.5],
                "S3": [np.nan, np.nan, 0.8, 0.6],
            },
            index=["CpG1", "CpG2", "CpG3", "CpG4"],
        )

        filtered, n_removed, n_kept = filter_cpgs_by_missingness(
            M, max_missing_rate=0.5
        )
        assert n_kept > 0
        assert n_removed >= 0

    def test_filter_cpgs_invalid_rate_and_group_errors(self):
        M = pd.DataFrame(np.random.rand(3, 3))
        with pytest.raises(ValueError):
            filter_cpgs_by_missingness(M, max_missing_rate=2)
        with pytest.raises(ValueError):
            filter_cpgs_by_missingness(M, min_samples_per_group=1)

    def test_impute_missing_values_mean(self):
        M = pd.DataFrame(
            {
                "S1": [0.5, np.nan, 0.3],
                "S2": [0.6, 0.7, np.nan],
            },
            index=["CpG1", "CpG2", "CpG3"],
        )

        result = impute_missing_values(M, method="mean")
        assert not result.isna().any().any()

    def test_impute_missing_values_median(self):
        M = pd.DataFrame({"S1": [0.5, np.nan, 0.3], "S2": [0.6, 0.7, np.nan]})

        result = impute_missing_values(M, method="median")
        assert not result.isna().any().any()

    def test_impute_invalid_and_knn(self):
        M = pd.DataFrame(np.random.rand(5, 5))
        with pytest.raises(ValueError):
            impute_missing_values(M, method="bad")
        # Force NaNs and knn path
        M.iloc[0, 0] = np.nan
        out = impute_missing_values(M, method="knn", use_sample_knn=True)
        assert not out.isna().any().any()

    def test_filter_min_per_group(self):
        M = pd.DataFrame(
            {
                "S1": [0.5, np.nan, 0.3],
                "S2": [0.6, 0.7, np.nan],
                "S3": [np.nan, 0.8, 0.9],
                "S4": [0.4, np.nan, 0.5],
            }
        )
        groups = pd.Series(["A", "A", "B", "B"], index=M.columns)

        result = filter_min_per_group(M, groups, min_per_group=2, verbose=True)
        assert len(result) > 0

    def test_filter_min_per_group_misaligned(self):
        M = pd.DataFrame(np.random.rand(3, 3), columns=["a", "b", "c"])
        groups = pd.Series(["x", "y"], index=["a", "b"])
        with pytest.raises(ValueError):
            filter_min_per_group(M, groups)


class TestAnalysisValidation:
    """Test validation functions"""

    def test_check_analysis_memory(self):
        M = pd.DataFrame(np.random.randn(1000, 100))
        result = check_analysis_memory(M)

        assert "data_gb" in result
        assert "peak_gb" in result
        assert "available_gb" in result

    def test_build_design_basic(self):
        design = ["A", "A", "A", "B", "B", "B"]
        result = build_design(pd.DataFrame({"group": design}))

        assert result.shape == (6, 2)
        assert (result["intercept"] == 1).all()

    def test_build_design_invalid(self):
        result = build_design(pd.DataFrame({"group": ["A", "A", "A"]}))
        assert result.shape == (3, 1)

    def test_build_design_errors(self):
        with pytest.raises(TypeError):
            build_design("string")
        build_design(pd.DataFrame({"group": [np.nan, "A"]}))
        build_design(pd.DataFrame({"group": ["A"]}))
        build_design(pd.DataFrame({"group": ["A", "B", "C"]}))

    def test_validate_contrast_basic(self):
        design = np.array([[1, 0], [1, 0], [1, 1], [1, 1]])
        contrast = np.array([0, 1])

        result = validate_contrast(design, contrast)
        assert len(result) == 2

    def test_validate_contrast_string(self):
        design = np.array([[1, 0], [1, 0], [1, 1], [1, 1]])
        contrast = "treatment-control"

        result = validate_contrast(design, contrast)
        assert len(result) == 2

    def test_validate_contrast_errors(self):
        X = np.eye(2)
        with pytest.raises(ValueError):
            validate_contrast(X, "badstring")
        with pytest.raises(ValueError):
            validate_contrast(X, [0, 0])

    def test_validate_contrast_zero_vector(self):
        X = np.eye(3)
        with pytest.raises(ValueError):
            validate_contrast(X, [0, 0, 0])

    def test_validate_alignment_basic(self):
        M = pd.DataFrame(np.random.randn(10, 4), columns=["S1", "S2", "S3", "S4"])
        design = np.random.randn(4, 2)

        cols, des, paired = validate_alignment(M, design, None, None)
        assert len(cols) == 4

    def test_validate_alignment_duplicate_cols(self):
        df = pd.DataFrame(np.random.rand(2, 2), columns=["s1", "s1"])
        X = np.eye(2)
        with pytest.raises(KeyError):
            validate_alignment(df, X)

    def test_validate_alignment_missing_sample(self):
        M = pd.DataFrame(np.random.randn(10, 4), columns=["S1", "S2", "S3", "S4"])
        design = np.random.randn(4, 2)
        pheno = pd.DataFrame(index=["S1", "S2", "S3"])  # Missing S4

        with pytest.raises(KeyError):
            validate_alignment(M, design, sample_names=pheno.index, paired_ids=None)

    def test_validate_alignment_errors(self):
        df = pd.DataFrame(np.random.rand(2, 2), columns=["s1", "s1"])
        X = np.eye(2)
        with pytest.raises(KeyError):
            validate_alignment(df, X)
        # Paired IDs length mismatch
        df = pd.DataFrame(np.random.rand(2, 2), columns=["s1", "s2"])
        with pytest.raises(ValueError):
            validate_alignment(df, X, paired_ids=["id1"])

    # mop up tests
    # check_analysis_memory
    def test_memory_psutil_available_memoryerror(self, monkeypatch):
        class VM:
            available = 0

        class PS:
            @staticmethod
            def virtual_memory():
                return VM()

        monkeypatch.setattr(validation, "psutil", PS)

        M = pd.DataFrame(np.zeros((10, 10)))  # small but nonzero
        with pytest.raises(MemoryError):
            check_analysis_memory(M)

    def test_memory_psutil_available_warn_and_return(self, monkeypatch):
        # warn_threshold_gb tiny to trigger warning
        class VM:
            available = 10 * (1024**3)  # 10 GB

        class PS:
            @staticmethod
            def virtual_memory():
                return VM()

        monkeypatch.setattr(validation, "psutil", PS)

        M = pd.DataFrame(np.zeros((2, 2)))  # tiny DF
        info = check_analysis_memory(M, warn_threshold_gb=1e-12)
        assert set(info.keys()) == {"data_gb", "peak_gb", "available_gb"}
        assert np.isfinite(info["available_gb"])

    # validate_contrast
    def test_contrast_design_matrix_type_error_line(self):
        # Not 2-D ndarray: 1-D array triggers TypeError
        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError):
            validate_contrast(X, np.array([1.0]))

    def test_contrast_ncoef_lt_2_line(self):
        # Single column design → ValueError
        X = np.ones((5, 1))
        with pytest.raises(ValueError):
            validate_contrast(X, np.array([1.0]))

    def test_contrast_numeric_conversion_failure_line(self):
        X = np.ones((4, 2))
        with pytest.raises(TypeError):
            validate_contrast(X, contrast=["a", "b"])

    def test_contrast_string_path_wrong_ncoef_line(self):
        X = np.ones((6, 3))
        with pytest.raises(ValueError):
            validate_contrast(X, contrast="A-B")

    def test_contrast_string_ref_flips_sign_line(self):
        group = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        X = np.column_stack([np.ones_like(group), group])
        c = validate_contrast(X, "ref-treat")
        # Expect treat - ref with flip for 'ref' → effectively -(treat - ref) → [0, -1]
        assert c.shape == (2,)
        assert c[0] == 0.0 and c[1] == -1.0

    def test_contrast_length_mismatch_line(self):
        # Contrast length != n_coef
        X = np.ones((5, 3))
        with pytest.raises(ValueError):
            validate_contrast(X, np.array([1.0, -1.0]))

    # validate_alignment
    def test_alignment_type_errors_line(self):
        # data not DataFrame → TypeError
        with pytest.raises(TypeError):
            validate_alignment(data=np.zeros((2, 2)), design_matrix=np.ones((2, 2)))
        # design_matrix not 2-D ndarray → TypeError
        with pytest.raises(TypeError):
            validate_alignment(
                data=pd.DataFrame(np.zeros((2, 2))), design_matrix=np.array([1.0, 2.0])
            )

    def test_alignment_sample_count_mismatch_line(self):
        # design rows = n_samples=3; data has 2 columns → ValueError
        design = np.ones((3, 2))
        data = pd.DataFrame(np.zeros((4, 2)), columns=["A", "B"])
        with pytest.raises(ValueError):
            validate_alignment(data=data, design_matrix=design)

    def test_alignment_duplicate_columns_line(self):
        # immediate duplicate detection when sample_names=None
        design = np.ones((2, 2))
        data = pd.DataFrame(np.zeros((4, 2)), columns=["A", "A"])
        with pytest.raises(KeyError):
            validate_alignment(data=data, design_matrix=design)

    def test_alignment_paired_ids_length_and_nan_line(self):
        design = np.column_stack([np.ones(4), np.array([0, 1, 0, 1], dtype=float)])
        data = pd.DataFrame(np.zeros((2, 4)), columns=["S1", "S2", "S3", "S4"])
        # paired_ids length mismatch → ValueError 308
        with pytest.raises(ValueError):
            validate_alignment(data=data, design_matrix=design, paired_ids=["A", "B"])
        # paired_ids contains NaN → ValueError 310
        with pytest.raises(ValueError):
            validate_alignment(
                data=data, design_matrix=design, paired_ids=["A", "B", np.nan, "D"]
            )

    def test_alignment_subject_not_twice_line(self):
        # Subject counts wrong (one appears once, another thrice) → ValueError
        design = np.column_stack(
            [np.ones(6), np.array([0, 1, 0, 1, 0, 1], dtype=float)]
        )
        data = pd.DataFrame(np.zeros((2, 6)), columns=[f"S{i}" for i in range(6)])
        paired_ids = ["U1", "U1", "U2", "U2", "U2", "U3"]  # U2 thrice, U3 once
        with pytest.raises(ValueError):
            validate_alignment(data=data, design_matrix=design, paired_ids=paired_ids)

    def test_alignment_pair_same_group_line(self):
        # Pairs not balanced across groups → ValueError
        design = np.column_stack(
            [np.ones(4), np.array([0, 0, 1, 1], dtype=float)]
        )  # two pairs but same groups
        data = pd.DataFrame(np.zeros((2, 4)), columns=["S1", "S2", "S3", "S4"])
        paired_ids = [
            "A",
            "A",
            "B",
            "B",
        ]  # each subject has two samples but group sums are 0 and 2
        with pytest.raises(ValueError):
            validate_alignment(data=data, design_matrix=design, paired_ids=paired_ids)
      validate_alignment(data=data, design_matrix=design, paired_ids=paired_ids)