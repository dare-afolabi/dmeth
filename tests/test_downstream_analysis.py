#!/usr/bin/env python
# coding: utf-8


"""
Tests for dmeth.core.downstream analysis modules.

Covers:
- Annotation, deconvolution, downstream statistics, and signature analysis.
- Branches for different statistical methods.
- Error handling for invalid inputs.
"""


from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dmeth.core.downstream import annotation, deconvolution
from dmeth.core.downstream.annotation import (
    annotate_dms_with_genes,
    correlate_methylation_expression,
    gene_set_enrichment,
    liftover_coordinates,
    pathway_methylation_scores,
)
from dmeth.core.downstream.deconvolution import estimate_cell_composition
from dmeth.core.downstream.downstream_stats import (
    adjust_pvalues,
    compute_delta_beta,
    compute_dms_reproducibility,
    compute_effect_size,
    filter_dms,
    find_dmrs_by_sliding_window,
    stouffer_combined_pvalue,
    summarize_regions,
)
from dmeth.core.downstream.helpers import _clean_chr, summarize_groups
from dmeth.core.downstream.signature import (
    model_dms_for_prediction,
    select_signature_panel,
    validate_signature,
)


class TestDownstreamAnnotation:
    """Test downstream annotation functions"""

    def test_annotate_dms_with_genes(self):
        dms = pd.DataFrame(
            {"chr": ["chr1", "chr2"], "pos": [1000, 2000]},
            index=["CpG1", "CpG2"],
        )
        genes = pd.DataFrame(
            {
                "chr": ["chr1", "chr2"],
                "start": [500, 1500],
                "end": [1500, 2500],
                "gene_symbol": ["GENE1", "GENE2"],
            }
        )

        result = annotate_dms_with_genes(dms, genes)

        assert "nearest_gene" in result.columns
        assert not result["nearest_gene"].isna().all()

    def test_gene_set_enrichment(self):
        gene_list = ["GENE1", "GENE2", "GENE3"]
        gene_sets = {
            "PATHWAY1": ["GENE1", "GENE2", "GENE4", "GENE5"],
            "PATHWAY2": ["GENE3", "GENE6", "GENE7"],
        }

        result = gene_set_enrichment(gene_list, gene_sets=gene_sets)

        assert "term" in result.columns
        assert "pvalue" in result.columns

    def test_pathway_methylation_scores(self):
        beta = pd.DataFrame(
            np.random.beta(2, 2, (100, 10)),
            index=[f"CpG{i}" for i in range(100)],
        )
        annotation = pd.DataFrame(
            {"gene": [f"GENE{i % 20}" for i in range(100)]},
            index=beta.index,
        )
        pathway_db = {
            "PATHWAY1": ["GENE0", "GENE1", "GENE2"],
            "PATHWAY2": ["GENE3", "GENE4", "GENE5"],
        }

        result = pathway_methylation_scores(beta, annotation, pathway_db, method="mean")

        assert len(result) > 0

    def test_correlate_methylation_expression(self):
        beta = pd.DataFrame(
            np.random.randn(50, 10),
            columns=[f"S{i}" for i in range(10)],
        )
        expression = pd.DataFrame(
            np.random.randn(50, 10),
            columns=[f"S{i}" for i in range(10)],
        )

        result = correlate_methylation_expression(beta, expression, method="pearson")

        assert "r" in result.columns
        assert "pval" in result.columns

    @patch("dmeth.core.downstream.annotation.LiftOver")
    def test_liftover_coordinates(self, mock_lifter_class):
        mock_lifter = MagicMock()
        mock_lifter.liftover.return_value = [("1", 1500, "+", None)]
        mock_lifter_class.return_value = mock_lifter

        regions = pd.DataFrame({"chr": ["chr1"], "start": [1000], "end": [2000]})

        result = liftover_coordinates(regions)

        assert "lifted_chr" in result.columns
        assert "lifted" in result.columns

    # mop up
    def test_annotate_empty_dms_or_genes(self):
        dms = pd.DataFrame()
        genes = pd.DataFrame()
        assert annotate_dms_with_genes(dms, genes).empty

    def test_annotate_intervaltree_missing(self, monkeypatch):
        monkeypatch.setattr(annotation, "IntervalTree", None)
        dms = pd.DataFrame({"chr": ["chr1"], "pos": [100]})
        genes = pd.DataFrame(
            {"chr": ["chr1"], "start": [50], "end": [150], "gene_symbol": ["G1"]}
        )
        res = annotate_dms_with_genes(dms, genes)
        assert "nearest_gene" in res.columns

    def test_gene_set_enrichment_empty_list(self):
        df = gene_set_enrichment([])
        assert df.empty

    def test_gene_set_enrichment_all_filtered(self):
        gene_sets = {"path": ["A", "B"]}
        df = gene_set_enrichment(["A"], gene_sets=gene_sets, min_set_size=10)
        assert df.empty

    def test_liftover_coordinates_missing(self, monkeypatch):
        monkeypatch.setattr(annotation, "LiftOver", None)
        with pytest.raises(RuntimeError):
            liftover_coordinates(
                pd.DataFrame({"chr": ["chr1"], "start": [1], "end": [2]})
            )

    def test_annotate_empty_genes(self):
        dms = pd.DataFrame({"chr": ["chr1"], "pos": [100]})
        res = annotate_dms_with_genes(dms, pd.DataFrame())
        assert res["nearest_gene"].isna().all()

    def test_gene_set_enrichment_no_results(self):
        df = gene_set_enrichment(["A"], gene_sets={"path": ["B"]})
        assert df.empty

    def test_liftover_empty_regions(self, monkeypatch):
        monkeypatch.setattr(annotation, "LiftOver", None)
        with pytest.raises(RuntimeError):
            liftover_coordinates(pd.DataFrame())

    def test_annotation_empty_genes(self):
        dms = pd.DataFrame({"chr": ["chr1"], "pos": [100]})
        res = annotate_dms_with_genes(dms, pd.DataFrame())
        assert res["nearest_gene"].isna().all()

    def test_annotation_distance_fallback(self, monkeypatch):
        monkeypatch.setattr(annotation, "IntervalTree", None)
        dms = pd.DataFrame({"chr": ["chr1"], "pos": [200]})
        genes = pd.DataFrame(
            {"chr": ["chr1"], "start": [100], "end": [150], "gene_symbol": ["G1"]}
        )
        res = annotate_dms_with_genes(dms, genes, max_distance=100)
        assert "nearest_gene" in res.columns

    def test_gene_set_enrichment_filtered_out(self):
        df = gene_set_enrichment(["A"], gene_sets={"path": ["B"]}, min_set_size=10)
        assert df.empty

    def test_liftover_empty(self, monkeypatch):
        monkeypatch.setattr(annotation, "LiftOver", None)
        with pytest.raises(RuntimeError):
            liftover_coordinates(pd.DataFrame())


class TestDownstreamStats:
    """Test downstream statistics functions"""

    def test_adjust_pvalues_fdr(self):
        pvals = pd.Series([0.001, 0.01, 0.05, 0.1, 0.5])
        result = adjust_pvalues(pvals, method="fdr_bh")

        assert len(result) == len(pvals)
        assert all(result >= pvals)

    def test_stouffer_combined_pvalue(self):
        pvals = [0.01, 0.02, 0.05]
        result = stouffer_combined_pvalue(pvals)

        assert 0 <= result <= 1

    def test_compute_delta_beta(self):
        mean1 = pd.Series([0.5, 0.6, 0.7])
        mean2 = pd.Series([0.3, 0.4, 0.5])

        result = compute_delta_beta(mean1, mean2)

        assert len(result) == 3
        assert all(result == mean1 - mean2)

    def test_compute_effect_size(self):
        beta1 = pd.DataFrame(np.random.randn(50, 5))
        beta2 = pd.DataFrame(np.random.randn(50, 5))

        result = compute_effect_size(beta1, beta2, method="cohens_d")

        assert len(result) == 50

    def test_filter_dms(self):
        res = pd.DataFrame(
            {
                "logFC": [2.0, -2.5, 0.3, -0.2],
                "padj": [0.001, 0.002, 0.05, 0.3],
            }
        )

        result = filter_dms(res, lfc_thresh=1.0, pval_thresh=0.05)

        assert len(result) == 2

    def test_find_dmrs_by_sliding_window(self):
        dms = pd.DataFrame(
            {
                "logFC": [2.0, 2.1, 2.2, 0.5],
                "padj": [0.001, 0.002, 0.003, 0.3],
            },
            index=["CpG1", "CpG2", "CpG3", "CpG4"],
        )
        annotation = pd.DataFrame(
            {
                "chr": ["chr1", "chr1", "chr1", "chr2"],
                "pos": [1000, 1100, 1200, 5000],
            },
            index=["CpG1", "CpG2", "CpG3", "CpG4"],
        )

        result = find_dmrs_by_sliding_window(dms, annotation, max_gap=500, min_cpgs=2)

        assert "chr" in result.columns
        assert "n_cpgs" in result.columns

    def test_summarize_regions(self):
        dmrs = pd.DataFrame(
            {
                "chr": ["chr1", "chr2"],
                "start": [1000, 2000],
                "end": [2000, 3000],
                "n_cpgs": [5, 8],
                "mean_delta_beta": [0.3, -0.4],
                "min_padj": [0.001, 0.002],
            }
        )

        result = summarize_regions(dmrs)

        assert "n_regions" in result.columns
        assert result["n_regions"].iloc[0] == 2

    def test_compute_dms_reproducibility(self):
        res1 = pd.DataFrame(
            {"logFC": [2.0, -1.5], "padj": [0.001, 0.002]},
            index=["CpG1", "CpG2"],
        )
        res2 = pd.DataFrame(
            {"logFC": [1.8, -1.6], "padj": [0.002, 0.001]},
            index=["CpG1", "CpG2"],
        )

        result = compute_dms_reproducibility(res1, res2)

        assert "n_overlap" in result
        assert "jaccard" in result

    # mop up
    def test_adjust_pvalues_none(self):
        with pytest.raises(ValueError):
            adjust_pvalues(None)

    def test_adjust_pvalues_none_method(self):
        arr = [0.1, 0.2]
        adj = adjust_pvalues(arr, method="none")
        assert list(adj) == arr

    def test_stouffer_combined_empty(self):
        with pytest.raises(ValueError):
            stouffer_combined_pvalue([])

    def test_compute_delta_beta_misaligned(self):
        s1 = pd.Series([0.1, 0.2], index=["cg1", "cg2"])
        s2 = pd.Series([0.3, 0.4], index=["cg3", "cg4"])
        delta = compute_delta_beta(s1, s2)
        assert "cg1" in delta.index

    def test_filter_dms_missing_padj(self):
        df = pd.DataFrame({"logFC": [1.0]}, index=["cg1"])
        with pytest.raises(KeyError):
            filter_dms(df)

    def test_find_dmrs_empty(self):
        df = pd.DataFrame()
        ann = pd.DataFrame()
        res = find_dmrs_by_sliding_window(df, ann)
        assert res.empty

    def test_summarize_regions_empty(self):
        df = pd.DataFrame()
        res = summarize_regions(df)
        assert res["n_regions"].iloc[0] == 0

    def test_compute_dms_reproducibility_empty(self):
        res = compute_dms_reproducibility(pd.DataFrame(), pd.DataFrame())
        assert res["jaccard"] == 0.0

    def test_stouffer_combined_invalid(self):
        with pytest.raises(ValueError):
            stouffer_combined_pvalue([])

    def test_filter_dms_directional(self):
        df = pd.DataFrame({"logFC": [1.0], "padj": [0.01]}, index=["cg1"])
        res = filter_dms(df, direction="hyper")
        assert not res.empty

    def test_downstream_stats_adjust_none(self):
        arr = [0.1, 0.2]
        adj = adjust_pvalues(arr, method="none")
        assert list(adj) == arr

    def test_downstream_stats_stouffer_invalid(self):
        with pytest.raises(ValueError):
            stouffer_combined_pvalue([])


class TestSignature:
    def test_validate_signature_binary(self):
        """Binary classification branch"""
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(20, 50), columns=[f"CpG{i}" for i in range(50)]
        )
        y_train = pd.Series([0] * 10 + [1] * 10)
        X_test = pd.DataFrame(
            np.random.randn(10, 50), columns=[f"CpG{i}" for i in range(50)]
        )
        y_test = pd.Series([0] * 5 + [1] * 5)

        features = [f"CpG{i}" for i in range(10)]

        result = validate_signature(
            X_train, y_train, X_test, y_test, features, method="elasticnet"
        )
        assert "accuracy" in result and "auc" in result

    def test_validate_signature_multiclass(self):
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(30, 20), columns=[f"CpG{i}" for i in range(20)]
        )
        y_train = pd.Series(np.random.randint(0, 3, size=30))
        X_test = pd.DataFrame(
            np.random.randn(15, 20), columns=[f"CpG{i}" for i in range(20)]
        )
        y_test = pd.Series(np.random.randint(0, 3, size=15))

        features = [f"CpG{i}" for i in range(5)]

        result = validate_signature(
            X_train, y_train, X_test, y_test, features, method="elasticnet"
        )
        assert "accuracy" in result or "rmse" in result or "r2" in result

    def test_validate_signature_regression(self):
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(25, 15), columns=[f"CpG{i}" for i in range(15)]
        )
        y_train = pd.Series(np.random.randn(25))
        X_test = pd.DataFrame(
            np.random.randn(10, 15), columns=[f"CpG{i}" for i in range(15)]
        )
        y_test = pd.Series(np.random.randn(10))

        features = [f"CpG{i}" for i in range(5)]

        result = validate_signature(
            X_train, y_train, X_test, y_test, features, method="elasticnet"
        )
        assert "rmse" in result and "r2" in result

    # mop up
    # --- py ---
    def test_select_signature_panel_empty(self):
        df = pd.DataFrame()
        assert select_signature_panel(df) == []

    def test_select_signature_panel_no_pval_or_importance(self):
        df = pd.DataFrame({"x": [1]}, index=["cg1"])
        sel = select_signature_panel(df)
        assert sel == ["cg1"]

    def test_validate_signature_mismatched_y(self):
        Xtr = pd.DataFrame([[0.1], [0.2]], columns=["cg1"])
        ytr = [1]  # length mismatch
        Xte = pd.DataFrame([[0.3]], columns=["cg1"])
        yte = [1]
        with pytest.raises(ValueError):
            validate_signature(Xtr, ytr, Xte, yte, ["cg1"])

    def test_model_dms_for_prediction_bad_method(self):
        beta = pd.DataFrame([[0.1, 0.2]], index=["cg1"], columns=["S1", "S2"])
        labels = pd.Series([0, 1], index=["S1", "S2"])
        with pytest.raises(ValueError):
            model_dms_for_prediction(beta, labels, method="bad")

    def test_model_dms_for_prediction_regression(self):
        beta = pd.DataFrame(
            [[0.1, 0.2, 0.2, 0.3]], index=["cg1"], columns=["S1", "S2", "S1", "S2"]
        )
        labels = pd.Series([1.0, 2.0, 1.0, 2.0], index=["S1", "S2", "S1", "S2"])
        out = model_dms_for_prediction(
            beta, labels, method="random_forest", task="regression", n_splits=2
        )
        assert "test_rmse" in out

    def test_model_dms_for_prediction_multiclass(self):
        beta = pd.DataFrame(
            [[0.1, 0.2, 0.2, 0.3, 0.1, 0.2]],
            index=["cg1"],
            columns=["S1", "S2", "S1", "S2", "S1", "S2"],
        )
        labels = pd.Series(
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0], index=["S1", "S2", "S1", "S2", "S1", "S2"]
        )
        out = model_dms_for_prediction(
            beta, labels, method="random_forest", task="classification", n_splits=2
        )
        assert "test_auc" in out

    def test_signature_validate_mismatched_y(self):
        Xtr = pd.DataFrame([[0.1], [0.2]], columns=["cg1"])
        ytr = [1]
        Xte = pd.DataFrame([[0.3]], columns=["cg1"])
        yte = [1]
        with pytest.raises(ValueError):
            validate_signature(Xtr, ytr, Xte, yte, ["cg1"])


class TestDownstreamHelpers:
    """Test helper functions"""

    def test_clean_chr(self):
        assert _clean_chr("1") == "chr1"
        assert _clean_chr("chr1") == "chr1"
        assert _clean_chr("X") == "chrX"
        assert _clean_chr("chrx") == "chrX"

    def test_summarize_groups(self):
        beta = pd.DataFrame(
            np.random.randn(50, 10), columns=[f"S{i}" for i in range(10)]
        )
        groups = pd.Series(["A"] * 5 + ["B"] * 5, index=beta.columns)

        result = summarize_groups(beta, groups)

        assert "mean_A" in result.columns
        assert "mean_B" in result.columns
        assert "var_A" in result.columns

    # mop up
    def test_clean_chr_none(self):
        assert _clean_chr(None) is None

    def test_summarize_groups_empty(self):
        df = pd.DataFrame()
        groups = pd.Series([], dtype=str)
        assert summarize_groups(df, groups).empty

    def test_summarize_groups_missing_group(self):
        beta = pd.DataFrame([[0.1]], index=["cg1"], columns=["S1"])
        groups = pd.Series({"S2": "A"})
        res = summarize_groups(beta, groups)
        assert "mean_A" in res.columns


class TestDeconvolution:
    """Test cell-type deconvolution"""

    def test_estimate_cell_composition(self):
        np.random.seed(42)
        beta = pd.DataFrame(
            np.random.beta(2, 2, (100, 10)),
            index=[f"CpG{i}" for i in range(100)],
        )
        ref_profiles = pd.DataFrame(
            np.random.beta(2, 2, (100, 3)),
            index=[f"CpG{i}" for i in range(100)],
            columns=["CellType1", "CellType2", "CellType3"],
        )

        result = estimate_cell_composition(beta, ref_profiles, n_jobs=1)

        assert result.shape[0] == beta.shape[1]
        assert result.shape[1] == ref_profiles.shape[1]
        # Check proportions sum to ~1
        assert all(np.abs(result.sum(axis=1) - 1.0) < 0.01)

    # mop up
    def test_estimate_cell_composition_empty_inputs(self):
        assert estimate_cell_composition(pd.DataFrame(), pd.DataFrame()).empty

    def test_estimate_cell_composition_no_overlap(self):
        beta = pd.DataFrame([[0.1]], index=["cg1"], columns=["S1"])
        ref = pd.DataFrame([[0.1]], index=["cg2"], columns=["T1"])
        with pytest.raises(ValueError):
            estimate_cell_composition(beta, ref)

    def test_estimate_cell_composition_serial_fallback(self, monkeypatch):
        monkeypatch.setattr(deconvolution, "joblib", None)
        beta = pd.DataFrame([[0.1, 0.2]], index=["cg1"], columns=["S1", "S2"])
        ref = pd.DataFrame([[0.1]], index=["cg1"], columns=["T1"])
        res = estimate_cell_composition(beta, ref)
        assert "T1" in res.columns

    def test_deconvolution_empty_inputs(self):
        assert estimate_cell_composition(pd.DataFrame(), pd.DataFrame()).empty

    def test_deconvolution_parallel(self, monkeypatch):
        # Fake joblib with Parallel to hit parallel branch
        class DummyParallel:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def __call__(self, tasks):
                return [[1.0], [1.0]]

        monkeypatch.setattr(
            deconvolution,
            "joblib",
            type(
                "J",
                (),
                {"Parallel": lambda *a, **k: DummyParallel(), "delayed": lambda f: f},
            ),
        )
        beta = pd.DataFrame([[0.1, 0.2]], index=["cg1"], columns=["S1", "S2"])
        ref = pd.DataFrame([[0.1]], index=["cg1"], columns=["T1"])
        res = estimate_cell_composition(beta, ref, n_jobs=2)
        assert "T1" in res.columns
