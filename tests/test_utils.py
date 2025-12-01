#!/usr/bin/env python
# coding: utf-8


"""
Tests for dmeth.utils modules.

Covers:
- Logger configuration and exit paths.
- Plotting utilities and edge cases.
"""


import logging

import matplotlib
import numpy as np
import pandas as pd

from dmeth.io.data_utils import ProcessedData

# Import modules under test
from dmeth.utils import logger as logger_module
from dmeth.utils.logger import ProgressAwareLogger, get_logger
from dmeth.utils.plotting import (
    _new_fig,
    methylation_expression_heatmap,
    pca_plot,
    plot_mean_difference,
    plot_pvalue_qq,
    plot_shrinkage_diagnostics,
    plot_stage,
    plot_volcano,
    pvalue_histogram,
    visualize_dms,
)

matplotlib.use("Agg")


class TestLogger:
    """Test central logger"""

    def test_get_logger_returns_progressawarelogger(self):
        log = get_logger()
        assert isinstance(log, ProgressAwareLogger)
        assert get_logger() is log  # same instance

    def test_progress_methods(self, tmp_path):
        log = get_logger()
        log.handlers.clear()
        log.propagate = False
        logger_module._configure_logger(output_dir=str(tmp_path))

        log.progress("Testing", total=2)
        assert log._pbar is not None
        log.progress_update(1)
        log.info("Message")  # should auto-close
        assert log._pbar is None

    def test_file_logging(self, tmp_path):
        log = logging.getLogger("dmeth")
        log.handlers.clear()
        log.propagate = False
        log = logger_module._configure_logger(output_dir=str(tmp_path))
        log.info("File output test")

        files = list((tmp_path / "log").glob("*.log"))
        assert files, "No log file created"
        content = files[0].read_text()
        assert "File output test" in content


class TestPlotting:
    """Test full suit of plotting functions"""

    def setup_method(self):
        # Minimal fake data
        self.M = pd.DataFrame(
            np.random.randn(5, 3),
            index=[f"cpg{i}" for i in range(5)],
            columns=["s1", "s2", "s3"],
        )
        # Align res index with M for top_hits
        self.df = pd.DataFrame(
            {"logFC": [2, -2, 0.5], "pval": [0.001, 0.02, 0.5]},
            index=self.M.index[:3],
        )
        self.metadata = pd.DataFrame({"Type": ["A", "B", "A"]}, index=self.M.columns)
        self.beta1 = pd.DataFrame(
            np.random.rand(5, 3),
            index=[f"cpg{i}" for i in range(5)],
            columns=self.M.columns,
        )
        self.beta2 = pd.DataFrame(
            np.random.rand(5, 3),
            index=[f"cpg{i}" for i in range(5)],
            columns=self.M.columns,
        )
        # Ensure overlapping sample columns with beta for correlation
        self.expr = pd.DataFrame(
            np.random.rand(5, 3),
            index=self.M.index,
            columns=self.M.columns,
        )

    def test_new_fig(self):
        fig = _new_fig()
        assert fig is not None

    def test_plot_volcano_and_save(self, tmp_path):
        fig = plot_volcano(self.df, save_path=tmp_path / "volcano.png")
        assert fig is not None
        assert (tmp_path / "volcano.png").exists()

    def test_plot_pvalue_qq_and_save(self, tmp_path):
        fig = plot_pvalue_qq(self.df, save_path=tmp_path / "qq.png")
        assert fig is not None
        assert (tmp_path / "qq.png").exists()

    def test_plot_stage_qc_and_save(self, tmp_path):
        figs = plot_stage(
            "qc", self.M, metadata=self.metadata, save_path=tmp_path / "qc.png"
        )
        assert "missing" in figs
        assert "embedding" in figs

    def test_plot_stage_variance(self, tmp_path):
        res = pd.DataFrame(
            {"meanM_1": [0.1, 0.2], "s2": [0.5, 0.2], "s2_post": [0.3, 0.1]}
        )
        figs = plot_stage("variance", self.M, res=res, save_path=tmp_path / "var.png")
        assert "variance" in figs

    def test_plot_stage_differential(self):
        figs = plot_stage("differential", self.M, res=self.df)
        assert "volcano" in figs and "qq" in figs

    def test_plot_stage_top_hits(self):
        figs = plot_stage("top_hits", self.M, res=self.df)
        assert "top_hits" in figs

    def test_plot_stage_correlation(self):
        figs = plot_stage("correlation", self.M)
        assert "correlation" in figs and "dendrogram" in figs

    def test_plot_shrinkage_diagnostics(self):
        s2 = pd.Series([0.1, 0.2, 0.3])
        s2_post = pd.Series([0.05, 0.15, 0.25])
        fig = plot_shrinkage_diagnostics(s2, s2_post, d0=5)
        assert fig is not None

    def test_plot_mean_difference(self):
        fig = plot_mean_difference(self.beta1, self.beta2, top_n=3)
        assert fig is not None

    def test_pvalue_histogram(self):
        fig = pvalue_histogram([0.1, 0.2, 0.3])
        assert fig is not None

    def test_visualize_dms(self, tmp_path):
        df = self.df.copy()
        df["chr"] = ["chr1", "chr2", "chr3"]
        df["pos"] = [100, 200, 300]
        figs = visualize_dms(df, beta=self.M, save_dir=tmp_path)
        assert any(figs.values())

    def test_methylation_expression_heatmap(self):
        fig = methylation_expression_heatmap(self.M, self.expr)
        assert fig is not None

    def test_pca_plot(self):
        ann = pd.DataFrame(index=self.M.index)
        data = ProcessedData(M=self.M, pheno=self.metadata, ann=ann)
        # Should not raise
        pca_plot(data, "Type")