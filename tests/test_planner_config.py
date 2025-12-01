#!/usr/bin/env python
# coding: utf-8


"""
Tests for dmeth.config.config_manager.

Covers:
- Loading and saving configuration files.
- Validation of configuration options.
- Error paths for missing or malformed configs.
"""


import os
import tempfile
from pathlib import Path

import pytest

from dmeth.config.config_manager import (
    PlannerConfig,
    _atomic_write,
    _read_python_literal,
    get_config,
    reset_config,
)


class TestConfigManager:
    """Test configuration management functionality"""

    def setup_method(self):
        reset_config()

    def test_planner_config_singleton(self):
        cfg1 = PlannerConfig()
        cfg2 = PlannerConfig()
        assert cfg1 is cfg2

    def test_atomic_write_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            _atomic_write(path, "test content")
            assert path.read_text() == "test content"

    def test_atomic_write_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "test.txt"
            _atomic_write(path, "content")
            assert path.exists()

    def test_read_python_literal_valid(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("{'key': 'value', 'num': 42}")
            f.flush()
            result = _read_python_literal(Path(f.name))
            assert result == {"key": "value", "num": 42}
            os.unlink(f.name)

    def test_read_python_literal_invalid(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("not a dict")
            f.flush()
            with pytest.raises(
                ValueError, match="Failed to parse Python literal from .+"
            ):
                _read_python_literal(Path(f.name))
            os.unlink(f.name)

    def test_load_csv_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("platform_id,name,n_cpgs,cost_per_sample,processing_days\n")
            f.write("TEST,Test Platform,100000,300.0,7\n")
            f.flush()
            cfg = PlannerConfig()
            cfg.load_file(f.name)
            assert "TEST" in cfg.platforms
            os.unlink(f.name)

    def test_save_json_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            cfg = get_config()
            cfg.save_file(path)
            assert path.exists()
            # Verify it can be loaded back
            cfg2 = PlannerConfig()
            cfg2.load_file(path)
            assert cfg2.platforms == cfg.platforms

    def test_list_platforms_all(self):
        cfg = get_config()
        df = cfg.list_platforms()
        assert len(df) > 0
        assert "name" in df.columns

    def test_list_platforms_recommended_only(self):
        cfg = get_config()
        df = cfg.list_platforms(recommended_only=True)
        assert all(df["recommended"])

    def test_get_platform_valid(self):
        cfg = get_config()
        plat = cfg.get_platform("EPIC")
        assert plat["name"] == "MethylationEPIC v1.0"

    def test_get_platform_invalid(self):
        cfg = get_config()
        with pytest.raises(KeyError):
            cfg.get_platform("NONEXISTENT")

    def test_update_platform_cost(self):
        cfg = get_config()
        cfg.update_platform_cost("EPIC", 600.0)
        assert cfg.platforms["EPIC"]["cost_per_sample"] == 600.0

    def test_add_custom_platform(self):
        cfg = get_config()
        info = {
            "name": "CustomArray",
            "n_cpgs": 500000,
            "cost_per_sample": 400.0,
            "processing_days": 5,
        }
        cfg.add_custom_platform("CUSTOM", info)
        assert "CUSTOM" in cfg.platforms

    def test_update_regional_pricing(self):
        cfg = get_config()
        original = cfg.platforms["EPIC"]["cost_per_sample"]
        cfg.update_regional_pricing(1.2, region="EU")
        updated = cfg.platforms["EPIC"]["cost_per_sample"]
        assert updated == round(original * 1.2, 2)

    def test_get_platform_by_budget(self):
        cfg = get_config()
        df = cfg.get_platform_by_budget(450.0)
        assert all(df["cost_per_sample"] <= 450.0)

    def test_calculate_sample_size(self):
        cfg = get_config()
        result = cfg.calculate_sample_size(
            design_id="two_group",
            platform_id="EPIC",
            effect_size=0.5,
            alpha=0.05,
            power=0.8,
        )
        assert "n_per_group" in result
        assert "total_samples" in result
        assert result["n_per_group"] > 0

    def test_estimate_study_timeline(self):
        cfg = get_config()
        df = cfg.estimate_study_timeline(n_samples=100, platform_id="EPIC")
        assert "phase_id" in df.columns
        assert "estimated_days" in df.columns
        assert "TOTAL" in df["phase_id"].values

    def test_estimate_total_cost(self):
        cfg = get_config()
        result = cfg.estimate_total_cost(n_samples=50, platform_id="EPIC")
        assert "total" in result
        assert "per_sample" in result
        assert "components" in result
        assert result["total"] > 0
        assert "n_per_group" in result
        assert "total_samples" in result
        assert result["n_per_group"] > 0

    def test_estimate_study_timeline(self):
        cfg = get_config()
        df = cfg.estimate_study_timeline(n_samples=100, platform_id="EPIC")
        assert "phase_id" in df.columns
        assert "estimated_days" in df.columns
        assert "TOTAL" in df["phase_id"].values

    def test_estimate_total_cost(self):
        cfg = get_config()
        result = cfg.estimate_total_cost(n_samples=50, platform_id="EPIC")
        assert "total" in result
        assert "per_sample" in result
        assert "components" in result
        assert result["total"] > 0