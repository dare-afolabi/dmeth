#!/usr/bin/env python
# coding: utf-8


"""
Tests for dmeth.core.planner.

Covers:
- Planning logic for analysis pipelines.
- Branches for different configuration options.
- Error handling when required inputs are missing.
"""


import pandas as pd
import pytest

from dmeth.core import planner


class DummyConfig:
    """Replicate config"""

    def list_platforms(self, recommended_only=False):
        return pd.DataFrame({"id": ["p1"], "recommended": [True]})

    def list_designs(self):
        return pd.DataFrame({"id": ["d1"]})

    def get_platform(self, pid):
        if pid != "p1":
            raise KeyError("unknown platform")
        return {"id": "p1"}

    def get_design(self, did):
        if did != "d1":
            raise KeyError("unknown design")
        return {"id": "d1"}

    def get_cost_components(self, **kwargs):
        return {"c1": {"cost": 10}}

    def update_platform_cost(self, pid, cost):
        if pid != "p1":
            raise KeyError("no platform")
        if cost < 0:
            raise ValueError("negative cost")

    def add_custom_platform(self, pid, info):
        if "bad" in info:
            raise ValueError("invalid platform")

    def update_regional_pricing(self, mult, region=None):
        if mult <= 0:
            raise ValueError("multiplier must be >0")

    def get_platform_by_budget(self, max_cost):
        if max_cost < 1:
            return pd.DataFrame()
        return pd.DataFrame({"id": ["p1"], "cost": [5]})

    def calculate_sample_size(self, **kw):
        return {"n_per_group": 10}

    def estimate_study_timeline(self, **kw):
        return pd.DataFrame({"phase": ["TOTAL"], "days": [5]})

    def estimate_total_cost(self, **kw):
        return {"total": 100}


@pytest.fixture(autouse=True)
def patch_get_config(monkeypatch):
    monkeypatch.setattr(planner, "get_config", lambda: DummyConfig())


class TestPlannerWrappers:
    """Test planner"""

    # success‑path tests
    def test_list_platforms(self):
        df = planner.list_platforms()
        assert not df.empty

    def test_get_platform_valid(self):
        assert planner.get_platform("p1")["id"] == "p1"

    def test_get_design_valid(self):
        assert planner.get_design("d1")["id"] == "d1"

    def test_get_cost_components(self):
        comps = planner.get_cost_components()
        assert "c1" in comps

    def test_update_platform_cost_valid(self):
        planner.update_platform_cost("p1", 10)  # should not raise

    def test_add_custom_platform_valid(self):
        planner.add_custom_platform("p2", {"info": "ok"})  # should not raise

    def test_update_regional_pricing_valid(self):
        planner.update_regional_pricing(1.5)

    def test_get_platform_by_budget_valid(self):
        df = planner.get_platform_by_budget(10)
        assert not df.empty

    def test_calculate_sample_size(self):
        res = planner.calculate_sample_size()
        assert res["n_per_group"] == 10

    def test_estimate_study_timeline(self):
        df = planner.estimate_study_timeline(n_samples=5, platform_id="p1")
        assert "TOTAL" in df["phase"].values

    def test_estimate_total_cost(self):
        res = planner.estimate_total_cost(n_samples=5, platform_id="p1")
        assert res["total"] == 100

    # error‑path tests to cover uncovered lines
    def test_get_platform_invalid(self):
        with pytest.raises(KeyError):
            planner.get_platform("bad")

    def test_get_design_invalid(self):
        with pytest.raises(KeyError):
            planner.get_design("bad")

    def test_update_platform_cost_invalid_platform(self):
        with pytest.raises(KeyError):
            planner.update_platform_cost("bad", 10)

    def test_update_platform_cost_negative_cost(self):
        with pytest.raises(ValueError):
            planner.update_platform_cost("p1", -1)

    def test_add_custom_platform_invalid(self):
        with pytest.raises(ValueError):
            planner.add_custom_platform("p2", {"bad": True})

    def test_update_regional_pricing_invalid(self):
        with pytest.raises(ValueError):
            planner.update_regional_pricing(0)

    def test_get_platform_by_budget_empty(self):
        df = planner.get_platform_by_budget(0)
        assert df.empty
