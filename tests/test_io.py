#!/usr/bin/env python
# coding: utf-8


"""
Tests for dmeth.io readers and writers.

This suite covers:
- File format handling (CSV, TSV, XLSX, Feather, Parquet, HDF5).
- Error paths (file not found, unsupported formats, pheno/annotation mismatches).
- Export and save functions (overwrite protection, suffix correction, Excel fallbacks).
- IDAT HDF5 export and analysis report generation.
"""


import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dmeth.io import readers, writers
from dmeth.io.data_utils import (
    ProcessedData,
    _ensure_index_alignment,
    _ensure_index_strings,
)
from dmeth.io.readers import _read, load_methylation_data, load_processed_data
from dmeth.io.writers import export_idat_hdf5, export_results, save_processed_data


class TestDataUtils:
    """Test data utility functions"""

    def test_ensure_index_strings(self):
        df = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        result = _ensure_index_strings(df)
        assert result.index.dtype == object

    def test_ensure_index_alignment_success(self):
        M = pd.DataFrame(np.random.randn(10, 4), columns=["S1", "S2", "S3", "S4"])
        pheno = pd.DataFrame(index=["S1", "S2", "S3", "S4"])
        ann = pd.DataFrame(index=M.index)

        # Should not raise
        _ensure_index_alignment(M, pheno, ann)

    def test_ensure_index_alignment_missing_samples(self):
        M = pd.DataFrame(np.random.randn(10, 4), columns=["S1", "S2", "S3", "S4"])
        pheno = pd.DataFrame(index=["S1", "S2", "S3"])  # Missing S4

        with pytest.raises(KeyError, match="Samples in M but not in pheno"):
            _ensure_index_alignment(M, pheno, None)

    def test_ensure_index_alignment_missing_probes(self):
        M = pd.DataFrame(
            np.random.randn(10, 4),
            index=[f"CpG{i}" for i in range(10)],
            columns=["S1", "S2", "S3", "S4"],
        )
        pheno = pd.DataFrame(index=["S1", "S2", "S3", "S4"])
        ann = pd.DataFrame(index=[f"CpG{i}" for i in range(5)])  # Only 5 CpGs

        with pytest.raises(KeyError, match="Probes in M but not in ann"):
            _ensure_index_alignment(M, pheno, ann)

    def test_processed_data_creation(self):
        M = pd.DataFrame(np.random.randn(10, 4), columns=["S1", "S2", "S3", "S4"])
        pheno = pd.DataFrame({"group": ["A", "A", "B", "B"]}, index=M.columns)
        ann = pd.DataFrame({"chr": ["1"] * 10}, index=M.index)

        data = ProcessedData(M=M, pheno=pheno, ann=ann)

        assert data.M.index.dtype == object
        assert data.pheno.index.dtype == object
        assert "matrix_type" in data.meta

    def test_processed_data_invalid_alignment(self):
        M = pd.DataFrame(np.random.randn(10, 4), columns=["S1", "S2", "S3", "S4"])
        pheno = pd.DataFrame(index=["S1", "S2"])  # Not enough samples

        with pytest.raises(KeyError):
            ProcessedData(M=M, pheno=pheno, ann=None)


class TestReaders:
    """Test data reading functions"""

    def test_read_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            df.to_csv(f.name, index=False)

            result = _read(f.name, index_col=None)
            assert result.shape == df.shape

            Path(f.name).unlink()

    def test_read_excel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.xlsx"
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            # Write directly, let pandas manage writer lifecycle
            pytest.importorskip("openpyxl")
            pytest.importorskip("xlsxwriter")
            df.to_excel(str(path), index=False, engine="xlsxwriter")

            result = _read(path, index_col=None)
            assert result.shape == df.shape

    def test_read_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as f:
            with pytest.raises(ValueError, match="Unsupported format"):
                _read(f.name, index_col=None)
            Path(f.name).unlink()

    def test_load_methylation_data_from_dataframe(self):
        M = pd.DataFrame(
            np.random.randn(50, 10),
            index=[f"CpG{i}" for i in range(50)],
            columns=[f"S{i}" for i in range(10)],
        )
        pheno = pd.DataFrame({"group": ["A"] * 5 + ["B"] * 5}, index=M.columns)

        data = load_methylation_data(M, pheno_input=pheno)

        assert isinstance(data, ProcessedData)
        assert data.M.shape == M.shape

    def test_load_methylation_data_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            M = pd.DataFrame(
                np.random.randn(20, 5),
                index=[f"CpG{i}" for i in range(20)],
                columns=[f"S{i}" for i in range(5)],
            )
            M.to_csv(f.name)

            data = load_methylation_data(f.name)
            assert isinstance(data, ProcessedData)

            Path(f.name).unlink()

    def test_load_processed_data_pickle(self):
        M = pd.DataFrame(np.random.randn(10, 5))
        pheno = pd.DataFrame(index=M.columns)
        data = ProcessedData(M=M, pheno=pheno, ann=None)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pd.to_pickle(data, f.name)
            loaded = load_processed_data(f.name, trusted=True)
            assert isinstance(loaded, ProcessedData)
            Path(f.name).unlink()


class TestWriters:
    """Test data writing functions"""

    def test_export_results_csv(self):
        res = pd.DataFrame(
            {
                "logFC": [1.0, -1.5],
                "pval": [0.001, 0.01],
                "padj": [0.01, 0.05],
            },
            index=["CpG1", "CpG2"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            export_results(res, str(path), format="csv")
            assert path.exists()

    def test_export_results_excel(self):
        res = pd.DataFrame(
            {"logFC": [1.0, -1.5], "pval": [0.001, 0.01]},
            index=["CpG1", "CpG2"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.xlsx"
            export_results(res, str(path), format="excel")
            assert path.exists()

    def test_save_processed_data_csv(self):
        M = pd.DataFrame(np.random.randn(10, 5))
        pheno = pd.DataFrame(index=M.columns)
        data = ProcessedData(M=M, pheno=pheno, ann=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.csv"
            save_processed_data(data, path, format="csv")
            assert path.exists()


class TestIOEdgeCases:
    """Target additional branches in readers and writers"""

    def test_read_untrusted_pickle(self, tmp_path):
        df = pd.DataFrame({"A": [1, 2, 3]})
        pkl = tmp_path / "bad.pkl"
        df.to_pickle(pkl)
        with pytest.raises(ValueError, match="Pickle files are not supported"):
            _read(pkl, index_col=None, trusted=False)

    def test_read_feather_index_out_of_bounds(self, tmp_path):
        pytest.importorskip("pyarrow")
        feather = tmp_path / "data.feather"
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df.to_feather(feather)
        with pytest.raises(IndexError):
            _read(feather, index_col=99)

    def test_read_hdf5_multiple_keys(self, tmp_path):
        pytest.importorskip("tables")
        h5 = tmp_path / "multi.h5"
        df1 = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"B": [4, 5, 6]})
        with pd.HDFStore(h5, "w") as store:
            store.put("M", df1)
            store.put("pheno", df2)
        with pytest.raises(ValueError, match="multiple keys"):
            _read(h5, index_col=None)

    def test_load_methylation_data_pheno_mismatch(self):
        M = pd.DataFrame([[1, 2], [3, 4]], columns=["S1", "S2"], index=["CpG1", "CpG2"])
        pheno = pd.DataFrame(index=["S1"])  # missing S2
        with pytest.raises(KeyError, match="Pheno missing sample metadata"):
            load_methylation_data(M, pheno_input=pheno)

    def test_export_results_overwrite_protection(self, tmp_path):
        path = tmp_path / "results.csv"
        df = pd.DataFrame({"logFC": [1.0]}, index=["CpG1"])
        df.to_csv(path)
        with pytest.raises(FileExistsError):
            export_results(df, str(path), format="csv", overwrite=False)

    def test_export_results_unsupported_format(self, tmp_path):
        path = tmp_path / "results.json"
        df = pd.DataFrame({"logFC": [1.0]}, index=["CpG1"])
        with pytest.raises(RuntimeError, match="Unsupported format"):
            export_results(df, str(path), format="json")

    def test_save_processed_data_pickle_and_reload(self, tmp_path):
        M = pd.DataFrame([[1, 2], [3, 4]], columns=["S1", "S2"])
        pheno = pd.DataFrame(index=M.columns)
        data = ProcessedData(M=M, pheno=pheno, ann=None)
        path = tmp_path / "data.pkl"
        save_processed_data(data, path, format="pickle")
        loaded = pd.read_pickle(path)
        assert isinstance(loaded, ProcessedData)

    def test_save_processed_data_hdf5_and_reload(self, tmp_path):
        pytest.importorskip("tables")
        M = pd.DataFrame([[1, 2], [3, 4]], columns=["S1", "S2"])
        pheno = pd.DataFrame(index=M.columns)
        data = ProcessedData(M=M, pheno=pheno, ann=None)
        path = tmp_path / "data.h5"
        save_processed_data(data, path, format="hdf5")
        loaded = load_methylation_data(path)
        assert isinstance(loaded, ProcessedData)

    def test_export_idat_hdf5_fallback(self, tmp_path, monkeypatch):
        # force h5py to None to hit pandas.HDFStore path
        import dmeth.io.writers as writers

        monkeypatch.setattr(writers, "h5py", None)
        pytest.importorskip("tables")
        beta = pd.DataFrame([[0.1, 0.2]], index=["CpG1"], columns=["S1", "S2"])
        mvals = pd.DataFrame([[1.0, 2.0]], index=["CpG1"], columns=["S1", "S2"])
        sample_sheet = pd.DataFrame(index=["S1", "S2"])
        path = tmp_path / "idat.h5"
        result = export_idat_hdf5(beta, mvals, sample_sheet, path)
        assert result.exists()


class TestIOMopup:
    """Comprehensive mopup tests to cover all remaining IO branches"""

    def test_read_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _read("nonexistent.csv", index_col=None)

    def test_read_untrusted_pickle_raises(self, tmp_path):
        df = pd.DataFrame({"A": [1, 2, 3]})
        pkl = tmp_path / "bad.pkl"
        df.to_pickle(pkl)
        with pytest.raises(ValueError, match="Pickle files are not supported"):
            _read(pkl, index_col=None, trusted=False)

    def test_read_feather_index_out_of_bounds(self, tmp_path):
        pytest.importorskip("pyarrow")
        feather = tmp_path / "data.feather"
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df.to_feather(feather)
        with pytest.raises(IndexError):
            _read(feather, index_col=99)

    def test_read_hdf5_multiple_keys(self, tmp_path):
        pytest.importorskip("tables")
        h5 = tmp_path / "multi.h5"
        df1 = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"B": [4, 5, 6]})
        with pd.HDFStore(h5, "w") as store:
            store.put("M", df1)
            store.put("pheno", df2)
        with pytest.raises(ValueError, match="multiple keys"):
            _read(h5, index_col=None)

    def test_load_methylation_data_pheno_missing_sample(self):
        M = pd.DataFrame([[1, 2]], columns=["S1", "S2"], index=["CpG1"])
        pheno = pd.DataFrame(index=["S1"])  # missing S2
        with pytest.raises(KeyError, match="Pheno missing sample metadata"):
            load_methylation_data(M, pheno_input=pheno)

    def test_processed_data_empty_annotation_warns(self, caplog):
        M = pd.DataFrame([[1, 2]], columns=["S1", "S2"], index=["CpG1"])
        pheno = pd.DataFrame(index=M.columns)
        ann = pd.DataFrame()  # empty

        with caplog.at_level("WARNING"):
            ProcessedData(M=M, pheno=pheno, ann=ann)
        assert "Annotation provided but empty" in caplog.text

    def test_processed_data_ann_mismatch(self):
        M = pd.DataFrame([[1, 2]], columns=["S1", "S2"], index=["CpG1", "CpG2"])
        pheno = pd.DataFrame(index=M.columns)
        ann = pd.DataFrame(index=["CpG1"])  # missing CpG2
        with pytest.raises(KeyError, match="Probes in M but not in ann"):
            ProcessedData(M=M, pheno=pheno, ann=ann)

    def test_export_results_overwrite_protection(self, tmp_path):
        path = tmp_path / "results.csv"
        df = pd.DataFrame({"logFC": [1.0]}, index=["CpG1"])
        df.to_csv(path)
        with pytest.raises(FileExistsError):
            export_results(df, str(path), format="csv", overwrite=False)

    def test_export_results_unsupported_format(self, tmp_path):
        path = tmp_path / "results.json"
        df = pd.DataFrame({"logFC": [1.0]}, index=["CpG1"])
        with pytest.raises(RuntimeError, match="Unsupported format"):
            export_results(df, str(path), format="json")

    def test_export_results_excel_failure(self, tmp_path, monkeypatch):
        df = pd.DataFrame({"logFC": [1.0]}, index=["CpG1"])
        path = tmp_path / "results.xlsx"
        # force to_excel to raise ImportError
        monkeypatch.setattr(
            pd.DataFrame,
            "to_excel",
            lambda *a, **k: (_ for _ in ()).throw(ImportError("no engine")),
        )
        with pytest.raises(RuntimeError, match="Export failed"):
            export_results(df, str(path), format="excel")

    def test_save_processed_data_pickle_and_reload(self, tmp_path):
        M = pd.DataFrame([[1, 2]], columns=["S1", "S2"])
        pheno = pd.DataFrame(index=M.columns)
        data = ProcessedData(M=M, pheno=pheno, ann=None)
        path = tmp_path / "data.pkl"
        save_processed_data(data, path, format="pickle")
        loaded = pd.read_pickle(path)
        assert isinstance(loaded, ProcessedData)

    def test_save_processed_data_hdf5_no_ann(self, tmp_path):
        pytest.importorskip("tables")
        M = pd.DataFrame([[1, 2]], columns=["S1", "S2"])
        pheno = pd.DataFrame(index=M.columns)
        data = ProcessedData(M=M, pheno=pheno, ann=None)
        path = tmp_path / "data.h5"
        save_processed_data(data, path, format="hdf5")
        loaded = load_methylation_data(path)
        assert isinstance(loaded, ProcessedData)
        assert loaded.ann is not None  # ann created as empty DataFrame

    def test_export_idat_hdf5_optional_branches(self, tmp_path, monkeypatch):
        pytest.importorskip("tables")
        import dmeth.io.writers as writers

        monkeypatch.setattr(writers, "h5py", None)  # force pandas fallback
        beta = pd.DataFrame([[0.1, 0.2]], index=["CpG1"], columns=["S1", "S2"])
        # mvals=None, sample_sheet=None to hit optional branches
        path = tmp_path / "idat.h5"
        result = export_idat_hdf5(beta, mvals=None, sample_sheet=None, filepath=path)
        assert result.exists()

    # more
    def test_read_various_formats_and_hdf5_keys(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        # CSV
        csv_p = tmp_path / "t.csv"
        df.to_csv(csv_p)
        out = readers._read(csv_p, index_col=None)
        assert "a" in out.columns

        # XLSX
        xlsx_p = tmp_path / "t.xlsx"
        df.to_excel(xlsx_p, index=False)
        out = readers._read(xlsx_p, index_col=None)
        assert out.shape[0] == 2

        # Feather
        pytest.importorskip("pyarrow")
        feather_p = tmp_path / "t.feather"
        df.reset_index(drop=True).to_feather(feather_p)
        out = readers._read(feather_p, index_col=None)
        assert list(out.columns) == list(df.columns)

        # Parquet
        pytest.importorskip("pyarrow")
        pq_p = tmp_path / "t.parquet"
        df.to_parquet(pq_p)
        out = readers._read(pq_p, index_col=None)
        assert out.shape == df.shape

        # HDF5 with /M key
        pytest.importorskip("tables")
        h5_p = tmp_path / "t.h5"
        with pd.HDFStore(h5_p, mode="w") as store:
            store.put("/M", df)
        out = readers._read(h5_p, index_col=None)
        assert out.shape == df.shape

        # HDF5 with single key (no /M)
        h5_p2 = tmp_path / "t2.h5"
        with pd.HDFStore(h5_p2, mode="w") as store:
            store.put("only", df)
        out = readers._read(h5_p2, index_col=None)
        assert out.shape == df.shape

        # HDF5 with multiple keys -> ValueError
        h5_p3 = tmp_path / "t3.h5"
        with pd.HDFStore(h5_p3, mode="w") as store:
            store.put("a", df)
            store.put("b", df)
        with pytest.raises(ValueError):
            readers._read(h5_p3, index_col=None)

    def test_feather_index_col_behaviour(self, tmp_path):
        pytest.importorskip("pyarrow")
        df = pd.DataFrame({"id": ["x", "y"], "val": [1, 2]})
        p = tmp_path / "f.feather"
        df.to_feather(p)

        # index_col out of bounds should raise IndexError
        with pytest.raises(IndexError):
            readers._read(p, index_col=5)

        # index_col as string works
        out = readers._read(p, index_col="id")
        assert out.index.tolist() == df["id"].tolist()

    def test_read_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "file.unknown"
        p.write_text("hello")
        with pytest.raises(ValueError):
            readers._read(p, index_col=None)

    def test_load_methylation_data_missing_pheno_and_success(self, tmp_path):
        M = pd.DataFrame(
            np.random.rand(3, 2), index=["cg1", "cg2", "cg3"], columns=["s1", "s2"]
        )
        pheno = pd.DataFrame({"age": [30]}, index=["s1"])
        p_m = tmp_path / "M.csv"
        M.to_csv(p_m)
        p_pheno = tmp_path / "pheno.csv"
        pheno.to_csv(p_pheno)

        # missing sample should raise KeyError
        with pytest.raises(KeyError):
            readers.load_methylation_data(methylation_input=p_m, pheno_input=p_pheno)

        # proper pheno + ann should succeed
        pheno2 = pd.DataFrame({"age": [30, 40]}, index=["s1", "s2"])
        pheno2.to_csv(p_pheno)
        ann = pd.DataFrame({"probe_type": ["A", "B", "C"]}, index=["cg1", "cg2", "cg3"])
        p_ann = tmp_path / "ann.csv"
        ann.to_csv(p_ann)

        data = readers.load_methylation_data(
            methylation_input=p_m, pheno_input=p_pheno, ann_input=p_ann
        )
        assert list(data.M.columns) == ["s1", "s2"]
        assert data.ann is not None

    def test_export_results_csv_tsv_excel_and_badformat(self, tmp_path, monkeypatch):
        res = pd.DataFrame(
            {
                "logFC": [1.0, -1],
                "t": [2.0, 3.0],
                "pval": [0.01, 0.2],
                "padj": [0.02, 0.25],
                "meanM_A": [0.4, 0.6],
            },
            index=["CpG1", "CpG2"],
        )

        out_csv = tmp_path / "res.csv"
        writers.export_results(res, str(out_csv), format="csv")
        assert out_csv.exists()

        out_tsv = tmp_path / "res.tsv"
        writers.export_results(res, str(out_tsv), format="tsv")
        assert out_tsv.exists()

        # Force to_excel failure
        def fake_to_excel(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)
        out_xlsx = tmp_path / "res.xlsx"
        with pytest.raises(RuntimeError):
            writers.export_results(res, str(out_xlsx), format="xlsx")
        monkeypatch.undo()

        # unsupported format -> RuntimeError
        with pytest.raises(RuntimeError, match="Unsupported format"):
            writers.export_results(res, str(out_csv), format="badfmt")

    def test_save_processed_data_variants_and_suffix_handling(self, tmp_path):
        M = pd.DataFrame({"s1": [0.1, 0.2], "s2": [0.3, 0.4]}, index=["cg1", "cg2"])
        pheno = pd.DataFrame(index=["s1", "s2"])
        ann = pd.DataFrame({"x": [1, 2]}, index=["cg1", "cg2"])
        data = ProcessedData(M=M, pheno=pheno, ann=ann)

        # CSV
        out = tmp_path / "out.csv"
        writers.save_processed_data(data, out, format="csv")
        assert out.exists()
        assert (tmp_path / "out_pheno.csv").exists()

        # TSV
        out2 = tmp_path / "out2.tsv"
        writers.save_processed_data(data, out2, format="tsv")
        assert out2.exists()

        # Pickle
        outp = tmp_path / "out.pkl"
        writers.save_processed_data(data, outp, format="pkl")
        assert outp.exists()

        # HDF5
        pytest.importorskip("tables")
        outh = tmp_path / "out.h5"
        writers.save_processed_data(data, outh, format="hdf5")
        assert outh.exists()

        # suffix handling
        wrong = tmp_path / "wrong.txt"
        writers.save_processed_data(data, wrong, format="csv")
        assert wrong.with_suffix(".csv").exists()

    def test_export_idat_hdf5_fallback_and_report(self, tmp_path, monkeypatch):
        beta = pd.DataFrame(
            np.random.rand(4, 2),
            index=[f"cg{i}" for i in range(4)],
            columns=["s1", "s2"],
        )
        sample_sheet = pd.DataFrame({"age": [10, 20]}, index=["s1", "s2"])
        fp = tmp_path / "idat.h5"

        # Force writers.h5py to None to test pandas.HDFStore fallback
        monkeypatch.setattr(writers, "h5py", None)
        pytest.importorskip("tables")
        outp = writers.export_idat_hdf5(
            beta,
            mvals=None,
            sample_sheet=sample_sheet,
            filepath=fp,
            compress=False
        )
        assert outp.exists()