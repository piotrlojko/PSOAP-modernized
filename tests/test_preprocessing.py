"""Tests for shared preprocessing pipeline order and chunking behavior."""

import numpy as np
import pytest

from psoap import preprocessing
from psoap.data import Chunk


def _make_chunk(n_epochs=2, n_pix=200, wl_start=5000.0, wl_step=0.1):
    wl_1d = wl_start + np.arange(n_pix) * wl_step
    wl = np.tile(wl_1d, (n_epochs, 1))
    fl = np.ones((n_epochs, n_pix), dtype=np.float64)
    sigma_arr = np.full((n_epochs, n_pix), 0.01, dtype=np.float64)
    dates = np.arange(n_epochs, dtype=np.float64) + 2451545.0
    date_arr = dates[:, np.newaxis] * np.ones((n_epochs, n_pix))
    return Chunk(wl, fl, sigma_arr, date_arr)


def test_pipeline_order_barycentric_then_crop_then_degrade_then_chunk(monkeypatch):
    call_order = []
    state = {"chunk": _make_chunk(n_epochs=3, n_pix=240)}

    def fake_parse_spectra_list(_path):
        return ["a.txt", "b.txt", "c.txt"], np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def fake_from_textfiles(*_args, **kwargs):
        call_order.append("load_full")
        assert kwargs.get("wl_min") is None
        assert kwargs.get("wl_max") is None
        return state["chunk"]

    def fake_compute_barycentric(_dates, _ra, _dec):
        call_order.append("compute_bary")
        return np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def fake_apply_bary(self, _v_bary):
        call_order.append("apply_bary")
        return None

    orig_crop = Chunk.crop_wavelength_range

    def fake_crop(self, wl_min=None, wl_max=None):
        if wl_min == 5005.0 and wl_max == 5015.0:
            call_order.append("crop_window")
        else:
            call_order.append("crop_chunk")
        return orig_crop(self, wl_min=wl_min, wl_max=wl_max)

    def fake_degrade(chunk, **_kwargs):
        call_order.append("degrade")
        return chunk

    def fake_plan_chunks(wl_ref, **_kwargs):
        call_order.append("plan_chunks")
        assert np.isclose(wl_ref[0], 5005.0)
        assert np.isclose(wl_ref[-1], 5015.0)
        mid = wl_ref[len(wl_ref) // 2]
        return [
            {"wl_min": float(wl_ref[0]), "wl_max": float(mid)},
            {"wl_min": float(mid), "wl_max": float(wl_ref[-1])},
        ]

    monkeypatch.setattr(preprocessing, "parse_spectra_list", fake_parse_spectra_list)
    monkeypatch.setattr(Chunk, "from_textfiles", classmethod(lambda cls, *a, **k: fake_from_textfiles(*a, **k)))
    monkeypatch.setattr(preprocessing, "compute_barycentric_corrections", fake_compute_barycentric)
    monkeypatch.setattr(Chunk, "apply_barycentric_correction", fake_apply_bary)
    monkeypatch.setattr(Chunk, "crop_wavelength_range", fake_crop)
    monkeypatch.setattr(preprocessing, "degrade_resolution", fake_degrade)
    monkeypatch.setattr(preprocessing, "plan_chunks", fake_plan_chunks)

    cfg = {
        "spectra_list": "spectra_list.txt",
        "barycentric_corrected": False,
        "target_ra": 26.0,
        "target_dec": -15.0,
        "wl_min": 5005.0,
        "wl_max": 5015.0,
        "resolution_degrade": {"enabled": True, "factor": 2},
        "samples": 100,
    }
    chunks = preprocessing.build_preprocessed_chunks(cfg)
    assert len(chunks) == 2
    assert call_order == [
        "load_full",
        "compute_bary",
        "apply_bary",
        "crop_window",
        "degrade",
        "plan_chunks",
        "crop_chunk",
        "crop_chunk",
    ]


def test_pipeline_autochunks_even_when_wavelength_window_is_set(monkeypatch):
    chunk = _make_chunk(n_epochs=2, n_pix=100)

    monkeypatch.setattr(
        preprocessing,
        "parse_spectra_list",
        lambda _path: (["a.txt", "b.txt"], np.array([1.0, 2.0], dtype=np.float64)),
    )
    monkeypatch.setattr(
        Chunk,
        "from_textfiles",
        classmethod(lambda cls, *a, **k: chunk),
    )

    called = {"plan": 0}

    def fake_plan_chunks(wl_ref, **_kwargs):
        called["plan"] += 1
        assert wl_ref[0] >= 5005.0
        assert wl_ref[-1] <= 5008.0
        return [{"wl_min": float(wl_ref[0]), "wl_max": float(wl_ref[-1])}]

    monkeypatch.setattr(preprocessing, "plan_chunks", fake_plan_chunks)

    cfg = {
        "spectra_list": "spectra_list.txt",
        "wl_min": 5005.0,
        "wl_max": 5008.0,
        "barycentric_corrected": True,
        "samples": 10,
    }
    out = preprocessing.build_preprocessed_chunks(cfg)
    assert called["plan"] == 1
    assert len(out) == 1
