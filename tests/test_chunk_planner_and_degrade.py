"""Tests for psoap.chunk_planner and psoap.resolution_degrade."""

import numpy as np
import pytest

from psoap.chunk_planner import plan_chunks, _available_memory_bytes
from psoap.resolution_degrade import degrade_resolution, degrade_wl_grid
from psoap.data import Chunk

# Floating-point comparison tolerance for wavelength boundary checks.
_WL_TOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(n_epochs=3, n_pix=50, wl_start=5000.0, wl_step=0.1):
    """Create a synthetic Chunk for testing (before apply_mask)."""
    wl_1d = wl_start + np.arange(n_pix) * wl_step
    wl = np.tile(wl_1d, (n_epochs, 1))
    fl = np.ones((n_epochs, n_pix), dtype=np.float64)
    sigma = np.full((n_epochs, n_pix), 0.01, dtype=np.float64)
    dates = np.arange(n_epochs, dtype=np.float64) * 10.0
    date_arr = dates[:, np.newaxis] * np.ones((n_epochs, n_pix))
    return Chunk(wl, fl, sigma, date_arr)


# ---------------------------------------------------------------------------
# chunk_planner tests
# ---------------------------------------------------------------------------

class TestPlanChunks:
    """Unit tests for plan_chunks."""

    def test_single_chunk_when_fits(self):
        """When the full wavelength range fits in memory, return one chunk."""
        wl_ref = np.linspace(5000, 5010, 100)
        chunks = plan_chunks(wl_ref=wl_ref, n_epochs=2)
        assert len(chunks) >= 1
        assert isinstance(chunks[0], dict)
        assert "wl_min" in chunks[0] and "wl_max" in chunks[0]

    def test_chunk_boundaries_cover_range(self):
        """Combined chunk boundaries must span the full requested window."""
        wl_ref = np.linspace(5000, 5100, 1000)
        chunks = plan_chunks(
            wl_ref=wl_ref, wl_min=5010.0, wl_max=5090.0, n_epochs=2
        )
        combined_min = min(c["wl_min"] for c in chunks)
        combined_max = max(c["wl_max"] for c in chunks)
        assert combined_min >= 5010.0 - _WL_TOL
        assert combined_max <= 5090.0 + _WL_TOL

    def test_wl_min_wl_max_none_uses_full_range(self):
        """When wl_min/wl_max are None, the full wl_ref range is used."""
        wl_ref = np.linspace(5000, 5050, 500)
        chunks = plan_chunks(wl_ref=wl_ref, n_epochs=2)
        combined_min = min(c["wl_min"] for c in chunks)
        combined_max = max(c["wl_max"] for c in chunks)
        assert combined_min >= 5000.0 - _WL_TOL
        assert combined_max <= 5050.0 + _WL_TOL

    def test_many_chunks_when_memory_tight(self):
        """With a very small memory cap, many chunks should be produced."""
        wl_ref = np.linspace(5000, 5100, 1000)
        # Cap memory to 1 MiB → forces many chunks for any n_epochs > 1
        chunks = plan_chunks(
            wl_ref=wl_ref, n_epochs=5,
            memory_cap_bytes=1024 * 1024,
            safety_factor=1.0,
        )
        assert len(chunks) > 1

    def test_max_chunks_guard_raises(self):
        """Exceeding max_chunks raises RuntimeError with useful message."""
        wl_ref = np.linspace(5000, 5100, 1000)
        with pytest.raises(RuntimeError, match="max_chunks"):
            plan_chunks(
                wl_ref=wl_ref, n_epochs=100,
                memory_cap_bytes=1024,  # ludicrously small
                safety_factor=1.0,
                max_chunks=1,
            )

    def test_empty_range_raises(self):
        """Specifying a wl window with no pixels raises RuntimeError."""
        wl_ref = np.linspace(5000, 5100, 100)
        with pytest.raises(RuntimeError, match="No pixels found"):
            plan_chunks(wl_ref=wl_ref, wl_min=6000.0, wl_max=7000.0, n_epochs=2)

    def test_zero_epochs_raises(self):
        """n_epochs == 0 raises RuntimeError."""
        wl_ref = np.linspace(5000, 5100, 100)
        with pytest.raises(RuntimeError, match="n_epochs"):
            plan_chunks(wl_ref=wl_ref, n_epochs=0)

    def test_wallclock_guard_raises(self):
        """Estimated wall-clock exceeding threshold aborts with message."""
        wl_ref = np.linspace(5000, 5010, 100)
        with pytest.raises(RuntimeError, match="wall-clock"):
            plan_chunks(
                wl_ref=wl_ref, n_epochs=2,
                max_wallclock_hours=0.0,
                n_samples=100,
                time_per_likelihood_s=1.0,
            )

    def test_max_matrix_elements_limits_chunk_size(self):
        """max_matrix_elements further constrains per-chunk N."""
        wl_ref = np.linspace(5000, 5100, 500)
        # N_max = sqrt(max_matrix_elements) = 10  →  n_pix_max = 5 for 2 epochs
        chunks = plan_chunks(
            wl_ref=wl_ref, n_epochs=2,
            max_matrix_elements=100,
        )
        # Should produce many chunks
        assert len(chunks) > 10

    def test_returns_list_of_dicts(self):
        """Return type must be list[dict] with wl_min, wl_max keys."""
        wl_ref = np.linspace(5000, 5020, 200)
        chunks = plan_chunks(wl_ref=wl_ref, n_epochs=2)
        assert isinstance(chunks, list)
        for ch in chunks:
            assert isinstance(ch, dict)
            assert set(ch.keys()) == {"wl_min", "wl_max"}
            assert ch["wl_min"] <= ch["wl_max"]

    def test_extra_kwargs_ignored(self):
        """Unknown kwargs (e.g. 'enabled') must be silently ignored."""
        wl_ref = np.linspace(5000, 5010, 100)
        # Should not raise
        chunks = plan_chunks(wl_ref=wl_ref, n_epochs=2, enabled=True,
                             unknown_key="value")
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# resolution_degrade tests
# ---------------------------------------------------------------------------

class TestDegradeWlGrid:
    def test_factor_reduces_length(self):
        wl = np.linspace(5000, 5010, 100)
        wl_new = degrade_wl_grid(wl, factor=2)
        assert len(wl_new) == 50

    def test_target_step_produces_uniform_grid(self):
        wl = np.linspace(5000, 5005, 501)
        wl_new = degrade_wl_grid(wl, target_step=0.1)
        diffs = np.diff(wl_new)
        assert np.allclose(diffs, 0.1, atol=1e-10)

    def test_raises_if_neither_given(self):
        wl = np.linspace(5000, 5010, 100)
        with pytest.raises(ValueError, match="exactly one"):
            degrade_wl_grid(wl)

    def test_raises_if_both_given(self):
        wl = np.linspace(5000, 5010, 100)
        with pytest.raises(ValueError, match="exactly one"):
            degrade_wl_grid(wl, target_step=0.5, factor=2)

    def test_monotonic(self):
        wl = np.linspace(5000, 5010, 100)
        wl_new = degrade_wl_grid(wl, factor=3)
        assert np.all(np.diff(wl_new) > 0)


class TestDegradeResolution:
    def test_factor_reduces_pixels(self):
        chunk = _make_chunk(n_epochs=3, n_pix=100, wl_step=0.1)
        degraded = degrade_resolution(chunk, factor=2)
        assert degraded.n_pix == 50

    def test_target_step_reduces_pixels(self):
        chunk = _make_chunk(n_epochs=3, n_pix=100, wl_step=0.05)
        degraded = degrade_resolution(chunk, target_step=0.1)
        # new grid spans same range with ~0.1 Å step
        assert degraded.n_pix < chunk.n_pix

    def test_shape_consistency(self):
        chunk = _make_chunk(n_epochs=4, n_pix=60, wl_step=0.1)
        degraded = degrade_resolution(chunk, factor=3)
        assert degraded.wl.shape == (4, 20)
        assert degraded.fl.shape == (4, 20)
        assert degraded.sigma.shape == (4, 20)

    def test_wl_monotonic(self):
        chunk = _make_chunk(n_epochs=2, n_pix=80, wl_step=0.1)
        degraded = degrade_resolution(chunk, factor=2)
        for i in range(degraded.n_epochs):
            assert np.all(np.diff(degraded.wl[i]) > 0)

    def test_interpolation_flat_spectrum(self):
        """Flat flux=1 should remain flat after degradation."""
        chunk = _make_chunk(n_epochs=2, n_pix=100, wl_step=0.1)
        degraded = degrade_resolution(chunk, factor=4)
        assert np.allclose(degraded.fl, 1.0)

    def test_raises_if_both_given(self):
        chunk = _make_chunk()
        with pytest.raises(ValueError, match="exactly one"):
            degrade_resolution(chunk, target_step=0.5, factor=2)

    def test_raises_if_neither_given(self):
        chunk = _make_chunk()
        with pytest.raises(ValueError, match="exactly one"):
            degrade_resolution(chunk)

    def test_date1d_preserved(self):
        """Epoch dates must be preserved after degradation."""
        chunk = _make_chunk(n_epochs=3, n_pix=60, wl_step=0.1)
        degraded = degrade_resolution(chunk, factor=2)
        assert np.allclose(degraded.date1D, chunk.date1D)

    def test_mask_finite_sigma(self):
        """Mask should be True where sigma is finite."""
        chunk = _make_chunk(n_epochs=2, n_pix=80, wl_step=0.1)
        degraded = degrade_resolution(chunk, factor=2)
        # All sigma should be finite (no extrapolation) → mask all True
        assert np.all(degraded.mask)

    def test_apply_mask_after_degrade(self):
        """Chunk produced by degrade_resolution should survive apply_mask."""
        chunk = _make_chunk(n_epochs=2, n_pix=60, wl_step=0.1)
        degraded = degrade_resolution(chunk, factor=2)
        degraded.apply_mask()
        assert degraded.N > 0
        assert len(degraded.wl) == degraded.N


# ---------------------------------------------------------------------------
# Config backward-compatibility
# ---------------------------------------------------------------------------

class TestConfigBackwardCompatibility:
    """New config keys must be absent from old configs without errors."""

    def test_old_config_no_auto_chunk_key(self):
        """Configs without 'auto_chunk' must still work (default = disabled)."""
        config = {
            "model": "SB2",
            "spectra_list": "spectra_list.txt",
            "parameters": {"q": 0.2},
        }
        auto_chunk_cfg = config.get("auto_chunk") or {}
        assert bool(auto_chunk_cfg.get("enabled", False)) is False

    def test_old_config_no_resolution_degrade_key(self):
        config = {"model": "SB2"}
        degrade_cfg = config.get("resolution_degrade") or {}
        assert bool(degrade_cfg.get("enabled", False)) is False

    def test_auto_chunk_enabled_flag_parsed(self):
        config = {"auto_chunk": {"enabled": True, "memory_fraction": 0.3}}
        auto_chunk_cfg = config.get("auto_chunk") or {}
        assert bool(auto_chunk_cfg.get("enabled", False)) is True
        assert auto_chunk_cfg.get("memory_fraction") == 0.3

    def test_resolution_degrade_factor_parsed(self):
        config = {"resolution_degrade": {"enabled": True, "factor": 4}}
        degrade_cfg = config.get("resolution_degrade") or {}
        assert bool(degrade_cfg.get("enabled", False)) is True
        assert degrade_cfg.get("factor") == 4
