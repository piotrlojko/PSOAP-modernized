"""
Tests for barycentric velocity correction utilities in psoap.data.
"""

import numpy as np
import pytest

from psoap.data import compute_barycentric_corrections, Chunk, redshift


# A pair of BJD_TDB dates close to J2000 used throughout the tests
BJD_DATES = np.array([2451545.0, 2451600.0])

# Approximate coordinates of Tau Ceti (ICRS, decimal degrees)
TAU_CETI_RA = 26.021
TAU_CETI_DEC = -15.939


def test_compute_barycentric_corrections_returns_array():
    v = compute_barycentric_corrections(BJD_DATES, TAU_CETI_RA, TAU_CETI_DEC)
    assert isinstance(v, np.ndarray)
    assert v.shape == (len(BJD_DATES),)


def test_compute_barycentric_corrections_scalar_date():
    """Single date should still return a 1-element array."""
    v = compute_barycentric_corrections(BJD_DATES[0], TAU_CETI_RA, TAU_CETI_DEC)
    assert v.shape == (1,)


def test_compute_barycentric_corrections_range():
    """Earth's barycentric speed projected along any LOS is < 30 km/s."""
    v = compute_barycentric_corrections(BJD_DATES, TAU_CETI_RA, TAU_CETI_DEC)
    assert np.all(np.abs(v) < 35.0)


def test_compute_barycentric_corrections_different_epochs_differ():
    """Corrections at different epochs should generally not be identical."""
    v = compute_barycentric_corrections(BJD_DATES, TAU_CETI_RA, TAU_CETI_DEC)
    assert v[0] != v[1]


def _make_chunk(wl0=5000.0, n_pix=50, n_epochs=2):
    """Create a minimal Chunk with flat flux for testing."""
    wl = np.tile(np.linspace(wl0, wl0 + 10, n_pix), (n_epochs, 1))
    fl = np.ones((n_epochs, n_pix))
    sigma = np.full((n_epochs, n_pix), 0.01)
    dates = np.tile(BJD_DATES[:n_epochs, np.newaxis], (1, n_pix))
    return Chunk(wl, fl, sigma, dates)


def test_apply_barycentric_correction_shifts_wl():
    chunk = _make_chunk()
    wl_before = chunk.wl.copy()
    v_bary = np.array([10.0, -5.0])
    chunk.apply_barycentric_correction(v_bary)

    expected_wl0 = redshift(wl_before[0], 10.0)
    expected_wl1 = redshift(wl_before[1], -5.0)
    np.testing.assert_allclose(chunk.wl[0], expected_wl0, rtol=1e-12)
    np.testing.assert_allclose(chunk.wl[1], expected_wl1, rtol=1e-12)


def test_apply_barycentric_correction_updates_lwl():
    chunk = _make_chunk()
    v_bary = np.array([10.0, -5.0])
    chunk.apply_barycentric_correction(v_bary)
    np.testing.assert_allclose(chunk.lwl, np.log(chunk.wl), rtol=1e-12)


def test_apply_barycentric_correction_zero_velocity_no_change():
    chunk = _make_chunk()
    wl_before = chunk.wl.copy()
    chunk.apply_barycentric_correction(np.zeros(2))
    np.testing.assert_allclose(chunk.wl, wl_before, rtol=1e-12)


def test_apply_barycentric_correction_does_not_change_flux():
    chunk = _make_chunk()
    fl_before = chunk.fl.copy()
    chunk.apply_barycentric_correction(np.array([10.0, -5.0]))
    np.testing.assert_array_equal(chunk.fl, fl_before)
