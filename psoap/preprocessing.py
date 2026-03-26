"""Shared spectral preprocessing pipeline for PSOAP samplers.

Pipeline order:
1) optional barycentric correction,
2) optional wavelength-window cropping,
3) optional resolution degradation,
4) mandatory auto-chunking.
"""

import numpy as np

from psoap.data import Chunk, compute_barycentric_corrections
from psoap.input_parsing import parse_spectra_list
from psoap.chunk_planner import plan_chunks
from psoap.resolution_degrade import degrade_resolution


def _auto_chunk_config(config):
    """Return auto-chunk planner kwargs with defaults applied."""
    auto_chunk_cfg = dict(config.get("auto_chunk") or {})
    # Auto-chunking is mandatory in the preprocessing pipeline.
    auto_chunk_cfg.pop("enabled", None)
    return auto_chunk_cfg


def build_preprocessed_chunks(config):
    """Build preprocessed chunks in the canonical order required by PSOAP."""
    filenames, dates = parse_spectra_list(config["spectra_list"])

    epoch_limit = config.get("epoch_limit")
    if epoch_limit is not None:
        filenames = list(filenames)[:epoch_limit]
        dates = np.asarray(dates)[:epoch_limit]
    n_epochs = len(filenames)

    # Load full spectra first so barycentric correction is applied before any
    # wavelength-window truncation.
    data = Chunk.from_textfiles(
        filenames,
        dates,
        limit=None,
        wl_min=None,
        wl_max=None,
    )

    # 1) Barycentric correction (optional)
    if not config.get("barycentric_corrected", True):
        ra = config["target_ra"]
        dec = config["target_dec"]
        v_bary = compute_barycentric_corrections(data.date1D, ra, dec)
        print("Applying barycentric corrections (km/s):", v_bary)
        data.apply_barycentric_correction(v_bary)

    # 2) Wavelength window cut (optional)
    data = data.crop_wavelength_range(
        wl_min=config.get("wl_min"),
        wl_max=config.get("wl_max"),
    )

    # 3) Optional resolution degradation
    degrade_cfg = config.get("resolution_degrade") or {}
    do_degrade = bool(degrade_cfg.get("enabled", False))
    if do_degrade:
        data = degrade_resolution(
            data,
            target_step=degrade_cfg.get("target_step"),
            factor=degrade_cfg.get("factor"),
        )

    # Validate epoch alignment and monotonic wavelength grids prior to planning.
    if data.n_epochs <= 0:
        raise RuntimeError("No epochs available after preprocessing.")
    if not np.allclose(data.date1D, data.date[:, 0]):
        raise RuntimeError("Inconsistent epoch-date bookkeeping in preprocessed data.")
    for i in range(data.n_epochs):
        if not np.all(np.diff(data.wl[i]) > 0):
            raise RuntimeError("Wavelength grid must be strictly increasing per epoch.")

    # 4) Mandatory auto-chunking
    auto_chunk_cfg = _auto_chunk_config(config)
    wl_ranges = plan_chunks(
        wl_ref=data.wl[0],
        wl_min=None,
        wl_max=None,
        n_epochs=n_epochs,
        n_samples=config.get("samples", 1000),
        **auto_chunk_cfg,
    )

    chunks = [
        data.crop_wavelength_range(wl_min=wr["wl_min"], wl_max=wr["wl_max"])
        for wr in wl_ranges
    ]
    return chunks
