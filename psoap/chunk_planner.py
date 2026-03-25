"""Memory-aware wavelength chunk planner for PSOAP parallel sampling.

Given spectral metadata (wavelength reference grid, number of epochs) and a
memory budget derived from system resources, the planner splits the wavelength
range into sub-chunks such that the dense ``N × N`` covariance matrix needed
by each worker fits comfortably in RAM.

Typical usage::

    from psoap.chunk_planner import plan_chunks

    wl_ranges = plan_chunks(
        wl_ref=wl_first_spectrum,
        wl_min=config.get("wl_min"),
        wl_max=config.get("wl_max"),
        n_epochs=n_epochs,
        **config.get("auto_chunk", {}),
    )
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# dtype used for the covariance matrix (float64 → 8 bytes/element)
_DTYPE_BYTES = 8

# Conservative estimate of the number of simultaneous V11-sized arrays needed
# (e.g. V11 itself + temporaries inside the Cholesky solve + gradient buffer).
_DEFAULT_SAFETY_FACTOR = 4.0

# Fraction of available RAM to budget per worker by default.
_DEFAULT_MEMORY_FRACTION = 0.25

# Fallback available-memory value used when psutil is not installed.
_FALLBACK_MEMORY_BYTES = 2 * 1024 ** 3  # 2 GiB


def _available_memory_bytes():
    """Return available system memory in bytes, falling back to 2 GiB."""
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        logger.warning(
            "psutil is not installed; assuming 2 GiB available RAM for chunk "
            "planning.  Install psutil for accurate memory detection."
        )
        return _FALLBACK_MEMORY_BYTES


def plan_chunks(
    wl_ref,
    wl_min=None,
    wl_max=None,
    n_epochs=1,
    enabled=True,
    memory_fraction=_DEFAULT_MEMORY_FRACTION,
    safety_factor=_DEFAULT_SAFETY_FACTOR,
    max_chunks=None,
    max_matrix_elements=None,
    memory_cap_bytes=None,
    max_wallclock_hours=48.0,
    n_samples=None,
    time_per_likelihood_s=None,
    **_extra,
):
    """Plan chunk wavelength ranges that fit within a memory budget.

    Parameters
    ----------
    wl_ref : array-like
        Reference wavelength grid (sorted, 1-D) in Angstroms, typically taken
        from the first epoch of the first spectrum file.
    wl_min, wl_max : float or None
        Wavelength window limits.  ``None`` means use the extremes of
        *wl_ref*.
    n_epochs : int
        Number of epochs (spectra) that will be stacked per chunk.
    enabled : bool
        Passed through by callers that read from config; ignored here (callers
        should only invoke :func:`plan_chunks` when ``enabled`` is ``True``).
    memory_fraction : float
        Fraction of available system RAM to allocate per worker (default
        0.25).
    safety_factor : float
        Multiplier on the raw ``N²`` matrix-size estimate (accounts for
        temporary copies, solver buffers, etc.).
    max_chunks : int or None
        Hard upper bound on the number of chunks that may be returned.  If
        the planner requires more chunks than this limit, a :class:`RuntimeError`
        is raised with guidance on how to reduce data volume.
    max_matrix_elements : int or None
        Absolute cap on ``N²`` (number of matrix elements per chunk).
        Overrides the memory-derived limit when smaller.
    memory_cap_bytes : int or None
        Absolute memory cap per worker in bytes.  Overrides *memory_fraction*
        when given.
    max_wallclock_hours : float
        Abort threshold for estimated total wall-clock time.  Only checked
        when both *n_samples* and *time_per_likelihood_s* are provided.
    n_samples : int or None
        Number of MCMC samples (used for optional wall-clock estimate).
    time_per_likelihood_s : float or None
        Estimated seconds per likelihood evaluation per worker.  When
        ``None``, the wall-clock check is skipped.

    Returns
    -------
    list of dict
        Each element is ``{"wl_min": float, "wl_max": float}``.

    Raises
    ------
    RuntimeError
        If the plan is infeasible (no valid chunk size exists) or if a
        runtime / chunk-count guard is triggered.
    """
    wl_ref = np.asarray(wl_ref, dtype=np.float64)

    # Apply wavelength window
    lo = wl_min if wl_min is not None else float(wl_ref[0])
    hi = wl_max if wl_max is not None else float(wl_ref[-1])
    mask = (wl_ref >= lo) & (wl_ref <= hi)
    wl_window = wl_ref[mask]
    total_pix = len(wl_window)

    if total_pix == 0:
        raise RuntimeError(
            "No pixels found in the wavelength range [{:.2f}, {:.2f}] Å.  "
            "Check wl_min/wl_max in config.".format(lo, hi)
        )

    if n_epochs <= 0:
        raise RuntimeError("n_epochs must be > 0.")

    # --- Memory budget ---
    avail = _available_memory_bytes()
    budget = (
        min(memory_cap_bytes, avail * memory_fraction)
        if memory_cap_bytes is not None
        else avail * memory_fraction
    )

    # N_max such that N² × bytes_per_element × safety_factor ≤ budget
    N_max_mem = int(np.sqrt(budget / (_DTYPE_BYTES * safety_factor)))
    if max_matrix_elements is not None:
        N_max = min(N_max_mem, int(np.sqrt(float(max_matrix_elements))))
    else:
        N_max = N_max_mem

    # N = n_epochs × n_pix_per_chunk  →  derive n_pix_max per chunk
    n_pix_max = max(1, N_max // n_epochs)

    # Minimum number of chunks required
    n_chunks_needed = max(1, int(np.ceil(total_pix / n_pix_max)))

    logger.info(
        "Chunk planner: total_pix=%d, n_epochs=%d, N_max=%d, "
        "n_pix_max_per_chunk=%d, n_chunks_needed=%d",
        total_pix,
        n_epochs,
        N_max,
        n_pix_max,
        n_chunks_needed,
    )
    logger.info(
        "Chunk planner: available_mem=%.2f GiB, budget=%.2f GiB, "
        "safety_factor=%.1f",
        avail / 1024 ** 3,
        budget / 1024 ** 3,
        safety_factor,
    )

    # Infeasibility: even a single-pixel chunk is too large
    mem_single_pix = (n_epochs ** 2) * _DTYPE_BYTES * safety_factor
    if mem_single_pix > avail:
        raise RuntimeError(
            "Infeasible: even a single-pixel chunk requires {:.2f} GiB "
            "for {:d} epochs, but only {:.2f} GiB is available.  "
            "Reduce epoch_limit or enable resolution_degrade in config.".format(
                mem_single_pix / 1024 ** 3, n_epochs, avail / 1024 ** 3
            )
        )

    # max_chunks guard
    if max_chunks is not None and n_chunks_needed > max_chunks:
        raise RuntimeError(
            "Auto-chunker requires {:d} chunks to fit within the memory budget, "
            "but max_chunks={:d} is set.  "
            "Enable or tune 'resolution_degrade' in config to reduce data "
            "volume, or increase max_chunks / memory_fraction.  "
            "Current estimates: n_epochs={:d}, n_pix_max={:d}/chunk, "
            "available_mem={:.2f} GiB.".format(
                n_chunks_needed,
                max_chunks,
                n_epochs,
                n_pix_max,
                avail / 1024 ** 3,
            )
        )

    # Wall-clock guard
    if time_per_likelihood_s is not None and n_samples is not None:
        est_wall_s = n_samples * time_per_likelihood_s
        est_wall_h = est_wall_s / 3600.0
        logger.info(
            "Chunk planner: estimated wall-clock %.1f h "
            "(%.0f samples × %.2f s/eval)",
            est_wall_h,
            n_samples,
            time_per_likelihood_s,
        )
        if est_wall_h > max_wallclock_hours:
            raise RuntimeError(
                "Estimated wall-clock time ({:.1f} h) exceeds threshold "
                "({:.1f} h).  Enable 'resolution_degrade' in config to reduce "
                "computation, or reduce 'samples' or increase "
                "'max_wallclock_hours'.".format(est_wall_h, max_wallclock_hours)
            )

    # --- Build chunk boundaries ---
    chunk_edges = np.array_split(wl_window, n_chunks_needed)
    chunks = [
        {"wl_min": float(sub[0]), "wl_max": float(sub[-1])}
        for sub in chunk_edges
        if len(sub) > 0
    ]

    logger.info(
        "Chunk planner: created %d chunk(s) spanning [%.2f, %.2f] Å.",
        len(chunks),
        chunks[0]["wl_min"],
        chunks[-1]["wl_max"],
    )
    for i, ch in enumerate(chunks):
        logger.debug("  chunk %d: [%.4f, %.4f] Å", i, ch["wl_min"], ch["wl_max"])

    print(
        "[chunk_planner] Planning {:d} wavelength chunk(s) across "
        "[{:.2f}, {:.2f}] Å  "
        "(N_max={:d}, n_pix_max={:d}/chunk, n_epochs={:d}, "
        "budget={:.2f} GiB).".format(
            len(chunks),
            lo,
            hi,
            N_max,
            n_pix_max,
            n_epochs,
            budget / 1024 ** 3,
        )
    )

    return chunks
