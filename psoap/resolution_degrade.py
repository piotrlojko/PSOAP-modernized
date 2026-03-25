"""Spectral resolution degradation utility for PSOAP.

Provides :func:`degrade_resolution`, which performs linear interpolation of
flux and sigma onto a sparser wavelength grid as a pre-processing step before
GP sampling.  Reducing the number of spectral pixels ``N`` decreases both the
memory footprint (``O(N²)``) and the compute cost (``O(N³)``) of the dense
covariance operations.

Typical usage::

    from psoap.resolution_degrade import degrade_resolution

    chunkSpec = Chunk.from_textfiles(...)
    chunkSpec = degrade_resolution(chunkSpec, factor=2)
    chunkSpec.apply_mask()
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def degrade_wl_grid(wl_ref, target_step=None, factor=None):
    """Compute a sparser wavelength grid without constructing a full Chunk.

    Exactly one of *target_step* or *factor* must be provided.

    Parameters
    ----------
    wl_ref : array-like
        Reference (sorted) wavelength grid in Angstroms.
    target_step : float or None
        Desired uniform spacing in Angstroms.
    factor : int or None
        Integer down-sampling factor (keep every *factor*-th pixel).

    Returns
    -------
    np.ndarray
        The degraded wavelength grid.
    """
    wl_ref = np.asarray(wl_ref, dtype=np.float64)
    if (target_step is None) == (factor is None):
        raise ValueError("Provide exactly one of 'target_step' or 'factor'.")
    if target_step is not None:
        wl_new = np.arange(
            wl_ref[0], wl_ref[-1] + target_step / 2.0, float(target_step)
        )
    else:
        wl_new = wl_ref[:: int(factor)]
    return wl_new


def degrade_resolution(chunk, target_step=None, factor=None):
    """Return a new :class:`~psoap.data.Chunk` with reduced spectral resolution.

    Interpolates each epoch's ``fl`` and ``sigma`` onto a sparser wavelength
    grid via piecewise-linear interpolation.  Must be called **before**
    :meth:`~psoap.data.Chunk.apply_mask`.

    Exactly one of *target_step* or *factor* must be provided.

    Parameters
    ----------
    chunk : psoap.data.Chunk
        Input chunk object (before ``apply_mask``).
    target_step : float or None
        Desired uniform wavelength spacing in Angstroms.  A new grid is
        constructed with this step size spanning the same wavelength range as
        the original chunk.
    factor : int or None
        Integer down-sampling factor: keep every *factor*-th pixel.

    Returns
    -------
    psoap.data.Chunk
        New chunk interpolated onto the sparser grid, with an updated mask
        (pixels outside the original wavelength coverage are masked out).

    Raises
    ------
    ValueError
        If neither or both of *target_step* and *factor* are given.
    """
    from scipy.interpolate import interp1d

    from psoap.data import Chunk

    if (target_step is None) == (factor is None):
        raise ValueError("Provide exactly one of 'target_step' or 'factor'.")

    # Reference wavelength grid from epoch 0
    wl_ref = chunk.wl[0]
    wl_new = degrade_wl_grid(wl_ref, target_step=target_step, factor=factor)

    n_epochs = chunk.n_epochs
    n_pix_orig = chunk.n_pix
    n_pix_new = len(wl_new)

    wl_arr = np.empty((n_epochs, n_pix_new), dtype=np.float64)
    fl_arr = np.empty((n_epochs, n_pix_new), dtype=np.float64)
    sigma_arr = np.empty((n_epochs, n_pix_new), dtype=np.float64)

    for i in range(n_epochs):
        f_fl = interp1d(
            chunk.wl[i],
            chunk.fl[i],
            kind="linear",
            bounds_error=False,
            fill_value=1.0,
        )
        f_sig = interp1d(
            chunk.wl[i],
            chunk.sigma[i],
            kind="linear",
            bounds_error=False,
            fill_value=np.inf,
        )
        wl_arr[i] = wl_new
        fl_arr[i] = f_fl(wl_new)
        sigma_arr[i] = f_sig(wl_new)

    # Broadcast dates to (n_epochs, n_pix_new)
    date_arr = chunk.date1D[:, np.newaxis] * np.ones((n_epochs, n_pix_new))

    # Mask: valid where sigma is finite (i.e. within original wavelength coverage)
    mask_new = np.isfinite(sigma_arr)

    reduction = n_pix_orig / n_pix_new if n_pix_new > 0 else float("inf")
    logger.info(
        "Resolution degrade: %d epochs, %d → %d pixels/epoch (%.1f× reduction).",
        n_epochs,
        n_pix_orig,
        n_pix_new,
        reduction,
    )
    print(
        "[resolution_degrade] {:d} epochs: {:d} → {:d} pixels/epoch "
        "({:.1f}× reduction).".format(n_epochs, n_pix_orig, n_pix_new, reduction)
    )

    return Chunk(wl_arr, fl_arr, sigma_arr, date_arr, mask=mask_new)
