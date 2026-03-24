Data
====

PSOAP loads per-epoch spectra from plain-text files through :mod:`psoap.data`.

Expected input format
---------------------

A run is defined by ``spectra_list`` (set in ``config.yaml``), containing two
columns:

* ``filename``: path to spectrum text file
* ``date``: observation date (JD)

Each spectrum file must have 3 columns:

1. wavelength [Angstrom]
2. normalized flux
3. sigma

Data handling details
---------------------

:class:`psoap.data.Chunk` is the core container. ``Chunk.from_textfiles``:

* reads all epochs
* interpolates onto a common wavelength grid (from first epoch)
* supports optional ``wl_min`` / ``wl_max`` clipping
* stores arrays for wavelength, flux, sigma, and epoch dates

Masking is applied with :meth:`psoap.data.Chunk.apply_mask` before likelihood
calls.

API reference
-------------

.. automodule:: psoap.data
    :members:

.. automodule:: psoap.utils
    :members:
