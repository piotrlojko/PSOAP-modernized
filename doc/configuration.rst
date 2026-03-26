.. _configuration:

Configuration
=============

PSOAP is configured through a local ``config.yaml`` in your run directory.
Generate a template with:

::

    $ psoap-initialize --model SB2

or ``--model ST2`` / ``--model ST3``.

Top-level keys
--------------

``spectra_list``
    Path to a text table with columns ``filename`` and ``date``.

``model``
    One of ``SB2``, ``ST2``, ``ST3``.

``epoch_limit``
    Maximum number of epochs to load (``~`` for all).

``soften``
    Multiplicative factor applied to per-pixel ``sigma``.

``parameters``
    Dictionary of orbital and GP hyperparameters for selected model.

``jumps``
    Proposal scale per free parameter.

``fix_params``
    List of parameter names to hold fixed.

``samples``
    Number of MCMC steps.

``seed``
    RNG seed (``~`` for unseeded).

``opt_jump``
    Optional ``.npy`` covariance matrix file for proposals.

``outdir``
    Output directory root.

Preprocessing and chunking controls
-----------------------------------

PSOAP now applies the same preprocessing sequence in both ``psoap-sample`` and
``psoap-sample-parallel``:

1. optional barycentric correction (when ``barycentric_corrected: false``),
2. optional wavelength-window cut (``wl_min`` / ``wl_max``),
3. optional resolution degradation (``resolution_degrade``),
4. automatic chunk planning (always enabled).

Optional wavelength-window keys:

* ``wl_min``
* ``wl_max``

Automatic chunk planning is always used and is configured via ``auto_chunk``.
If ``auto_chunk`` is omitted, planner defaults are used.

Supported ``auto_chunk`` keys:

* ``memory_fraction``: fraction of available RAM to budget per worker.
* ``safety_factor``: headroom multiplier on the dense ``N²`` matrix estimate.
* ``target_workers``: target number of parallel chunks to enable CPU utilization
  (defaults to CPU count).
* ``min_pixels_per_chunk``: lower bound on chunk size when splitting for worker parallelism.
* ``max_chunks``: hard cap on the number of chunks.
* ``max_wallclock_hours``: optional runtime guard for planner estimates.

Model parameter sets
--------------------

SB2
^^^
Orbital: ``q, K, e, omega, P, T0, gamma``

GP: ``amp_f, l_f, amp_g, l_g``

ST2
^^^
Orbital: ``q_in, K_in, e_in, omega_in, P_in, T0_in, K_out, e_out, omega_out, P_out, T0_out, gamma``

GP: ``amp_f, l_f, amp_g, l_g``

ST3
^^^
Orbital: ``q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma``

GP: ``amp_f, l_f, amp_g, l_g, amp_h, l_h``

Data assumptions
----------------

Input spectra are plain text files with three columns:

1. wavelength [Angstrom]
2. normalized flux
3. flux uncertainty

PSOAP interpolates each epoch onto the first epoch's wavelength grid and uses
``fill_value=1.0`` for flux and ``fill_value=inf`` for sigma outside coverage.
