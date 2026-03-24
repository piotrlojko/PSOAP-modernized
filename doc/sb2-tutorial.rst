.. _sb2-tutorial:

SB2 Spectral Disentangling Tutorial
===================================

This tutorial walks through a practical end-to-end SB2 run using the current
PSOAP-modernized command-line workflow.

1. Prepare an analysis directory
--------------------------------

Create and enter a clean working directory:

::

    $ mkdir sb2-demo
    $ cd sb2-demo

Initialize the SB2 configuration template:

::

    $ psoap-initialize --model SB2

You now have a ``config.yaml``.

2. Prepare input files
----------------------

Create ``spectra_list.txt`` with two whitespace-separated columns named
``filename`` and ``date``. Example:

::

    filename date
    data/epoch_001.txt 2459001.1234
    data/epoch_002.txt 2459004.5678

Each ``epoch_XXX.txt`` must be a 3-column file:

* wavelength in Angstrom
* normalized flux (continuum near 1)
* per-pixel sigma

3. Configure ``config.yaml``
----------------------------

At minimum, verify and edit:

* ``spectra_list`` path
* ``model: SB2``
* orbital parameters in ``parameters``:
  ``q, K, e, omega, P, T0, gamma``
* GP parameters:
  ``amp_f, l_f, amp_g, l_g``
* proposal scales in ``jumps``
* fixed parameters in ``fix_params``
* run controls: ``samples``, ``seed``, ``outdir``

Optional wavelength restriction:

::

    wl_min: 5260.0
    wl_max: 5280.0

4. Run the sampler
------------------

Single-core (``emcee``):

::

    $ psoap-sample --run-index 0

Parallel chunked likelihood:

::

    $ psoap-sample-parallel 0

Both workflows write output to ``outdir/runXX/``.

5. Inspect and summarize chains
-------------------------------

Move into the run directory and generate diagnostics:

::

    $ cd output/run00
    $ psoap-plot-samples --burn 200 --cov

Typical outputs:

* ``chain.png``
* ``flatchain.npy``
* ``lnprob.npy``
* optional ``opt_jump.npy``

If using ``--tri``, a corner plot is also produced (requires the ``corner``
Python package).

6. Iterate
----------

Use early runs to tune:

* initial ``parameters``
* ``jumps`` or ``opt_jump``
* ``samples`` length
* fixed vs free parameter choices

Then rerun until chains are stable and physically plausible.
