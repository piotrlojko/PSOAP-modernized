.. _getting-started:

Getting Started
===============

Installation
------------

PSOAP-modernized is tested on Python 3.12 and requires a compiled Cython
extension for fast covariance operations.

Core dependencies:

* ``numpy>=1.26``
* ``scipy>=1.12``
* ``astropy>=6.0``
* ``matplotlib>=3.8``
* ``pyyaml>=6.0``
* ``cython>=3.0``

Single-core sampling (``psoap-sample``) additionally requires:

* ``emcee``

Clone and install:

::

    $ git clone https://github.com/piotrlojko/PSOAP-modernized.git
    $ cd PSOAP-modernized
    $ python -m pip install -e .
    $ python setup.py build_ext --inplace
    $ python -m pip install emcee

Verify installation:

::

    $ psoap-initialize --check
    PSOAP successfully installed and linked.

Create a working directory
--------------------------

Run PSOAP from a dedicated project directory containing your ``config.yaml`` and
input spectra list. For an SB2 project:

::

    $ mkdir my-sb2-run
    $ cd my-sb2-run
    $ psoap-initialize --model SB2

This creates a local ``config.yaml`` template.

Input data format
-----------------

PSOAP expects a text table (``spectra_list`` in ``config.yaml``) with two
columns:

* ``filename``: path to a per-epoch spectrum file
* ``date``: observation time (JD)

Each spectrum file must contain exactly three columns:

1. wavelength in Angstroms
2. normalized flux
3. flux uncertainty (sigma)

The loader interpolates all epochs onto a shared wavelength grid and can apply
optional wavelength limits (``wl_min`` / ``wl_max``).

Run sampling
------------

From the run directory:

::

    $ psoap-sample

For chunk-based parallel likelihood evaluation:

::

    $ psoap-sample-parallel 0

Output products are written under ``outdir/runXX/`` (default
``output/run00/``), including:

* ``flatchain.npy``
* ``lnprob.npy``
* copied ``config.yaml``

Inspect chains
--------------

::

    $ cd output/run00
    $ psoap-plot-samples --burn 200

This generates chain diagnostics (``chain.png``) and prints the last sampled
parameter values.

Testing
-------

From repository root:

::

    $ pytest

Citation
--------

If PSOAP contributes to scientific work, cite
`Czekala et al. 2017 <http://adsabs.harvard.edu/abs/2017ApJ...840...49C>`_.
