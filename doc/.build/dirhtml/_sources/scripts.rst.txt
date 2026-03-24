.. _scripts:

Command-line Scripts
====================

PSOAP-modernized installs the following console commands via ``setup.py``.

psoap-initialize
----------------

Create a local configuration template or verify installation.

::

    $ psoap-initialize --help

Key options:

* ``--check``: print installation diagnostics
* ``--model {SB2,ST2,ST3}``: choose config template to copy as ``config.yaml``

Example:

::

    $ psoap-initialize --model SB2

psoap-sample
------------

Single-core MCMC sampler using ``emcee``.

::

    $ psoap-sample --help

Key options:

* ``--run-index``: output run number (default ``0``)
* ``--debug``: enable detailed logging in run output directory

Example:

::

    $ psoap-sample --run-index 0

psoap-sample-parallel
---------------------

Parallel Metropolis-Hastings sampler using one process per wavelength chunk.

::

    $ psoap-sample-parallel --help

Usage requires positional run index:

::

    $ psoap-sample-parallel 0

psoap-plot-samples
------------------

Post-processing tool for chain diagnostics and summary statistics. Run it in a
run directory that contains ``flatchain.npy``.

::

    $ psoap-plot-samples --help

Common options:

* ``--burn N``: discard first ``N`` samples
* ``--cov``: estimate and save ``opt_jump.npy``
* ``--tri``: also create a corner plot (requires ``corner`` package)

Example:

::

    $ cd output/run00
    $ psoap-plot-samples --burn 200 --cov
