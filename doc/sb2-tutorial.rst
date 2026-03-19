.. _sb2-tutorial:

SB2 Spectral Disentangling Tutorial
===================================

This tutorial shows the minimal workflow for disentangling a double-lined spectroscopic binary (SB2) with ``PSOAP``.

1. Initialize a new SB2 project
-------------------------------

From an empty working directory, run:

::

    $ psoap-initialize --model SB2

This creates a ``config.yaml`` template for the SB2 model.

2. Configure the SB2 model
--------------------------

Edit ``config.yaml`` and set:

* ``data_file`` to your input HDF5 spectra file
* orbital parameters (``q``, ``K``, ``e``, ``omega``, ``P``, ``T0``, ``gamma``)
* GP parameters (``amp_f``, ``l_f``, ``amp_g``, ``l_g``)
* MCMC settings (``nwalkers``, ``nsamples``)

See :ref:`configuration` for parameter details.

3. Run the sampler
------------------

::

    $ psoap-sample

For multi-core execution, use:

::

    $ psoap-sample-parallel

4. Plot posterior and RV outputs
--------------------------------

::

    $ psoap-plot-samples

This produces the standard posterior diagnostic plots and radial velocity summaries for the SB2 solution.
