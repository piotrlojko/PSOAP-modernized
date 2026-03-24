Sampling
========

PSOAP provides two sampling entry points:

* ``psoap-sample``: single-core ``emcee`` ensemble sampler
* ``psoap-sample-parallel``: parallel Metropolis-Hastings sampler with one
  worker process per wavelength chunk

Both workflows:

* read ``config.yaml`` in the current directory
* build orbital velocities from :mod:`psoap.orbit`
* evaluate GP log-likelihood through :mod:`psoap.covariance`
* write ``flatchain.npy`` and ``lnprob.npy`` in ``outdir/runXX/``

The parallel sampler uses :class:`psoap.samplers.StateSampler` for the global
proposal loop.

API reference
-------------

.. automodule:: psoap.samplers
    :members:
