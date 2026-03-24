Covariance Routines
===================

PSOAP models stellar component spectra with squared-exponential Gaussian
process kernels in log-wavelength space.

For each component, the kernel uses amplitude ``amp_*`` and velocity-like
length scale ``l_*``. In multi-component models, component kernels are summed
and observational variance ``sigma^2`` is added to the diagonal.

Implementation notes
--------------------

* Fast matrix assembly is provided by the compiled Cython module
  :mod:`psoap.matrix_functions`.
* Core log-likelihood dispatch used by samplers is exposed via
  ``psoap.covariance.lnlike`` for models ``SB2``, ``ST2``, ``ST3``.
* Prediction helpers provide posterior means/covariances for component spectra.

API reference
-------------

.. automodule:: psoap.covariance
    :members:
