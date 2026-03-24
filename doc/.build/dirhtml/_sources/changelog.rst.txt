=========
Changelog
=========

------
v0.2.0
------

Documentation and workflow refresh
----------------------------------

* Removed obsolete ``attic/`` legacy code and outdated notebook-derived docs.
* Rewrote README and Sphinx docs to match the current CLI and text-based data workflow.
* Expanded installation, getting started, and SB2 tutorial content with complete runnable steps.
* Updated documentation metadata and repository links for ``piotrlojko/PSOAP-modernized``.

Packaging consistency
---------------------

* Aligned ``psoap.__version__`` with package version ``0.2.0``.
* Added missing runtime dependency ``emcee`` required by ``psoap-sample``.

------
v0.1.1
------

standardized different model selections
---------------------------------------

Now models can be specified as combinations of the number of gravitationally significant bodies, and the number of spectroscopically significant bodies. See `models.md` for more information. This has an impact for the way `orbit.py` is used.


Calibration optimization
------------------------

For now, only implemented for the `ST3` model.

log-lambda input coordinates
----------------------------

Using log-lambda input coordinates instead of lambda enables a small but not insignificant speedup in the calculation of the kernel spacing, since the calculation of velocity spacing becomes a simple subtraction.

The following modules and scripts have been converted

* psoap_sample_parallel.py
* covariance.py
* matrix_functions.pyx
* psoap_retrieve_SB2.py
* psoap_retrieve_ST3.py

v0.1.0
------

Beta Release corresponding to code used for paper on the arXiv.
