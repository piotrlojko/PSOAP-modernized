Reconstruction and Diagnostics
==============================

After sampling, use ``psoap-plot-samples`` in a run directory to inspect chain
quality and summarize fitted parameters.

Typical products include:

* ``chain.png`` for trace diagnostics
* ``flatchain_burned.npy`` after burn-in trimming
* optional ``opt_jump.npy`` when ``--cov`` is requested
* optional ``triangle.png`` when ``--tri`` is requested and ``corner`` is
  installed

The script also prints parameter values for the latest sample in a
``config.yaml``-compatible order.
