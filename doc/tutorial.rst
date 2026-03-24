.. _tutorial:

Tutorial
========

This page summarizes the modern PSOAP workflow and links to the practical
step-by-step guide.

For a complete executable example, use :ref:`sb2-tutorial`.

Recommended workflow
--------------------

1. Install PSOAP and compile extensions (:ref:`getting-started`).
2. Initialize a run directory with ``psoap-initialize --model ...``.
3. Provide plain-text spectra and a ``spectra_list`` table.
4. Edit ``config.yaml`` orbital/GP parameters and run controls.
5. Run ``psoap-sample`` or ``psoap-sample-parallel``.
6. Post-process chains with ``psoap-plot-samples``.

Notes on legacy material
------------------------

Older PSOAP documentation referred to scripts such as
``psoap_initialize.py``, ``psoap_generate_chunks.py``, and HDF5-based helper
utilities. Those commands are not part of the current modernized package
workflow and should not be used with this repository.
