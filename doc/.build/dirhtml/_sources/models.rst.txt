.. _models:

Models
======

PSOAP uses a model code that combines dynamical hierarchy (``SB`` or ``ST``)
with the number of spectroscopically visible components.

Supported models in this repository
-----------------------------------

``SB2``
    Double-lined spectroscopic binary.

``ST2``
    Hierarchical triple with an inner visible SB2 pair and an outer companion
    that affects dynamics but is not directly visible spectroscopically.

``ST3``
    Hierarchical triple with three visible components.

Orbital model implementation
----------------------------

Model classes are implemented in :mod:`psoap.orbit` and dispatched through
``orbit.models``:

* ``SB2 -> psoap.orbit.SB2``
* ``ST2 -> psoap.orbit.ST2``
* ``ST3 -> psoap.orbit.ST3``

The samplers call ``get_velocities()`` on these models to obtain component
radial velocities at each epoch.

Parameter bookkeeping
---------------------

The canonical parameter order for each model is defined in
:mod:`psoap.utils` (`registered_params`). This ordering is used consistently by:

* config parsing
* fixed/free parameter conversion
* sampler vectors
* plotting labels

See :ref:`configuration` and :ref:`sb2-tutorial` for practical setup.
