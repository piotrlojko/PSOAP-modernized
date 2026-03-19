.. _getting-started:

Getting Started
===============

Installation
------------

`PSOAP` requires a few packages standard to the scientific Python ecosystem. It is written and tested for currently maintained Python releases.

* ``numpy`` (>=1.26)
* ``scipy`` (>=1.12)
* ``astropy`` (>=6.0)
* ``matplotlib`` (>=3.8)
* ``pyyaml`` (>=6.0)
* ``cython`` (>=3.0)
* ``scikit-learn`` (optional, for alternative GP implementations)

All of these packages can be installed with your normal Python package manager. Once you have installed them, clone the `PSOAP` package from the `github repository <https://github.com/piotrlojko/PSOAP-modernized>`_ ::

    $ git clone https://github.com/piotrlojko/PSOAP-modernized.git
    $ cd PSOAP-modernized

and change to the top level ``PSOAP`` directory. Build and install the package via ::

    $ pip install -e .
    $ python setup.py build_ext --inplace

Which should build the cython extensions (used for faster matrix evaluations) and install the system scripts to your shell ``PATH``. To check that you've got everything installed properly, try running from your shell ::

    $ psoap-initialize --check
    PSOAP successfully installed and linked.
    Using Python Version 3.12.3

If this doesn't work, try double-checking the output from your install process to see if any errors popped up. If you are unable to fix these issues via the normal means of debugging python installs, please `raise an issue <https://github.com/piotrlojko/PSOAP-modernized/issues>`_ with specifics about your system.

Testing
-------

If you really want to make sure everything works on your system, you can run the test suite by installing the `pytest <https://docs.pytest.org/en/latest/>`_ package, changing to the directory where you cloned the repository, and then running ::

    $ pytest

If any of these tests fail, please report them by `raising an issue <https://github.com/piotrlojko/PSOAP-modernized/issues>`_ with specifics about your system.

Citing
------

If you use our paper, code, or a derivative of it in your research, we would really appreciate a citation to `Czekala et al. 2017 <http://adsabs.harvard.edu/abs/2017ApJ...840...49C>`_ ::

    @ARTICLE{2017ApJ...840...49C,
        author = {{Czekala}, I. and {Mandel}, K.~S. and {Andrews}, S.~M. and {Dittmann}, J.~A. and
        {Ghosh}, S.~K. and {Montet}, B.~T. and {Newton}, E.~R.},
        title = "{Disentangling Time-series Spectra with Gaussian Processes: Applications to Radial Velocity Analysis}",
        journal = {\apj},
        archivePrefix = "arXiv",
        eprint = {1702.05652},
        primaryClass = "astro-ph.SR",
        keywords = {binaries: spectroscopic, celestial mechanics, stars: fundamental parameters, stars: individual: LP661-13, techniques: radial velocities, techniques: spectroscopic},
         year = 2017,
        month = may,
        volume = 840,
          eid = {49},
        pages = {49},
          doi = {10.3847/1538-4357/aa6aab},
        adsurl = {http://adsabs.harvard.edu/abs/2017ApJ...840...49C},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Copyright Ian Czekala and collaborators 2016-17.


Because the ``PSOAP`` package is still under occasional development, it may be wise to install the package in 'development mode' via the following commands::

    $ pip install -e .
    $ python setup.py build_ext --inplace


If you installed the package via development mode, then it in the case that the package is upgraded, it is easy to upgrade your local copy by simply pulling down the latest changes and rerunning the build script::

    $ git pull
    $ python setup.py build_ext --inplace
