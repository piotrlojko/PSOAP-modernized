> [!CAUTION]
> ## This repo is a vibe-coded fork of the original `PSOAP` code, and does not work yet 😒
> The goal is to modernize the original repository and completely abandon the `HDF5` data format for the input specra files.
>
> All credit belongs to **Ian Czekala** and **Collaborators**.

# PSOAP-modernized
Pronounced "soap."

[![Documentation Status](https://readthedocs.org/projects/psoap/badge/?version=latest)](https://psoap.readthedocs.io/en/latest/?badge=latest)

**Precision Spectroscopic Orbits A-Parametrically**

PSOAP-modernized is a Python package for jointly inferring orbital parameters and component spectra from time-series spectroscopy using Gaussian processes.

Supported production workflows in this repository are:

- `SB2` — double-lined spectroscopic binaries
- `ST2` — hierarchical triples with two visible components
- `ST3` — hierarchical triples with three visible components

## Installation

PSOAP-modernized targets Python 3.12+.

```bash
git clone https://github.com/piotrlojko/PSOAP-modernized.git
cd PSOAP-modernized
python -m pip install -e .
python setup.py build_ext --inplace
```

The single-core sampler uses `emcee`; install it if your environment did not bring it in automatically:

```bash
python -m pip install emcee
```

Sanity check your install:

```bash
psoap-initialize --check
```

## Quick start (SB2)

1. Create a new empty working directory for your run.
2. Initialize a configuration template:

```bash
psoap-initialize --model SB2
```

3. Create `spectra_list.txt` with two whitespace-separated columns and a single-line header:

```text
filename date
/path/to/epoch_001.txt 2459001.1234
/path/to/epoch_002.txt 2459004.5678
```

4. Each spectrum file should contain 3 columns and a single-line header:

```text
wavelength_Angstrom flux sigma
5265.0001 0.9982 0.0100
5265.0214 1.0031 0.0101
...
```

5. Edit `config.yaml` (copied in step 2) so that model parameters and file paths match your dataset.
6. Run MCMC:

```bash
psoap-sample
```

Or for chunked parallel likelihood evaluation:

```bash
psoap-sample-parallel 0
```

7. In the selected output run directory (for example `output/run00/`), summarize chains:

```bash
cd output/run00
psoap-plot-samples --burn 200
```

## Documentation

- ReadTheDocs: <https://psoap.readthedocs.io>
- Local Sphinx sources: `/home/runner/work/PSOAP-modernized/PSOAP-modernized/doc`

Build docs locally:

```bash
cd doc
make dirhtml
cd .build/dirhtml
python -m http.server
```
and open the generated link in a web browser to see the contents of the documentation.

## Tests

From the repository root:

```bash
pytest
```

## Citation

If you use PSOAP in scientific work, please cite:

*Czekala et al. 2017, ApJ, 840, 49* — <http://adsabs.harvard.edu/abs/2017ApJ...840...49C>
