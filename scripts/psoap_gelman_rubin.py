#!/usr/bin/env python
"""
Compute Gelman-Rubin convergence statistics across multiple MCMC chains.

Usage::

    psoap-gelman-rubin "output/run*/flatchain.npy" --burn 200
"""

import argparse
import sys
import numpy as np
from glob import glob
from astropy.table import Table
from astropy.io import ascii

parser = argparse.ArgumentParser(
    description="Measure Gelman-Rubin statistics across multiple chains.")
parser.add_argument("glob", help="Quoted glob pattern for flatchain files, "
                    "e.g. \"output/run*/flatchain.npy\".")
parser.add_argument("--burn", type=int, default=0,
                    help="Number of samples to discard from the beginning "
                         "of each chain as burn-in.")
args = parser.parse_args()


def gelman_rubin(samplelist):
    full_iterations = len(samplelist[0])
    assert full_iterations % 2 == 0, \
        "Number of iterations must be even."
    shape = samplelist[0].shape
    for flatchain in samplelist:
        assert len(flatchain) == full_iterations and flatchain.shape == shape, \
            "All chains must have the same shape."

    n = full_iterations // 2
    m = 2 * len(samplelist)
    nparams = samplelist[0].shape[-1]

    chains = np.empty((n, m, nparams))
    for k, flatchain in enumerate(samplelist):
        chains[:, 2 * k, :] = flatchain[:n]
        chains[:, 2 * k + 1, :] = flatchain[n:]

    avg_phi_j = np.mean(chains, axis=0, dtype="f8")
    avg_phi = np.mean(chains, axis=(0, 1), dtype="f8")
    B = n / (m - 1.0) * np.sum((avg_phi_j - avg_phi)**2, axis=0, dtype="f8")
    s2j = 1.0 / (n - 1.0) * np.sum((chains - avg_phi_j)**2, axis=0, dtype="f8")
    W = 1.0 / m * np.sum(s2j, axis=0, dtype="f8")
    var_hat = (n - 1.0) / n * W + B / n
    std_hat = np.sqrt(var_hat)
    R_hat = np.sqrt(var_hat / W)

    data = Table({"Value": avg_phi, "Uncertainty": std_hat},
                 names=["Value", "Uncertainty"])
    print(data)
    ascii.write(data, sys.stdout, format="latex",
                formats={"Value": "%0.3f", "Uncertainty": "%0.3f"})
    print("R_hat:", R_hat)
    if np.any(R_hat >= 1.1):
        print("Consider running chains longer; not all R_hats < 1.1.")


files = sorted(glob(args.glob))
flatchains = []
for f in files:
    try:
        flatchains.append(np.load(f)[args.burn:])
    except OSError as exc:
        print("{} skipped: {}".format(f, exc))

print("Using {} flatchains.".format(len(flatchains)))
assert len(flatchains) > 1, "Need at least 2 chains for Gelman-Rubin."

gelman_rubin(flatchains)

combined = np.concatenate(flatchains, axis=0)
print("Combined chain shape:", combined.shape)
np.save("flatchain_combined.npy", combined)
print("Saved to flatchain_combined.npy")
