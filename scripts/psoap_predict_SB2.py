#!/usr/bin/env python
"""
Predict and plot component spectra for an SB2 model.

Reads ``config.yaml`` from the current directory, loads the spectra, and
uses the best-fit (or specified) GP/orbit parameters to predict the
disentangled component spectra ``f`` and ``g`` for each epoch.
"""

import argparse

parser = argparse.ArgumentParser(
    description="Predict disentangled SB2 component spectra.")
parser.add_argument("--draws", type=int, default=0,
                    help="Plot this many GP draws in addition to the mean.")
args = parser.parse_args()

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from astropy.io import ascii
from scipy.linalg import cho_factor, cho_solve

from psoap import constants as C
from psoap.data import redshift, Chunk
from psoap import covariance
from psoap import orbit
from psoap import utils

try:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("You need a config.yaml file in this directory.")
    raise

pars = config["parameters"]

# Load spectra
spectra_table = ascii.read(config["spectra_list"])
filenames = list(spectra_table["filename"])
dates = np.array(spectra_table["date"])

chunk = Chunk.from_textfiles(
    filenames, dates,
    limit=config.get("epoch_limit"),
    wl_min=config.get("wl_min"),
    wl_max=config.get("wl_max"),
)

# Data arrays
wls = chunk.wl
fl = chunk.fl
sigma = chunk.sigma
dates_obs = chunk.date1D

q      = pars["q"]
K      = pars["K"]
e      = pars["e"]
omega  = pars["omega"]
P      = pars["P"]
T0     = pars["T0"]
gamma  = pars["gamma"]
amp_f  = pars["amp_f"]
l_f    = pars["l_f"]
amp_g  = pars["amp_g"]
l_g    = pars["l_g"]

orb = orbit.SB2(q, K, e, omega, P, T0, gamma, obs_dates=dates_obs)
vAs, vBs = orb.get_component_velocities()

# Doppler-shift wavelengths to rest frame of each component
wls_A = redshift(wls, -vAs[:, np.newaxis])
wls_B = redshift(wls, -vBs[:, np.newaxis])

lwls_A = np.log(wls_A)
lwls_B = np.log(wls_B)

n_epochs, n_pix = wls_A.shape

# Predict component spectra
mu, Sigma = covariance.predict_f_g(
    lwls_A.flatten(), lwls_B.flatten(),
    fl.flatten(), sigma.flatten(),
    lwls_A.flatten(), lwls_B.flatten(),
    mu_f=0.0, mu_g=0.0,
    amp_f=amp_f, l_f=l_f,
    amp_g=amp_g, l_g=l_g,
)

mu_sum, Sigma_sum = covariance.predict_f_g_sum(
    lwls_A.flatten(), lwls_B.flatten(),
    fl.flatten(), sigma.flatten(),
    lwls_A.flatten(), lwls_B.flatten(),
    mu_fg=1.0,
    amp_f=amp_f, l_f=l_f,
    amp_g=amp_g, l_g=l_g,
)

mu_f = mu[:(n_pix * n_epochs)].reshape(n_epochs, n_pix)
mu_g = mu[(n_pix * n_epochs):].reshape(n_epochs, n_pix)
mu_sum = mu_sum.reshape(n_epochs, n_pix)

plots_dir = "plots_SB2"
os.makedirs(plots_dir, exist_ok=True)

for i in range(n_epochs):
    fig, ax = plt.subplots(nrows=4, sharex=True)

    ax[0].plot(wls[i], fl[i], ".", color="0.4")
    ax[0].plot(wls[i], mu_sum[i], "b")
    ax[0].plot(wls[i], mu_f[i] + mu_g[i] + 1.0, "m", ls="-.")
    ax[0].set_ylabel(r"$f + g$")

    ax[1].plot(wls[i], mu_f[i], "b")
    ax[1].set_ylabel(r"$f$")

    ax[2].plot(wls[i], mu_g[i], "g")
    ax[2].set_ylabel(r"$g$")

    residuals = fl[i] - mu_sum[i]
    ax[3].plot(wls[i], residuals, ".", color="0.4")
    ax[3].set_ylabel("residuals")
    ax[-1].set_xlabel(r"$\lambda\;[\AA]$")

    fig.savefig(os.path.join(plots_dir, "epoch_{:0>2}.png".format(i)), dpi=150)

    # Residual histogram
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist(residuals / sigma[i], density=True)
    sig = np.linspace(-4, 4, 50)
    ax.plot(sig, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * sig**2))
    ax.set_xlabel(r"$(f-\hat{f})/\sigma$")
    fig.savefig(os.path.join(plots_dir, "epoch_{:0>2}_hist.png".format(i)), dpi=150)

    plt.close("all")

print("Plots saved to", plots_dir)
