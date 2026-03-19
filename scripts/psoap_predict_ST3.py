#!/usr/bin/env python
"""
Predict and plot component spectra for an ST3 (triple-lined) model.

Reads ``config.yaml`` from the current directory and uses the GP/orbit
parameters to predict the disentangled component spectra ``f``, ``g``, ``h``.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from astropy.io import ascii

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

spectra_table = ascii.read(config["spectra_list"])
filenames = list(spectra_table["filename"])
dates = np.array(spectra_table["date"])

chunk = Chunk.from_textfiles(
    filenames, dates,
    limit=config.get("epoch_limit"),
    wl_min=config.get("wl_min"),
    wl_max=config.get("wl_max"),
)

wls    = chunk.wl
fl     = chunk.fl
sigma  = chunk.sigma
dates_obs = chunk.date1D

q_in   = pars["q_in"]
K_in   = pars["K_in"]
e_in   = pars["e_in"]
omega_in = pars["omega_in"]
P_in   = pars["P_in"]
T0_in  = pars["T0_in"]
q_out  = pars["q_out"]
K_out  = pars["K_out"]
e_out  = pars["e_out"]
omega_out = pars["omega_out"]
P_out  = pars["P_out"]
T0_out = pars["T0_out"]
gamma  = pars["gamma"]
amp_f  = pars["amp_f"]
l_f    = pars["l_f"]
amp_g  = pars["amp_g"]
l_g    = pars["l_g"]
amp_h  = pars["amp_h"]
l_h    = pars["l_h"]

orb = orbit.ST3(q_in, K_in, e_in, omega_in, P_in, T0_in,
                q_out, K_out, e_out, omega_out, P_out, T0_out,
                gamma, obs_dates=dates_obs)

vAs, vBs, vCs = orb.get_velocities()

n_epochs, n_pix = wls.shape

wls_A = redshift(wls, -vAs[:, np.newaxis])
wls_B = redshift(wls, -vBs[:, np.newaxis])
wls_C = redshift(wls, -vCs[:, np.newaxis])

lwls_A = np.log(wls_A)
lwls_B = np.log(wls_B)
lwls_C = np.log(wls_C)

mu, Sigma = covariance.predict_f_g_h(
    lwls_A.flatten(), lwls_B.flatten(), lwls_C.flatten(),
    fl.flatten(), sigma.flatten(),
    lwls_A.flatten(), lwls_B.flatten(), lwls_C.flatten(),
    mu_f=0.0, mu_g=0.0, mu_h=0.0,
    amp_f=amp_f, l_f=l_f,
    amp_g=amp_g, l_g=l_g,
    amp_h=amp_h, l_h=l_h,
)

N = n_pix * n_epochs
mu_f = mu[:N].reshape(n_epochs, n_pix)
mu_g = mu[N: 2 * N].reshape(n_epochs, n_pix)
mu_h = mu[2 * N:].reshape(n_epochs, n_pix)

plots_dir = "plots_ST3"
os.makedirs(plots_dir, exist_ok=True)

for i in range(n_epochs):
    fig, ax = plt.subplots(nrows=4, sharex=True)
    ax[0].plot(wls[i], fl[i], ".", color="0.4")
    ax[0].set_ylabel(r"$f + g + h$")
    ax[1].plot(wls[i], mu_f[i], "b")
    ax[1].set_ylabel(r"$f$")
    ax[2].plot(wls[i], mu_g[i], "g")
    ax[2].set_ylabel(r"$g$")
    ax[3].plot(wls[i], mu_h[i], "r")
    ax[3].set_ylabel(r"$h$")
    ax[-1].set_xlabel(r"$\lambda\;[\AA]$")
    fig.savefig(os.path.join(plots_dir, "epoch_{:0>2}.png".format(i)), dpi=150)
    plt.close("all")

print("Plots saved to", plots_dir)
