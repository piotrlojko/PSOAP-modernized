#!/usr/bin/env python
"""
Reconstruct disentangled component spectra f, g, h for an ST3 (triple-lined)
model, and save them as text files and NumPy arrays.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from astropy.io import ascii

from psoap.data import lredshift, redshift, Chunk
from psoap import covariance
from psoap import orbit

parser = argparse.ArgumentParser(
    description="Reconstruct ST3 component spectra from the GP mean.")
parser.add_argument("--draws", type=int, default=0,
                    help="Number of GP draws to overlay on the mean prediction.")
args = parser.parse_args()

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
n_pix = chunk.n_pix

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

dates_obs = chunk.date1D
orb = orbit.ST3(q_in, K_in, e_in, omega_in, P_in, T0_in,
                q_out, K_out, e_out, omega_out, P_out, T0_out,
                gamma, obs_dates=dates_obs)
vAs, vBs, vCs = orb.get_velocities()

wls = chunk.wl
lwls = chunk.lwl

wls_A = redshift(wls, -vAs[:, np.newaxis])
wls_B = redshift(wls, -vBs[:, np.newaxis])
wls_C = redshift(wls, -vCs[:, np.newaxis])
lwls_A = lredshift(lwls, -vAs[:, np.newaxis])
lwls_B = lredshift(lwls, -vBs[:, np.newaxis])
lwls_C = lredshift(lwls, -vCs[:, np.newaxis])

chunk.apply_mask()
mask = chunk.mask
lwls_A_flat = lwls_A[mask]
lwls_B_flat = lwls_B[mask]
lwls_C_flat = lwls_C[mask]
fl = chunk.fl
sigma = chunk.sigma

n_pix_predict = 2 * n_pix
lwls_predict = np.linspace(lwls_A_flat.min(), lwls_A_flat.max(), n_pix_predict)
wls_predict = np.exp(lwls_predict)

mu, Sigma = covariance.predict_f_g_h(
    lwls_A_flat, lwls_B_flat, lwls_C_flat,
    fl, sigma,
    lwls_predict, lwls_predict, lwls_predict,
    mu_f=0.0, mu_g=0.0, mu_h=0.0,
    amp_f=amp_f, l_f=l_f,
    amp_g=amp_g, l_g=l_g,
    amp_h=amp_h, l_h=l_h,
)

sigma_diag = np.sqrt(np.diag(Sigma))
mu_f = mu[:n_pix_predict]
mu_g = mu[n_pix_predict: 2 * n_pix_predict]
mu_h = mu[2 * n_pix_predict:]
sigma_f = sigma_diag[:n_pix_predict]
sigma_g = sigma_diag[n_pix_predict: 2 * n_pix_predict]
sigma_h = sigma_diag[2 * n_pix_predict:]

plots_dir = "plots_ST3"
os.makedirs(plots_dir, exist_ok=True)

fig, ax = plt.subplots(nrows=3, sharex=True)
ax[0].plot(wls_predict, mu_f, "b")
ax[0].set_ylabel(r"$f$")
ax[1].plot(wls_predict, mu_g, "g")
ax[1].set_ylabel(r"$g$")
ax[2].plot(wls_predict, mu_h, "r")
ax[2].set_ylabel(r"$h$")
ax[-1].set_xlabel(r"$\lambda\;[\AA]$")
fig.savefig(os.path.join(plots_dir, "reconstructed.png"), dpi=300)
plt.close("all")

np.save(os.path.join(plots_dir, "f.npy"),
        np.vstack((wls_predict, mu_f, sigma_f)))
np.save(os.path.join(plots_dir, "g.npy"),
        np.vstack((wls_predict, mu_g, sigma_g)))
np.save(os.path.join(plots_dir, "h.npy"),
        np.vstack((wls_predict, mu_h, sigma_h)))

for label, wl_arr, mu_arr, sig_arr in [
        ("f", wls_predict, mu_f, sigma_f),
        ("g", wls_predict, mu_g, sigma_g),
        ("h", wls_predict, mu_h, sigma_h)]:
    np.savetxt(os.path.join(plots_dir, "{}.txt".format(label)),
               np.column_stack([wl_arr, mu_arr + 1.0, sig_arr]),
               header="wavelength[AA]  flux  sigma",
               fmt="%.8f  %.8f  %.8f")

print("Component spectra saved to", plots_dir)
