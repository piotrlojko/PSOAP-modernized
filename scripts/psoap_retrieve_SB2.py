#!/usr/bin/env python
"""
Reconstruct the disentangled component spectra f (primary) and g (secondary)
for an SB2 model, and save them as text files and NumPy arrays.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from astropy.io import ascii

from psoap import constants as C
from psoap.data import lredshift, redshift, Chunk
from psoap import covariance
from psoap import orbit

parser = argparse.ArgumentParser(
    description="Reconstruct SB2 component spectra from the GP mean.")
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
n_epochs = chunk.n_epochs
n_pix = chunk.n_pix

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

dates_obs = chunk.date1D
orb = orbit.SB2(q, K, e, omega, P, T0, gamma, obs_dates=dates_obs)
vAs, vBs = orb.get_velocities()

wls = chunk.wl
lwls = chunk.lwl

# Doppler-shift to rest frames
wls_A = redshift(wls, -vAs[:, np.newaxis])
wls_B = redshift(wls, -vBs[:, np.newaxis])
lwls_A = lredshift(lwls, -vAs[:, np.newaxis])
lwls_B = lredshift(lwls, -vBs[:, np.newaxis])

# Apply mask and flatten
chunk.apply_mask()
mask = chunk.mask
lwls_A_flat = lwls_A[mask]
lwls_B_flat = lwls_B[mask]
fl = chunk.fl
sigma = chunk.sigma

# Prediction grid (2× finer than data, common wavelength for both components)
n_pix_predict = 2 * n_pix
lwls_A_predict = np.linspace(lwls_A_flat.min(), lwls_A_flat.max(), n_pix_predict)
wls_A_predict = np.exp(lwls_A_predict)
lwls_B_predict = lwls_A_predict  # same grid
wls_B_predict = wls_A_predict

mu, Sigma = covariance.predict_f_g(
    lwls_A_flat, lwls_B_flat,
    fl, sigma,
    lwls_A_predict, lwls_B_predict,
    mu_f=0.0, mu_g=0.0,
    amp_f=amp_f, l_f=l_f,
    amp_g=amp_g, l_g=l_g,
)

sigma_diag = np.sqrt(np.diag(Sigma))
mu_f = mu[:n_pix_predict]
sigma_f = sigma_diag[:n_pix_predict]
mu_g = mu[n_pix_predict:]
sigma_g = sigma_diag[n_pix_predict:]

plots_dir = "plots_SB2"
os.makedirs(plots_dir, exist_ok=True)

# GP draws
if args.draws > 0:
    mu_draw = np.random.multivariate_normal(mu, Sigma, size=args.draws)
    mu_draw_f = mu_draw[:, :n_pix_predict]
    mu_draw_g = mu_draw[:, n_pix_predict:]

fig, ax = plt.subplots(nrows=2, sharex=True)

if args.draws > 0:
    for j in range(args.draws):
        ax[0].plot(wls_A_predict, mu_draw_f[j], color="0.2", lw=0.5)
        ax[1].plot(wls_B_predict, mu_draw_g[j], color="0.2", lw=0.5)

ax[0].plot(wls_A_predict, mu_f, "b")
ax[0].set_ylabel(r"$f$")
ax[1].plot(wls_B_predict, mu_g, "g")
ax[1].set_ylabel(r"$g$")
ax[-1].set_xlabel(r"$\lambda\;[\AA]$")
fig.savefig(os.path.join(plots_dir, "reconstructed.png"), dpi=300)
plt.close("all")

# Save component spectra: rows = [wavelength, mean_flux, sigma]
np.save(os.path.join(plots_dir, "f.npy"), np.vstack((wls_A_predict, mu_f, sigma_f)))
np.save(os.path.join(plots_dir, "g.npy"), np.vstack((wls_B_predict, mu_g, sigma_g)))

np.savetxt(os.path.join(plots_dir, "f.txt"),
           np.column_stack([wls_A_predict, mu_f + 1.0, sigma_f]),
           header="wavelength[AA]  flux  sigma", fmt="%.8f  %.8f  %.8f")
np.savetxt(os.path.join(plots_dir, "g.txt"),
           np.column_stack([wls_B_predict, mu_g + 1.0, sigma_g]),
           header="wavelength[AA]  flux  sigma", fmt="%.8f  %.8f  %.8f")

print("Component spectra saved to", plots_dir)
