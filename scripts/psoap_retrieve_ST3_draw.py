#!/usr/bin/env python
"""
Draw random realizations from the GP posterior for an ST3 model.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Plot GP draws for the ST3 component spectra.")
parser.add_argument("--draws", type=int, default=10)
args = parser.parse_args()

plots_dir = "plots_ST3"

mu = np.load(os.path.join(plots_dir, "mu.npy"))
Sigma = np.load(os.path.join(plots_dir, "Sigma.npy"))
n_pix_predict = len(mu) // 3

f_data = np.load(os.path.join(plots_dir, "f.npy"))
wls_predict = f_data[0]
mu_f = mu[:n_pix_predict]
mu_g = mu[n_pix_predict: 2 * n_pix_predict]
mu_h = mu[2 * n_pix_predict:]

mu_draw = np.random.multivariate_normal(mu, Sigma, size=args.draws)
mu_draw_f = mu_draw[:, :n_pix_predict]
mu_draw_g = mu_draw[:, n_pix_predict: 2 * n_pix_predict]
mu_draw_h = mu_draw[:, 2 * n_pix_predict:]

fig, ax = plt.subplots(nrows=3, sharex=True)
for j in range(args.draws):
    ax[0].plot(wls_predict, mu_draw_f[j], color="0.2", lw=0.5)
    ax[1].plot(wls_predict, mu_draw_g[j], color="0.2", lw=0.5)
    ax[2].plot(wls_predict, mu_draw_h[j], color="0.2", lw=0.5)

ax[0].plot(wls_predict, mu_f, "b"); ax[0].set_ylabel(r"$f$")
ax[1].plot(wls_predict, mu_g, "g"); ax[1].set_ylabel(r"$g$")
ax[2].plot(wls_predict, mu_h, "r"); ax[2].set_ylabel(r"$h$")
ax[-1].set_xlabel(r"$\lambda\;[\AA]$")
fig.savefig(os.path.join(plots_dir, "reconstructed_draws.png"), dpi=300)
plt.close("all")
print("Draws saved to", plots_dir)
