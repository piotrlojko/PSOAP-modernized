#!/usr/bin/env python
"""
Draw random realizations from the GP posterior for an SB2 model and plot them.

Reads pre-computed ``mu.npy`` and ``Sigma.npy`` from the ``plots_SB2`` directory
(as produced by ``psoap_retrieve_SB2``).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Plot GP draws for the SB2 component spectra.")
parser.add_argument("--draws", type=int, default=10,
                    help="Number of GP draws to plot.")
args = parser.parse_args()

plots_dir = "plots_SB2"

mu = np.load(os.path.join(plots_dir, "mu.npy"))
Sigma = np.load(os.path.join(plots_dir, "Sigma.npy"))
n_pix_predict = len(mu) // 2

f_data = np.load(os.path.join(plots_dir, "f.npy"))
g_data = np.load(os.path.join(plots_dir, "g.npy"))

wls_A_predict = f_data[0]
wls_B_predict = g_data[0]
mu_f = mu[:n_pix_predict]
mu_g = mu[n_pix_predict:]

mu_draw = np.random.multivariate_normal(mu, Sigma, size=args.draws)
mu_draw_f = mu_draw[:, :n_pix_predict]
mu_draw_g = mu_draw[:, n_pix_predict:]

np.save(os.path.join(plots_dir, "f_draws.npy"), mu_draw_f)
np.save(os.path.join(plots_dir, "g_draws.npy"), mu_draw_g)

fig, ax = plt.subplots(nrows=2, sharex=True)
for j in range(args.draws):
    ax[0].plot(wls_A_predict, mu_draw_f[j], color="0.2", lw=0.5)
    ax[1].plot(wls_B_predict, mu_draw_g[j], color="0.2", lw=0.5)

ax[0].plot(wls_A_predict, mu_f, "b")
ax[0].set_ylabel(r"$f$")
ax[1].plot(wls_B_predict, mu_g, "g")
ax[1].set_ylabel(r"$g$")
ax[-1].set_xlabel(r"$\lambda\;[\AA]$")
fig.savefig(os.path.join(plots_dir, "reconstructed_draws.png"), dpi=300)
plt.close("all")
print("Draws saved to", plots_dir)
