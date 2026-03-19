#!/usr/bin/env python
"""
Generate synthetic ST3 (triple-lined) spectra from three template spectra
and a given hierarchical triple orbit.

Templates must be 2-column text files (wavelength [AA], flux) named
``primary_wl_fl.txt``, ``secondary_wl_fl.txt``, and ``tertiary_wl_fl.txt``
in the current directory.  Output text spectra and a ``spectra_list.txt``
are written to the ``ST3/`` directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from psoap.data import redshift, Chunk
from psoap import orbit

# ---------------------------------------------------------------------------
# Orbital parameters
# ---------------------------------------------------------------------------
q_in     = 0.4
K_in     = 5.0    # km/s
e_in     = 0.2
omega_in = 10.0   # deg
P_in     = 10.0   # days
T0_in    = 0.0
q_out    = 0.2
K_out    = 4.0    # km/s
e_out    = 0.2
omega_out = 80.0  # deg
P_out    = 100.0  # days
T0_out   = 3.0
gamma    = 5.0    # km/s

n_epochs = 20
obs_dates = np.linspace(5, 150, num=n_epochs)

# Wavelength windows [Angstroms]
chunk_wls = [[5240, 5250], [5255, 5265], [5270, 5280]]

# Flux fractions
alpha = 0.5
beta  = 0.3
# tertiary fraction = 1 - alpha - beta

S_N = 40
noise_amp = 1.0 / S_N

outdir = "ST3"
os.makedirs(outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# Orbit
# ---------------------------------------------------------------------------
orb = orbit.ST3(q_in, K_in, e_in, omega_in, P_in, T0_in,
                q_out, K_out, e_out, omega_out, P_out, T0_out,
                gamma, obs_dates=obs_dates)
vAs, vBs, vCs = orb.get_component_velocities()

dates_fine = np.linspace(0, 35, 200)
vA_fine, vB_fine, vC_fine = orb.get_component_velocities(dates_fine)

for label, arr in [("vAs", vAs), ("vBs", vBs), ("vCs", vCs)]:
    np.save(os.path.join(outdir, "{}_relative.npy".format(label)), arr - arr[0])

fig, ax = plt.subplots(nrows=4, sharex=True)
ax[0].plot(dates_fine, vA_fine, "b"); ax[0].plot(obs_dates, vAs, "bo")
ax[0].plot(dates_fine, vB_fine, "g"); ax[0].plot(obs_dates, vBs, "go")
ax[0].plot(dates_fine, vC_fine, "r"); ax[0].plot(obs_dates, vCs, "ro")
ax[0].axhline(gamma, ls="-.", color="0.5")
ax[-1].set_xlabel(r"$t$ [days]")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "orbit.png"), dpi=150)
plt.close("all")

# ---------------------------------------------------------------------------
# Load templates (2-column: wl, flux)
# ---------------------------------------------------------------------------
for fname in ("primary_wl_fl.txt", "secondary_wl_fl.txt", "tertiary_wl_fl.txt"):
    if not os.path.exists(fname):
        raise FileNotFoundError("Template '{}' not found.".format(fname))

primary   = np.loadtxt("primary_wl_fl.txt")
secondary = np.loadtxt("secondary_wl_fl.txt")
tertiary  = np.loadtxt("tertiary_wl_fl.txt")

wl_f, fl_f = primary[:, 0],   primary[:, 1]
wl_g, fl_g = secondary[:, 0], secondary[:, 1]
wl_h, fl_h = tertiary[:, 0],  tertiary[:, 1]

n_pix = min(len(wl_f), len(wl_g), len(wl_h))
wl   = wl_f[:n_pix]
fl_f = fl_f[:n_pix]
fl_g = fl_g[:n_pix]
fl_h = fl_h[:n_pix]

# ---------------------------------------------------------------------------
# Doppler-shift to each epoch
# ---------------------------------------------------------------------------
wls_f = np.array([redshift(wl, vAs[i]) for i in range(n_epochs)])
wls_g = np.array([redshift(wl, vBs[i]) for i in range(n_epochs)])
wls_h = np.array([redshift(wl, vCs[i]) for i in range(n_epochs)])

# ---------------------------------------------------------------------------
# Build combined spectra on a common wavelength grid for each chunk
# ---------------------------------------------------------------------------
rng = np.random.default_rng()

for chunk_idx, (wl0, wl1) in enumerate(chunk_wls):
    print("Creating chunk {} to {}".format(wl0, wl1))
    ind = (wls_f[0] > wl0) & (wls_f[0] < wl1)
    n_pix_common = int(np.sum(ind))

    wls_comb  = np.zeros((n_epochs, n_pix_common))
    fls_noise = np.empty((n_epochs, n_pix_common))
    sigmas    = noise_amp * np.ones((n_epochs, n_pix_common))

    for i in range(n_epochs):
        ind0 = np.searchsorted(wls_f[i], wl0)
        wl_common = wls_f[i, ind0: ind0 + n_pix_common]

        fl_f_c = interp1d(wls_f[i], fl_f, bounds_error=False, fill_value=1.0)(wl_common)
        fl_g_c = interp1d(wls_g[i], fl_g, bounds_error=False, fill_value=1.0)(wl_common)
        fl_h_c = interp1d(wls_h[i], fl_h, bounds_error=False, fill_value=1.0)(wl_common)

        fl_c = (alpha * fl_f_c + beta * fl_g_c
                + (1.0 - alpha - beta) * fl_h_c)

        wls_comb[i]  = wl_common
        fls_noise[i] = fl_c + rng.normal(scale=noise_amp, size=n_pix_common)

    date_arr = obs_dates[:, np.newaxis] * np.ones((n_epochs, n_pix_common))
    chunk = Chunk(wls_comb, fls_noise, sigmas, date_arr)
    prefix = os.path.join(outdir, "chunk{:02d}_".format(chunk_idx))
    chunk.save_textfiles(prefix=prefix)

    # Write spectra_list for this chunk
    list_fname = os.path.join(
        outdir, "spectra_list_chunk{:02d}.txt".format(chunk_idx))
    with open(list_fname, "w") as fh:
        fh.write("filename  date\n")
        for i in range(n_epochs):
            fh.write("{}epoch_{:03d}.txt  {:.5f}\n".format(prefix, i, obs_dates[i]))
    print("  Saved", list_fname)
