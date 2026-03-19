#!/usr/bin/env python
"""
Generate synthetic SB2 spectra from two template spectra and a given orbit.

Template spectra must be plain-text files with two columns: wavelength [AA]
and normalized flux.  The script writes the combined (noisy) spectra as
3-column text files (wavelength, flux, sigma) in the output directory
and creates a ``spectra_list.txt`` that can be used directly with PSOAP.
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
q     = 0.2
K     = 5.0    # km/s
e     = 0.2
omega = 10.0   # deg
P     = 10.0   # days
T0    = 0.0    # JD epoch of periastron
gamma = 5.0    # km/s

# Observation dates [days]
obs_dates = np.array([2.1, 4.9, 8.0, 9.9, 12.2, 16.0, 16.9, 19.1, 22.3, 26.1])
n_epochs = len(obs_dates)

# Wavelength window [Angstroms]
wl_min = 5265.0
wl_max = 5275.0

# Noise
S_N = 60                         # per resolution element
noise_amp = 1.0 / (S_N / np.sqrt(2.5))

# Primary-to-secondary flux ratio
ratio = 0.2                      # flux_B / flux_A
alpha = 1.0 / (ratio + 1.0)     # fraction of total flux from A

# Output directory
outdir = "SB2"
os.makedirs(outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# Orbit
# ---------------------------------------------------------------------------
orb = orbit.SB2(q, K, e, omega, P, T0, gamma, obs_dates=obs_dates)
vAs, vBs = orb.get_component_velocities()

dates_fine = np.linspace(0, 30, 200)
vA_fine, vB_fine = orb.get_component_velocities(dates_fine)

np.save(os.path.join(outdir, "vAs_relative.npy"), vAs - vAs[0])
np.save(os.path.join(outdir, "vBs_relative.npy"), vBs - vBs[0])

fig, ax = plt.subplots(nrows=3, figsize=(6, 6))
ax[0].plot(dates_fine, vA_fine, "b")
ax[0].plot(obs_dates, vAs, "bo")
ax[0].plot(dates_fine, vB_fine, "g")
ax[0].plot(obs_dates, vBs, "go")
ax[0].axhline(gamma, ls="-.", color="0.5")
ax[1].plot(obs_dates, vAs - vAs[0], "bo")
ax[1].set_ylabel(r"$v_A$ relative")
ax[2].plot(obs_dates, vBs - vBs[0], "go")
ax[2].set_ylabel(r"$v_B$ relative")
ax[-1].set_xlabel(r"$t$ [days]")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "orbit.png"), dpi=150)
plt.close("all")

# ---------------------------------------------------------------------------
# Load template spectra (2-column text: wl, flux)
# ---------------------------------------------------------------------------
for fname in ("primary_wl_fl.txt", "secondary_wl_fl.txt"):
    if not os.path.exists(fname):
        raise FileNotFoundError(
            "Template spectrum '{}' not found in the current directory.".format(fname))

primary = np.loadtxt("primary_wl_fl.txt")
secondary = np.loadtxt("secondary_wl_fl.txt")
wl_f, fl_f = primary[:, 0], primary[:, 1]
wl_g, fl_g = secondary[:, 0], secondary[:, 1]

# Shorten to common length
n_pix = min(len(wl_f), len(wl_g))
wl   = wl_f[:n_pix]
fl_f = fl_f[:n_pix]
fl_g = fl_g[:n_pix]

# ---------------------------------------------------------------------------
# Create Doppler-shifted epochs for both components
# ---------------------------------------------------------------------------
wls_f = np.empty((n_epochs, n_pix))
wls_g = np.empty((n_epochs, n_pix))
for i in range(n_epochs):
    wls_f[i] = redshift(wl, vAs[i])
    wls_g[i] = redshift(wl, vBs[i])

# ---------------------------------------------------------------------------
# Build combined observed spectra on a common wavelength grid
# ---------------------------------------------------------------------------
# Use the reference (un-shifted) wavelength grid to define the common pixels
ind_ref = (wls_f[0] > wl_min) & (wls_f[0] < wl_max)
n_pix_common = np.sum(ind_ref)

wls_comb  = np.zeros((n_epochs, n_pix_common))
fls_f     = np.empty((n_epochs, n_pix_common))
fls_g     = np.empty((n_epochs, n_pix_common))
fls_noise = np.empty((n_epochs, n_pix_common))
sigmas    = noise_amp * np.ones((n_epochs, n_pix_common))

for i in range(n_epochs):
    ind0 = np.searchsorted(wls_f[i], wl_min)
    wl_common = wls_f[i, ind0: ind0 + n_pix_common]

    interp_f = interp1d(wls_f[i], fl_f, bounds_error=False, fill_value=1.0)
    interp_g = interp1d(wls_g[i], fl_g, bounds_error=False, fill_value=1.0)

    fl_f_common = interp_f(wl_common)
    fl_g_common = interp_g(wl_common)
    fl_common   = alpha * fl_f_common + (1 - alpha) * fl_g_common

    wls_comb[i]  = wl_common
    fls_f[i]     = fl_f_common
    fls_g[i]     = fl_g_common
    fls_noise[i] = fl_common + np.random.default_rng().normal(
        scale=noise_amp, size=n_pix_common)

date_arr = obs_dates[:, np.newaxis] * np.ones((n_epochs, n_pix_common))
chunk = Chunk(wls_comb, fls_noise, sigmas, date_arr)

# ---------------------------------------------------------------------------
# Save spectra as 3-column text files and write spectra_list.txt
# ---------------------------------------------------------------------------
chunk.save_textfiles(prefix=outdir + "/")

list_fname = os.path.join(outdir, "spectra_list.txt")
with open(list_fname, "w") as fh:
    fh.write("filename  date\n")
    for i in range(n_epochs):
        fh.write("{}/epoch_{:03d}.txt  {:.5f}\n".format(outdir, i, obs_dates[i]))

print("Wrote {} epoch files and {}".format(n_epochs, list_fname))

np.save(os.path.join(outdir, "fls_f.npy"),    alpha * fls_f)
np.save(os.path.join(outdir, "fls_g.npy"),    (1 - alpha) * fls_g)
np.save(os.path.join(outdir, "fls_comb.npy"), fls_noise)
