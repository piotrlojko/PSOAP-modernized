#!/usr/bin/env python
"""
Plot the orbital solution for an SB2 or ST3 model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from astropy.io import ascii

from psoap import orbit
from psoap import utils

try:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("You need a config.yaml file in this directory.")
    raise

pars = config["parameters"]
model = config["model"]

spectra_table = ascii.read(config["spectra_list"])
dates = np.array(spectra_table["date"])

dates_fine = np.linspace(dates.min() - 0.1 * np.ptp(dates),
                         dates.max() + 0.1 * np.ptp(dates), 500)

os.makedirs("plots_orbit", exist_ok=True)

if model in ("SB2", "ST2"):
    q      = pars["q"]
    K      = pars["K"]
    e      = pars["e"]
    omega  = pars["omega"]
    P      = pars["P"]
    T0     = pars["T0"]
    gamma  = pars["gamma"]

    orb = orbit.models[model](q, K, e, omega, P, T0, gamma, obs_dates=dates)
    vAs, vBs = orb.get_velocities()
    vA_fine, vB_fine = orb.get_velocities(dates_fine)

    fig, ax = plt.subplots()
    ax.plot(dates_fine, vA_fine, "b")
    ax.plot(dates, vAs, "bo", label=r"$v_A$ obs")
    ax.plot(dates_fine, vB_fine, "g")
    ax.plot(dates, vBs, "go", label=r"$v_B$ obs")
    ax.axhline(gamma, ls="-.", color="0.5")
    ax.set_xlabel(r"$t$ [days]")
    ax.set_ylabel(r"$v$ [km s$^{-1}$]")
    ax.legend()
    fig.savefig("plots_orbit/orbit.png", dpi=300)
    plt.close("all")

elif model == "ST3":
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

    orb = orbit.ST3(q_in, K_in, e_in, omega_in, P_in, T0_in,
                    q_out, K_out, e_out, omega_out, P_out, T0_out,
                    gamma, obs_dates=dates)
    vAs, vBs, vCs = orb.get_velocities()
    vA_fine, vB_fine, vC_fine = orb.get_velocities(dates_fine)

    fig, ax = plt.subplots()
    ax.plot(dates_fine, vA_fine, "b", label=r"$v_A$")
    ax.plot(dates, vAs, "bo")
    ax.plot(dates_fine, vB_fine, "g", label=r"$v_B$")
    ax.plot(dates, vBs, "go")
    ax.plot(dates_fine, vC_fine, "r", label=r"$v_C$")
    ax.plot(dates, vCs, "ro")
    ax.axhline(gamma, ls="-.", color="0.5")
    ax.set_xlabel(r"$t$ [days]")
    ax.set_ylabel(r"$v$ [km s$^{-1}$]")
    ax.legend()
    fig.savefig("plots_orbit/orbit.png", dpi=300)
    plt.close("all")

else:
    print("Unsupported model: {}".format(model))

print("Orbit plot saved to plots_orbit/orbit.png")
