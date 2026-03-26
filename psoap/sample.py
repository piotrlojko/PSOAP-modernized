"""
Single-core MCMC sampler for SB2 (and ST3/ST2 hook) models.

Reads a ``config.yaml`` in the current directory that describes the model
parameters, data, and sampler settings.
"""

import argparse
import yaml
import os
import shutil
import logging
from functools import partial

import numpy as np
import emcee

import psoap.constants as C
from psoap.data import replicate_wls
from psoap.input_parsing import (
    print_and_log_model_config,
)
from psoap import utils
from psoap import orbit
from psoap import covariance
from psoap.preprocessing import build_preprocessed_chunks


def _load_config():
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("You need to copy a config.yaml file to this directory, "
              "and then edit the values to your particular case.")
        raise


def prior_SB2(p, convert_vector_p):
    (q, K, e, omega, P, T0, gamma), (amp_f, l_f, amp_g, l_g) = \
        convert_vector_p(p)
    if (q <= 0.0 or K <= 0.0 or e < 0.0 or e >= 1.0 or P <= 0.0
            or omega < -90 or omega > 450
            or amp_f <= 0.0 or l_f <= 0.0
            or amp_g <= 0.0 or l_g <= 0.0):
        return -np.inf
    return 0.0


def prior_ST3(p, convert_vector_p):
    (q_in, K_in, e_in, omega_in, P_in, T0_in,
     q_out, K_out, e_out, omega_out, P_out, T0_out, gamma), \
        (amp_f, l_f, amp_g, l_g, amp_h, l_h) = convert_vector_p(p)
    if (q_in <= 0.0 or K_in <= 0.0 or e_in < 0.0 or e_in >= 1.0 or P_in <= 0.0
            or omega_in < -90 or omega_in > 450
            or q_out <= 0.0 or K_out <= 0.0 or e_out < 0.0 or e_out >= 1.0
            or P_out <= 0.0 or omega_out < -90 or omega_out > 450
            or amp_f <= 0.0 or l_f <= 0.0
            or amp_g <= 0.0 or l_g <= 0.0
            or amp_h <= 0.0 or l_h <= 0.0):
        return -np.inf
    return 0.0


# Hook: ST2 uses the same prior structure as SB2
prior_ST2 = prior_SB2

_default_priors = {"SB2": prior_SB2, "ST2": prior_ST2, "ST3": prior_ST3}


def main():
    parser = argparse.ArgumentParser(
        description="Run single-core MCMC for an SB2/ST3 model.")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug commands to log.log")
    parser.add_argument("--run-index", type=int, default=0,
                        help="Output subdirectory index.")
    args = parser.parse_args()

    config = _load_config()
    print_and_log_model_config(config)
    pars = config["parameters"]
    model = config["model"]

    # ----- output directory -----
    routdir = config["outdir"] + "/run{:0>2}/".format(args.run_index)
    if os.path.exists(routdir):
        print("Deleting", routdir)
        shutil.rmtree(routdir)
    print("Creating", routdir)
    os.makedirs(routdir)
    shutil.copy("config.yaml", routdir + "config.yaml")

    if args.debug:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            filename="{}log.log".format(routdir), level=logging.DEBUG,
            filemode="w", datefmt="%m/%d/%Y %I:%M:%S %p")

    # ----- load and preprocess data -----
    chunk_data = build_preprocessed_chunks(config)
    for data in chunk_data:
        data.apply_mask()

    lwls = [data.lwl for data in chunk_data]
    fls = [data.fl for data in chunk_data]
    sigmas = [data.sigma * config.get("soften", 1.0) for data in chunk_data]
    masks = [data.mask for data in chunk_data]
    dates = [data.date1D for data in chunk_data]
    V11s = [np.empty((data.N, data.N), dtype=np.float64) for data in chunk_data]

    convert_vector_p = partial(
        utils.convert_vector, model=model,
        fix_params=config["fix_params"], **pars)

    # ----- prior -----
    try:
        from prior import prior as user_prior
        print("Loaded user-defined prior.")
        prior_fn = user_prior
    except ImportError:
        print("Using default prior.")
        prior_fn = partial(_default_priors[model], convert_vector_p=convert_vector_p)

    # ----- lnprob -----
    def lnprob(p):
        lnprior = prior_fn(p)
        if not np.isfinite(lnprior):
            return -np.inf

        p_orb, p_GP = convert_vector_p(p)
        velocities = orbit.models[model](*p_orb, dates[0]).get_velocities()
        if np.any(np.abs(np.array(velocities)) >= C.c_kms):
            return -np.inf

        lnp = 0.0
        for lwl, fl, sigma, mask, V11 in zip(lwls, fls, sigmas, masks, V11s):
            shifted = replicate_wls(lwl, velocities, mask)
            lnp += covariance.lnlike[model](V11, *shifted, fl, sigma, *p_GP)
        return lnp + lnprior

    # ----- sampler -----
    dim = len(utils.registered_params[model]) - len(config["fix_params"])
    p0_center = utils.convert_dict(model, config["fix_params"], **pars)

    n_walkers = config.get("n_walkers", max(2 * dim, 32))
    if n_walkers % 2 != 0:
        n_walkers += 1

    try:
        cov = np.load(config["opt_jump"])
        print("Using optimal jumps from", config["opt_jump"])
        spread = np.sqrt(np.diag(cov))
    except (KeyError, FileNotFoundError):
        print("Using hand-specified jumps.")
        spread = utils.convert_dict(model, config["fix_params"],
                                    **config["jumps"])

    rng = np.random.default_rng(config.get("seed", None))
    p0_walkers = p0_center + spread * rng.standard_normal((n_walkers, dim))

    print("Testing starting position …")
    lnp0 = lnprob(p0_center)
    if not np.isfinite(lnp0):
        raise RuntimeError(
            "Starting position evaluates to -inf. Check config.yaml.")
    print("Starting lnp: {:.4f}".format(lnp0))

    sampler = emcee.EnsembleSampler(n_walkers, dim, lnprob)

    n_samples = config.get("samples", 1000)
    print("Running {} walkers for {} steps …".format(n_walkers, n_samples))
    sampler.run_mcmc(p0_walkers, n_samples, progress=True)

    print("Acceptance fraction:", sampler.acceptance_fraction.mean())
    np.save(routdir + "flatchain.npy", sampler.get_chain(flat=True))
    np.save(routdir + "lnprob.npy", sampler.get_log_prob(flat=True))


if __name__ == "__main__":
    main()
