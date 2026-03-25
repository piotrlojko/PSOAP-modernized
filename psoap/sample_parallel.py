"""
Multi-core MCMC sampler for SB2/ST3 (and ST2 hook) models using one process
per spectral chunk.

Reads a ``config.yaml`` in the current directory.  Orbital and GP parameters
are proposed by a top-level Metropolis-Hastings sampler (``psoap.samplers``);
the log-likelihood for each chunk is evaluated in a dedicated child process.
"""

import argparse
import os
import shutil
import gc
import logging
from functools import partial
from multiprocessing import Process, Pipe

import yaml
import numpy as np

import psoap.constants as C
from psoap.data import Chunk, lredshift, replicate_wls
from psoap.input_parsing import (
    parse_spectra_list,
    print_and_log_model_config,
)
from psoap import utils
from psoap import orbit
from psoap import covariance
from psoap.samplers import StateSampler


def _load_config():
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("You need to copy a config.yaml file to this directory, "
              "and then edit the values to your particular case.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-core MCMC for an SB2/ST3 model.")
    parser.add_argument("run_index", type=int, default=0,
                        help="Output subdirectory index.")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug commands to log.log")
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

    # ----- load data -----
    filenames, dates = parse_spectra_list(config["spectra_list"])

    wl_ranges = config.get("wl_ranges", [None])  # list of (wl_min, wl_max) dicts
    n_chunks = len(wl_ranges)
    chunk_keys = np.arange(n_chunks)

    chunk_data = []
    for wl_range in wl_ranges:
        wl_min = wl_range.get("wl_min") if isinstance(wl_range, dict) else None
        wl_max = wl_range.get("wl_max") if isinstance(wl_range, dict) else None
        chunkSpec = Chunk.from_textfiles(
            filenames, dates,
            limit=config.get("epoch_limit"),
            wl_min=wl_min,
            wl_max=wl_max,
        )

        # Optionally apply barycentric velocity correction as a preprocessing step
        if not config.get("barycentric_corrected", True):
            from psoap.data import compute_barycentric_corrections
            ra = config["target_ra"]
            dec = config["target_dec"]
            v_bary = compute_barycentric_corrections(chunkSpec.date1D, ra, dec)
            print("Applying barycentric corrections (km/s):", v_bary)
            chunkSpec.apply_barycentric_correction(v_bary)

        chunkSpec.apply_mask()
        chunk_data.append(chunkSpec)

    # ----- create convert_vector partial -----
    convert_vector_p = partial(
        utils.convert_vector, model=model,
        fix_params=config["fix_params"], **pars)

    # ----- priors -----
    def prior_SB2(p):
        (q, K, e, omega, P, T0, gamma), (amp_f, l_f, amp_g, l_g) = \
            convert_vector_p(p)
        if (q <= 0.0 or K <= 0.0 or e < 0.0 or e >= 1.0 or P <= 0.0
                or omega < -90 or omega > 450
                or amp_f <= 0.0 or l_f <= 0.0
                or amp_g <= 0.0 or l_g <= 0.0):
            return -np.inf
        return 0.0

    def prior_ST3(p):
        (q_in, K_in, e_in, omega_in, P_in, T0_in,
         q_out, K_out, e_out, omega_out, P_out, T0_out, gamma), \
            (amp_f, l_f, amp_g, l_g, amp_h, l_h) = convert_vector_p(p)
        if (q_in <= 0.0 or K_in <= 0.0 or e_in < 0.0 or e_in >= 1.0
                or P_in <= 0.0 or omega_in < -90 or omega_in > 450
                or q_out <= 0.0 or K_out <= 0.0 or e_out < 0.0 or e_out >= 1.0
                or P_out <= 0.0 or omega_out < -90 or omega_out > 450
                or amp_f <= 0.0 or l_f <= 0.0
                or amp_g <= 0.0 or l_g <= 0.0
                or amp_h <= 0.0 or l_h <= 0.0):
            return -np.inf
        return 0.0

    # Hook: ST2 uses the same structure as SB2
    prior_ST2 = prior_SB2

    try:
        from prior import prior
        print("Loaded user-defined prior.")
    except ImportError:
        print("Using default prior.")
        _priors = {"SB2": prior_SB2, "ST2": prior_ST2, "ST3": prior_ST3}
        prior = _priors[model]

    # ----- Worker class (each chunk = one subprocess) -----
    class Worker:
        def __init__(self):
            self.func_dict = {
                "INIT": self.initialize,
                "LNPROB": self.lnprob,
                "FINISH": self.finish,
            }
            if args.debug:
                self.logger = logging.getLogger(self.__class__.__name__)

        def initialize(self, key):
            self.key = key
            data = chunk_data[key]
            self.lwl = data.lwl
            self.fl = data.fl
            self.sigma = data.sigma * config.get("soften", 1.0)
            self.mask = data.mask
            self.date1D = data.date1D
            self.N = data.N
            self.V11 = np.empty((self.N, self.N), dtype=np.float64)
            if args.debug:
                self.logger = logging.getLogger(
                    "{} {}".format(self.__class__.__name__, key))
                self.logger.info("Initializing chunk {}.".format(key))

        def lnprob(self, p):
            p_orb, p_GP = convert_vector_p(p)
            velocities = orbit.models[model](*p_orb, self.date1D).get_velocities()
            if np.any(np.abs(np.array(velocities)) >= C.c_kms):
                return -np.inf
            lwls = replicate_wls(self.lwl, velocities, self.mask)
            lnp = covariance.lnlike[model](self.V11, *lwls, self.fl, self.sigma, *p_GP)
            gc.collect()
            return lnp

        def finish(self, *args):
            pass

        def brain(self, conn):
            self.conn = conn
            alive = True
            while alive:
                alive = self.interpret()
            self.conn.send("DEAD")

        def interpret(self):
            fname, arg = self.conn.recv()
            func = self.func_dict.get(fname)
            if func is None:
                return False
            response = func(arg)
            if response is not None:
                self.conn.send(response)
            return True

    # ----- spawn workers -----
    worker = Worker()
    pconns = {}
    cconns = {}
    ps = {}
    for key in chunk_keys:
        pconn, cconn = Pipe()
        pconns[key] = pconn
        cconns[key] = cconn
        p = Process(target=worker.brain, args=(cconn,))
        p.start()
        ps[key] = p
    for key, pconn in pconns.items():
        pconn.send(("INIT", key))

    # ----- global lnprob (distributes to workers) -----
    def lnprob(p):
        lnprior = prior(p)
        if not np.isfinite(lnprior):
            return -np.inf
        for pconn in pconns.values():
            pconn.send(("LNPROB", p))
        lnps = np.array([pconn.recv() for pconn in pconns.values()])
        return float(np.sum(lnps)) + lnprior

    # ----- sampling -----
    dim = len(utils.registered_params[model]) - len(config["fix_params"])
    p0 = utils.convert_dict(model, config["fix_params"], **pars)

    print("Testing starting position …")
    lnp0 = lnprob(p0)
    if not np.isfinite(lnp0):
        print("Starting position evaluates to -inf. Aborting.")
        for pconn in pconns.values():
            pconn.send(("FINISH", None))
            pconn.send(("DIE", None))
        for p in ps.values():
            p.join()
            p.terminate()
        raise RuntimeError("Bad starting position.")
    else:
        print("Starting position good. lnp = {:.4f}".format(lnp0))

    try:
        cov = np.load(config["opt_jump"])
        print("Using optimal jump matrix from", config["opt_jump"])
    except (KeyError, FileNotFoundError):
        print("Using hand-specified jumps.")
        cov = utils.convert_dict(model, config["fix_params"],
                                 **config["jumps"])**2 * np.eye(dim)

    sampler = StateSampler(lnprob, p0, cov, outdir=routdir)

    n_samples = config.get("samples", 1000)
    print("Running MH sampler for {} steps …".format(n_samples))
    for i, (pos, lnp, _) in enumerate(sampler.sample(p0, iterations=n_samples)):
        if (i + 1) % 100 == 0:
            print("Iteration {:d}  lnp = {:.4f}  acc = {:.3f}".format(
                i + 1, lnp, sampler.acceptance_fraction))

    print("Acceptance fraction: {:.3f}".format(sampler.acceptance_fraction))
    np.save(routdir + "lnprob.npy", sampler.lnprobability)
    np.save(routdir + "flatchain.npy", sampler.flatchain)

    # ----- shut down workers -----
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))
    for p in ps.values():
        p.join()
        p.terminate()


if __name__ == "__main__":
    main()
