"""
Utilities for parameter conversion and chain analysis.

Supported models:
  * **SB2**  – double-lined spectroscopic binary.
  * **ST2**  – double-lined tertiary (inner SB2 pair + invisible outer companion C).
  * **ST3**  – triple-lined tertiary (inner SB2 pair + visible outer companion C).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Parameter registries
# ---------------------------------------------------------------------------

# Ordered list of all parameters for each model.
registered_params = {
    "SB2": ["q", "K", "e", "omega", "P", "T0", "gamma",
            "amp_f", "l_f", "amp_g", "l_g"],
    # ST2: inner SB2 pair (A+B both visible) + dynamically-coupled outer
    #      companion C (not visible in the spectrum).
    # Hook: same GP structure as SB2 (two components f, g).
    "ST2": ["q_in", "K_in", "e_in", "omega_in", "P_in", "T0_in",
            "K_out", "e_out", "omega_out", "P_out", "T0_out", "gamma",
            "amp_f", "l_f", "amp_g", "l_g"],
    "ST3": ["q_in", "K_in", "e_in", "omega_in", "P_in", "T0_in",
            "q_out", "K_out", "e_out", "omega_out", "P_out", "T0_out", "gamma",
            "amp_f", "l_f", "amp_g", "l_g", "amp_h", "l_h"],
}

registered_models = list(registered_params.keys())

# Number of orbital parameters (up to and including gamma).
n_params_orb = {
    model: (registered_params[model].index("gamma") + 1)
    for model in registered_params
}

# LaTeX labels used for plotting.
registered_labels = {
    "SB2": [r"$q$", r"$K$", r"$e$", r"$\omega$", r"$P$", r"$T_0$",
            r"$\gamma$", r"$a_f$", r"$l_f$", r"$a_g$", r"$l_g$"],
    "ST2": [r"$q_\mathrm{in}$", r"$K_\mathrm{in}$", r"$e_\mathrm{in}$",
            r"$\omega_\mathrm{in}$", r"$P_\mathrm{in}$", r"$T_{0,\mathrm{in}}$",
            r"$K_\mathrm{out}$", r"$e_\mathrm{out}$", r"$\omega_\mathrm{out}$",
            r"$P_\mathrm{out}$", r"$T_{0,\mathrm{out}}$", r"$\gamma$",
            r"$a_f$", r"$l_f$", r"$a_g$", r"$l_g$"],
    "ST3": [r"$q_\mathrm{in}$", r"$K_\mathrm{in}$", r"$e_\mathrm{in}$",
            r"$\omega_\mathrm{in}$", r"$P_\mathrm{in}$", r"$T_{0,\mathrm{in}}$",
            r"$q_\mathrm{out}$", r"$K_\mathrm{out}$", r"$e_\mathrm{out}$",
            r"$\omega_\mathrm{out}$", r"$P_\mathrm{out}$",
            r"$T_{0,\mathrm{out}}$", r"$\gamma$",
            r"$a_f$", r"$l_f$", r"$a_g$", r"$l_g$", r"$a_h$", r"$l_h$"],
}


# ---------------------------------------------------------------------------
# Parameter vector conversion
# ---------------------------------------------------------------------------

def convert_vector(p, model, fix_params, **kwargs):
    '''
    Unroll a vector of free parameter values into a full parameter vector,
    splitting it into orbital and GP parts.

    Args:
        p (1D np.array): free parameter values.
        model (str): one of ``"SB2"``, ``"ST2"``, ``"ST3"``.
        fix_params (list of str): names of parameters held fixed.
        **kwargs: ``{param_name: default_value}`` for all parameters.

    Returns:
        tuple: ``(par_orb, par_GP)`` — the orbital and GP parameter arrays.
    '''
    reg_params = registered_params[model]
    nparams = len(reg_params)

    fit_ind = [i for i, param in enumerate(reg_params) if param not in fix_params]
    fix_ind = [reg_params.index(param) for param in fix_params]

    par_vec = np.empty(nparams, dtype=np.float64)
    par_vec[fit_ind] = p
    par_vec[fix_ind] = [kwargs[pn] for pn in fix_params]

    ind_split = n_params_orb[model]
    return par_vec[:ind_split], par_vec[ind_split:]


def convert_dict(model, fix_params, **kwargs):
    '''
    Convert a dictionary of all parameter values (e.g. from ``config.yaml``)
    to a free-parameter vector, skipping fixed parameters.

    Args:
        model (str): one of ``"SB2"``, ``"ST2"``, ``"ST3"``.
        fix_params (list of str): names of parameters held fixed.
        **kwargs: ``{param_name: value}`` for all parameters.

    Returns:
        1D np.array: the free-parameter vector.
    '''
    reg_params = registered_params[model]
    fit_params = [p for p in reg_params if p not in fix_params]
    return np.array([kwargs[pn] for pn in fit_params], dtype=np.float64)


def get_labels(model, fix_params):
    '''
    Return LaTeX labels for all free parameters of ``model``.

    Args:
        model (str): one of ``"SB2"``, ``"ST2"``, ``"ST3"``.
        fix_params (list of str): names of parameters held fixed.

    Returns:
        list of str: LaTeX labels for the free parameters.
    '''
    reg_params = registered_params[model]
    reg_labels = registered_labels[model]
    return [reg_labels[i] for i, p in enumerate(reg_params)
            if p not in fix_params]


# ---------------------------------------------------------------------------
# Chain statistics and diagnostics
# ---------------------------------------------------------------------------

def gelman_rubin(samplelist):
    '''
    Compute the Gelman-Rubin convergence statistic for a list of chains.

    Args:
        samplelist (list of 2D np.array): each element is a flatchain of shape
            ``(n_iters, n_params)``.  All chains must have the same shape.

    Returns:
        np.array: ``R_hat`` values for each parameter.
    '''
    full_iterations = len(samplelist[0])
    assert full_iterations % 2 == 0, \
        "Number of iterations must be even."
    shape = samplelist[0].shape
    for flatchain in samplelist:
        assert len(flatchain) == full_iterations and flatchain.shape == shape, \
            "All chains must have the same shape."

    n = full_iterations // 2
    m = 2 * len(samplelist)
    nparams = samplelist[0].shape[-1]

    chains = np.empty((n, m, nparams))
    for k, flatchain in enumerate(samplelist):
        chains[:, 2 * k, :] = flatchain[:n]
        chains[:, 2 * k + 1, :] = flatchain[n:]

    avg_phi_j = np.mean(chains, axis=0, dtype="f8")
    avg_phi = np.mean(chains, axis=(0, 1), dtype="f8")

    B = n / (m - 1.0) * np.sum((avg_phi_j - avg_phi)**2, axis=0, dtype="f8")
    s2j = 1.0 / (n - 1.0) * np.sum((chains - avg_phi_j)**2, axis=0, dtype="f8")
    W = 1.0 / m * np.sum(s2j, axis=0, dtype="f8")
    var_hat = (n - 1.0) / n * W + B / n
    R_hat = np.sqrt(var_hat / W)

    print("R_hat: {}".format(R_hat))
    if np.any(R_hat >= 1.1):
        print("Consider running chains longer; not all R_hats < 1.1.")
    return R_hat


def estimate_covariance(flatchain, ndim=0):
    '''
    Estimate an optimal Metropolis-Hastings proposal covariance from a flatchain.

    Args:
        flatchain (2D np.array): shape ``(n_samples, n_params)``.
        ndim (int): number of dimensions (0 = infer from flatchain).

    Returns:
        2D np.array: the scaled optimal-jump covariance matrix.
    '''
    import matplotlib.pyplot as plt

    d = flatchain.shape[1] if ndim == 0 else ndim
    cov = np.cov(flatchain, rowvar=False)
    cor = np.corrcoef(flatchain, rowvar=False)

    fig, ax = plt.subplots(figsize=(0.5 * d, 0.5 * d))
    ext = (0.5, d + 0.5, 0.5, d + 0.5)
    ax.imshow(cor, origin="upper", vmin=-1, vmax=1, cmap="bwr",
              interpolation="none", extent=ext)
    fig.savefig("cor_coefficient.png")

    opt_jump = 2.38**2 / d * cov
    std_dev = np.sqrt(np.diag(cov))
    print("Optimal jump std-devs:", 2.38 / np.sqrt(d) * std_dev)
    return opt_jump
