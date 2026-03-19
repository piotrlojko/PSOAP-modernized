"""
Covariance matrix construction and GP likelihood evaluation for PSOAP.

Supports double-lined spectroscopic binaries (SB2) and triple-lined systems
(ST3), with hooks for double-lined tertiary systems (ST2).

The underlying covariance kernel is a squared-exponential (Gaussian) kernel in
log-wavelength space, implemented via a fast Cython routine
(``psoap.matrix_functions``).  A scikit-learn-based alternative is also
provided via :func:`lnlike_f_g_sklearn`.
"""

import numpy as np
from numpy.polynomial import Chebyshev as Ch

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

import psoap
from psoap import constants as C
from psoap import matrix_functions
from psoap.data import lredshift


# ---------------------------------------------------------------------------
# GP prediction utilities
# ---------------------------------------------------------------------------

def predict_f_g(lwl_f, lwl_g, fl_fg, sigma_fg, lwl_f_predict, lwl_g_predict,
                mu_f, amp_f, l_f, mu_g, amp_g, l_g, get_Sigma=True):
    '''
    Given that f + g is the flux that we are modeling, jointly predict the
    individual component spectra.

    Args:
        lwl_f (1D np.array): log-wavelengths for primary component (data).
        lwl_g (1D np.array): log-wavelengths for secondary component (data).
        fl_fg (1D np.array): observed (combined) flux.
        sigma_fg (1D np.array): flux uncertainties.
        lwl_f_predict (1D np.array): prediction log-wavelengths for primary.
        lwl_g_predict (1D np.array): prediction log-wavelengths for secondary.
        mu_f (float): GP mean for the primary.
        amp_f (float): GP amplitude for the primary.
        l_f (float): GP length scale for the primary [km/s].
        mu_g (float): GP mean for the secondary.
        amp_g (float): GP amplitude for the secondary.
        l_g (float): GP length scale for the secondary [km/s].
        get_Sigma (bool): whether to compute and return the predictive covariance.

    Returns:
        (mu, Sigma) or mu depending on ``get_Sigma``.
    '''
    assert len(lwl_f) == len(lwl_g), "Input wavelengths must be the same length."
    n_pix = len(lwl_f)

    assert len(lwl_f_predict) == len(lwl_g_predict), \
        "Prediction wavelengths must be the same length."
    n_pix_predict = len(lwl_f_predict)

    mu_f_vec = mu_f * np.ones(n_pix_predict)
    mu_g_vec = mu_g * np.ones(n_pix_predict)
    mu_cat = np.hstack((mu_f_vec, mu_g_vec))

    V11_f = np.empty((n_pix, n_pix), dtype=np.float64)
    V11_g = np.empty((n_pix, n_pix), dtype=np.float64)
    matrix_functions.fill_V11_f(V11_f, lwl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, lwl_g, amp_g, l_g)

    B = V11_f + V11_g
    B[np.diag_indices_from(B)] += sigma_fg**2
    factor, flag = cho_factor(B)

    V11_f_pred = np.empty((n_pix_predict, n_pix_predict), dtype=np.float64)
    V11_g_pred = np.empty((n_pix_predict, n_pix_predict), dtype=np.float64)
    matrix_functions.fill_V11_f(V11_f_pred, lwl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g_pred, lwl_g_predict, amp_g, l_g)

    zeros = np.zeros((n_pix_predict, n_pix_predict))
    A = np.vstack((np.hstack([V11_f_pred, zeros]),
                   np.hstack([zeros, V11_g_pred])))

    V12_f = np.empty((n_pix_predict, n_pix), dtype=np.float64)
    V12_g = np.empty((n_pix_predict, n_pix), dtype=np.float64)
    matrix_functions.fill_V12_f(V12_f, lwl_f_predict, lwl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, lwl_g_predict, lwl_g, amp_g, l_g)

    C_mat = np.vstack((V12_f, V12_g))

    mu = mu_cat + np.dot(C_mat, cho_solve((factor, flag), fl_fg - 1.0))

    if get_Sigma:
        Sigma = A - np.dot(C_mat, cho_solve((factor, flag), C_mat.T))
        return mu, Sigma
    else:
        return mu


def predict_f_g_sum(lwl_f, lwl_g, fl_fg, sigma_fg,
                    lwl_f_predict, lwl_g_predict,
                    mu_fg, amp_f, l_f, amp_g, l_g):
    '''
    Predict the *sum* of the two component spectra using the GP model.

    Args:
        lwl_f (1D np.array): log-wavelengths for the primary component.
        lwl_g (1D np.array): log-wavelengths for the secondary component.
        fl_fg (1D np.array): observed combined flux.
        sigma_fg (1D np.array): flux uncertainties.
        lwl_f_predict (1D np.array): prediction log-wavelengths (primary).
        lwl_g_predict (1D np.array): prediction log-wavelengths (secondary).
        mu_fg (float): GP mean for the sum.
        amp_f (float): GP amplitude for the primary.
        l_f (float): GP length scale for the primary [km/s].
        amp_g (float): GP amplitude for the secondary.
        l_g (float): GP length scale for the secondary [km/s].

    Returns:
        (mu, Sigma): predictive mean and covariance for the sum.
    '''
    assert len(lwl_f) == len(lwl_g), "Input wavelengths must be the same length."

    M = len(lwl_f_predict)
    N = len(lwl_f)

    V11_f = np.empty((M, M), dtype=np.float64)
    V11_g = np.empty((M, M), dtype=np.float64)
    matrix_functions.fill_V11_f(V11_f, lwl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, lwl_g_predict, amp_g, l_g)
    V11 = V11_f + V11_g
    V11[np.diag_indices_from(V11)] += 1e-8

    V12_f = np.empty((M, N), dtype=np.float64)
    V12_g = np.empty((M, N), dtype=np.float64)
    matrix_functions.fill_V12_f(V12_f, lwl_f_predict, lwl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, lwl_g_predict, lwl_g, amp_g, l_g)
    V12 = V12_f + V12_g

    V22_f = np.empty((N, N), dtype=np.float64)
    V22_g = np.empty((N, N), dtype=np.float64)
    matrix_functions.fill_V11_f(V22_f, lwl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V22_g, lwl_g, amp_g, l_g)
    V22 = V22_f + V22_g
    V22[np.diag_indices_from(V22)] += sigma_fg**2

    factor, flag = cho_factor(V22)

    mu = mu_fg + np.dot(V12, cho_solve((factor, flag), (fl_fg - 1.0)))
    Sigma = V11 - np.dot(V12, cho_solve((factor, flag), V12.T))

    return mu, Sigma


def predict_f_g_h(lwl_f, lwl_g, lwl_h, fl_fgh, sigma_fgh,
                  lwl_f_predict, lwl_g_predict, lwl_h_predict,
                  mu_f, mu_g, mu_h,
                  amp_f, l_f, amp_g, l_g, amp_h, l_h):
    '''
    Predict the individual components of a triple-lined system (ST3).

    Args:
        lwl_f, lwl_g, lwl_h: log-wavelengths for each component.
        fl_fgh (1D np.array): observed combined flux.
        sigma_fgh (1D np.array): flux uncertainties.
        lwl_f_predict, lwl_g_predict, lwl_h_predict: prediction log-wavelengths.
        mu_f, mu_g, mu_h (float): GP means for each component.
        amp_f, l_f, amp_g, l_g, amp_h, l_h (float): GP hyperparameters.

    Returns:
        (mu, Sigma): predictive mean (concatenated) and covariance.
    '''
    assert len(lwl_f) == len(lwl_g) == len(lwl_h), \
        "Input wavelengths must be the same length."
    n_pix = len(lwl_f)

    assert len(lwl_f_predict) == len(lwl_g_predict) == len(lwl_h_predict), \
        "Prediction wavelengths must be the same length."
    n_pix_predict = len(lwl_f_predict)

    mu_cat = np.hstack([
        mu_f * np.ones(n_pix_predict),
        mu_g * np.ones(n_pix_predict),
        mu_h * np.ones(n_pix_predict),
    ])

    V11_f = np.empty((n_pix, n_pix), dtype=np.float64)
    V11_g = np.empty((n_pix, n_pix), dtype=np.float64)
    V11_h = np.empty((n_pix, n_pix), dtype=np.float64)
    matrix_functions.fill_V11_f(V11_f, lwl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, lwl_g, amp_g, l_g)
    matrix_functions.fill_V11_f(V11_h, lwl_h, amp_h, l_h)

    B = V11_f + V11_g + V11_h
    B[np.diag_indices_from(B)] += sigma_fgh**2
    factor, flag = cho_factor(B)

    V11_f_pred = np.empty((n_pix_predict, n_pix_predict), dtype=np.float64)
    V11_g_pred = np.empty((n_pix_predict, n_pix_predict), dtype=np.float64)
    V11_h_pred = np.empty((n_pix_predict, n_pix_predict), dtype=np.float64)
    matrix_functions.fill_V11_f(V11_f_pred, lwl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g_pred, lwl_g_predict, amp_g, l_g)
    matrix_functions.fill_V11_f(V11_h_pred, lwl_h_predict, amp_h, l_h)

    zeros = np.zeros((n_pix_predict, n_pix_predict))
    A = np.vstack((
        np.hstack([V11_f_pred, zeros, zeros]),
        np.hstack([zeros, V11_g_pred, zeros]),
        np.hstack([zeros, zeros, V11_h_pred]),
    ))

    V12_f = np.empty((n_pix_predict, n_pix), dtype=np.float64)
    V12_g = np.empty((n_pix_predict, n_pix), dtype=np.float64)
    V12_h = np.empty((n_pix_predict, n_pix), dtype=np.float64)
    matrix_functions.fill_V12_f(V12_f, lwl_f_predict, lwl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, lwl_g_predict, lwl_g, amp_g, l_g)
    matrix_functions.fill_V12_f(V12_h, lwl_h_predict, lwl_h, amp_h, l_h)

    C_mat = np.vstack((V12_f, V12_g, V12_h))

    mu = mu_cat + np.dot(C_mat, cho_solve((factor, flag), fl_fgh - 1.0))
    Sigma = A - np.dot(C_mat, cho_solve((factor, flag), C_mat.T))

    return mu, Sigma


def predict_f_g_h_sum(lwl_f, lwl_g, lwl_h, fl_fgh, sigma_fgh,
                      lwl_f_predict, lwl_g_predict, lwl_h_predict,
                      mu_fgh, amp_f, l_f, amp_g, l_g, amp_h, l_h):
    '''
    Predict the *sum* of three component spectra using the GP model (ST3).

    Returns:
        (mu, Sigma): predictive mean and covariance for the sum.
    '''
    assert len(lwl_f) == len(lwl_g) == len(lwl_h), \
        "Input wavelengths must be the same length."

    M = len(lwl_f_predict)
    N = len(lwl_f)

    V11_f = np.empty((M, M), dtype=np.float64)
    V11_g = np.empty((M, M), dtype=np.float64)
    V11_h = np.empty((M, M), dtype=np.float64)
    matrix_functions.fill_V11_f(V11_f, lwl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, lwl_g_predict, amp_g, l_g)
    matrix_functions.fill_V11_f(V11_h, lwl_h_predict, amp_h, l_h)
    V11 = V11_f + V11_g + V11_h

    V12_f = np.empty((M, N), dtype=np.float64)
    V12_g = np.empty((M, N), dtype=np.float64)
    V12_h = np.empty((M, N), dtype=np.float64)
    matrix_functions.fill_V12_f(V12_f, lwl_f_predict, lwl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, lwl_g_predict, lwl_g, amp_g, l_g)
    matrix_functions.fill_V12_f(V12_h, lwl_h_predict, lwl_h, amp_h, l_h)
    V12 = V12_f + V12_g + V12_h

    V22_f = np.empty((N, N), dtype=np.float64)
    V22_g = np.empty((N, N), dtype=np.float64)
    V22_h = np.empty((N, N), dtype=np.float64)
    matrix_functions.fill_V11_f(V22_f, lwl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V22_g, lwl_g, amp_g, l_g)
    matrix_functions.fill_V11_f(V22_h, lwl_h, amp_h, l_h)
    V22 = V22_f + V22_g + V22_h
    V22[np.diag_indices_from(V22)] += sigma_fgh**2

    factor, flag = cho_factor(V22)

    mu = mu_fgh + np.dot(V12.T, cho_solve((factor, flag), (fl_fgh - mu_fgh)))
    Sigma = V11 - np.dot(V12, cho_solve((factor, flag), V12.T))

    return mu, Sigma


# ---------------------------------------------------------------------------
# Log-likelihood functions (core — used by the samplers)
# ---------------------------------------------------------------------------

def lnlike_f_g(V11, wl_f, wl_g, fl, sigma, amp_f, l_f, amp_g, l_g, mu_GP=1.):
    '''
    Log-likelihood for a double-lined SB2 system using the Cython covariance
    matrix routine.

    Args:
        V11 (2D np.array): pre-allocated covariance matrix buffer.
        wl_f (1D np.array): Doppler-shifted log-wavelengths for the primary.
        wl_g (1D np.array): Doppler-shifted log-wavelengths for the secondary.
        fl (1D np.array): observed combined flux.
        sigma (1D np.array): flux uncertainties.
        amp_f (float): GP amplitude for the primary.
        l_f (float): GP length scale for the primary [km/s].
        amp_g (float): GP amplitude for the secondary.
        l_g (float): GP length scale for the secondary [km/s].
        mu_GP (float): GP mean (default 1.0 for normalized spectra).

    Returns:
        float: the log-likelihood value.
    '''
    if amp_f < 0.0 or l_f < 0.0 or amp_g < 0.0 or l_g < 0.0:
        return -np.inf

    matrix_functions.fill_V11_f_g(V11, wl_f, wl_g, amp_f, l_f, amp_g, l_g)
    V11[np.diag_indices_from(V11)] += sigma**2

    try:
        factor, flag = cho_factor(V11, overwrite_a=True, lower=False,
                                  check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf

    logdet = np.sum(2 * np.log(np.diag(factor)))
    return -0.5 * (
        np.dot((fl - mu_GP).T, cho_solve((factor, flag), (fl - mu_GP)))
        + logdet
    )


def lnlike_f_g_h(V11, wl_f, wl_g, wl_h, fl, sigma,
                 amp_f, l_f, amp_g, l_g, amp_h, l_h, mu_GP=1.):
    '''
    Log-likelihood for a triple-lined ST3 system using the Cython covariance
    matrix routine.

    Args:
        V11 (2D np.array): pre-allocated covariance matrix buffer.
        wl_f, wl_g, wl_h (1D np.array): Doppler-shifted log-wavelengths.
        fl (1D np.array): observed combined flux.
        sigma (1D np.array): flux uncertainties.
        amp_f, l_f, amp_g, l_g, amp_h, l_h (float): GP hyperparameters.
        mu_GP (float): GP mean.

    Returns:
        float: the log-likelihood value.
    '''
    if (amp_f < 0.0 or l_f < 0.0 or amp_g < 0.0 or l_g < 0.0
            or amp_h < 0.0 or l_h < 0.0):
        return -np.inf

    matrix_functions.fill_V11_f_g_h(V11, wl_f, wl_g, wl_h,
                                     amp_f, l_f, amp_g, l_g, amp_h, l_h)
    V11[np.diag_indices_from(V11)] += sigma**2

    try:
        factor, flag = cho_factor(V11)
    except np.linalg.LinAlgError:
        return -np.inf

    logdet = np.sum(2 * np.log(np.diag(factor)))
    return -0.5 * (
        np.dot((fl - mu_GP).T, cho_solve((factor, flag), (fl - mu_GP)))
        + logdet
    )


# Hook for ST2 (double-lined tertiary): same covariance structure as SB2
lnlike_f_g_ST2 = lnlike_f_g


# ---------------------------------------------------------------------------
# scikit-learn GP alternative for SB2
# ---------------------------------------------------------------------------

def lnlike_f_g_sklearn(lwl_f, lwl_g, fl, sigma, amp_f, l_f, amp_g, l_g,
                       mu_GP=1.):
    '''
    Log-likelihood for a double-lined SB2 system using scikit-learn's GP.

    This is an alternative to :func:`lnlike_f_g` that uses the
    ``sklearn.gaussian_process`` framework with an ``RBF`` kernel instead of
    the custom Cython matrix routines.  The mathematics are identical; the
    length-scale parameter ``l_f`` / ``l_g`` is in km/s and is interpreted as
    a velocity separation in log-wavelength space.

    .. note::
        This function is provided primarily for validation and debugging.
        It constructs the full ``(N, N)`` covariance matrix in pure NumPy,
        which is significantly slower than the Cython-based :func:`lnlike_f_g`
        for typical spectrum sizes.  **Do not use this function in production
        MCMC runs.**

    Args:
        lwl_f (1D np.array): Doppler-shifted log-wavelengths for the primary.
        lwl_g (1D np.array): Doppler-shifted log-wavelengths for secondary.
        fl (1D np.array): observed combined flux.
        sigma (1D np.array): flux uncertainties.
        amp_f, l_f, amp_g, l_g (float): GP hyperparameters.
        mu_GP (float): GP mean (default 1.0).

    Returns:
        float: the log-likelihood value.
    '''
    if amp_f < 0.0 or l_f < 0.0 or amp_g < 0.0 or l_g < 0.0:
        return -np.inf

    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    c_kms = C.c_kms

    n = len(lwl_f)
    # Build the covariance matrix explicitly using RBF kernels in log-wl space
    # k_f(i,j) = amp_f^2 * exp(-0.5 * ((lwl_f[i]-lwl_f[j])*c_kms)^2 / l_f^2)
    diff_f = (lwl_f[:, np.newaxis] - lwl_f[np.newaxis, :]) * c_kms
    diff_g = (lwl_g[:, np.newaxis] - lwl_g[np.newaxis, :]) * c_kms

    K_f = amp_f**2 * np.exp(-0.5 * diff_f**2 / l_f**2)
    K_g = amp_g**2 * np.exp(-0.5 * diff_g**2 / l_g**2)
    K = K_f + K_g + np.diag(sigma**2)

    try:
        factor, flag = cho_factor(K, overwrite_a=False, lower=False,
                                  check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf

    r = fl - mu_GP
    logdet = np.sum(2 * np.log(np.diag(factor)))
    return -0.5 * (np.dot(r, cho_solve((factor, flag), r)) + logdet)


# Dispatch dictionaries used by the samplers
lnlike = {
    "SB2": lnlike_f_g,
    "ST2": lnlike_f_g_ST2,  # hook: same structure as SB2
    "ST3": lnlike_f_g_h,
}


# ---------------------------------------------------------------------------
# Calibration utilities
# ---------------------------------------------------------------------------

def optimize_calibration(lwl0, lwl1, lwl_cal, fl_cal, fl_fixed,
                         A, B, C_matrix, order=1, mu_GP=1.0):
    '''
    Determine the Chebyshev calibration coefficients for a single epoch.

    Calibrates ``fl_cal`` with respect to the reference flux vector ``fl_fixed``
    by minimizing the GP-based least-squares residual.

    Args:
        lwl0 (float): left boundary for the Chebyshev domain.
        lwl1 (float): right boundary for the Chebyshev domain.
        lwl_cal (1D np.array): log-wavelengths of the epoch to calibrate.
        fl_cal (1D np.array): fluxes of the epoch to calibrate.
        fl_fixed (1D np.array): reference (combined remaining) flux vector.
        A (2D np.array): covariance of ``lwl_cal`` with sigma on diagonal.
        B (2D np.array): covariance of ``lwl_fixed`` with sigma on diagonal.
        C_matrix (2D np.array): cross-covariance between cal and fixed epochs.
        order (int): Chebyshev polynomial order (default 1 = linear).
        mu_GP (float): GP mean.

    Returns:
        (fl_cor, X): calibrated flux array and calibration coefficients.
    '''
    T = []
    for i in range(order + 1):
        coeff = [0] * i + [1]
        Chtemp = Ch(coeff, domain=[lwl0, lwl1])
        T.append(Chtemp(lwl_cal))
    T = np.array(T)
    D = fl_cal[:, np.newaxis] * T.T

    try:
        B_cho = cho_factor(B)
    except np.linalg.LinAlgError:
        print("Failed to solve matrix inverse. Calibration not valid.")
        raise

    fl_prime = mu_GP + np.dot(C_matrix, cho_solve(B_cho, (fl_fixed.flatten() - mu_GP)))
    C_prime = A - np.dot(C_matrix, cho_solve(B_cho, C_matrix.T))

    CP_cho = cho_factor(C_prime)
    left = np.dot(D.T, cho_solve(CP_cho, D))
    right = np.dot(D.T, cho_solve(CP_cho, fl_prime))
    left_cho = cho_factor(left)
    X = cho_solve(left_cho, right)
    fl_cor = np.dot(D, X)
    return fl_cor, X
