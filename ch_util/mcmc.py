"""Tools for fitting complex gain models with MCMC

This module fits a phase-only model to the complex gain of a single input
using the `emcee` ensemble sampler.  The model has unit amplitude and
consists of a linear delay plus a constant phase offset,

.. math:: g(\\nu) = \\exp\\left(i \\left(2 \\pi \\nu \\tau + \\phi_{0}\\right)\\right)

Note that the fit is performed on the unflagged channels only, and that
the phase residuals are wrapped into [-pi, pi) before the chi-squared is
evaluated.

The main routine is:

- :py:meth:`run_mcmc_gain_fit_simple`

The model and the terms of the posterior can also be evaluated directly:

- :py:meth:`generate_model_simple`
- :py:meth:`log_prior_simple`
- :py:meth:`log_likelihood_simple`
- :py:meth:`log_probability_simple`
"""

import emcee
import numpy as np


def generate_model_simple(freqs_array, delay, const):
    """Generate a unit-amplitude complex phasor.

    Parameters
    ----------
    freqs_array : np.ndarray[nfreq,]
        Frequency in MHz.
    delay : float
        Linear-phase delay in micro-seconds.
    const : float
        Constant phase offset in radians.

    Returns
    -------
    model : np.ndarray[nfreq,]
        The complex phasor exp(i * (2 pi * freq * delay + const)).
    """
    phase = 2 * np.pi * freqs_array * delay + const

    return np.exp(1j * phase)


def log_prior_simple(theta):
    """Evaluate the hard-box log-prior for the phase-only model.

    Parameters
    ----------
    theta : np.ndarray[2,]
        The model parameters [delay, const].

    Returns
    -------
    lp : float
        Zero if the parameters are within the bounds of the prior,
        and negative infinity otherwise.
    """
    delay, const = theta

    if not (-0.005 < delay < 0.005):
        return -np.inf

    if not (-np.pi < const < np.pi):
        return -np.inf

    return 0.0


def log_likelihood_simple(theta, freqs_valid, data_valid):
    """Evaluate the phase-only log-likelihood.

    There is no amplitude term.  The phase residuals are wrapped into
    [-pi, pi) so that the chi-squared is not sensitive to the branch
    cut of `np.angle`.

    Parameters
    ----------
    theta : np.ndarray[2,]
        The model parameters [delay, const].
    freqs_valid : np.ndarray[nvalid,]
        Frequency of the unflagged channels in MHz.
    data_valid : np.ndarray[nvalid,]
        Complex gain of the unflagged channels.

    Returns
    -------
    ll : float
        The log-likelihood.
    """
    model = generate_model_simple(freqs_valid, *theta)

    sig_phase = 0.05

    diff_phase = np.angle(data_valid / model)
    diff_phase = (diff_phase + np.pi) % (2 * np.pi) - np.pi

    return -0.5 * np.sum((diff_phase / sig_phase) ** 2)


def log_probability_simple(theta, freqs_valid, data_valid):
    """Evaluate the log-posterior for the phase-only model.

    Parameters
    ----------
    theta : np.ndarray[2,]
        The model parameters [delay, const].
    freqs_valid : np.ndarray[nvalid,]
        Frequency of the unflagged channels in MHz.
    data_valid : np.ndarray[nvalid,]
        Complex gain of the unflagged channels.

    Returns
    -------
    lp : float
        The sum of the log-prior and the log-likelihood.
    """
    lp = log_prior_simple(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood_simple(theta, freqs_valid, data_valid)


def run_mcmc_gain_fit_simple(
    gain_inp, freqs, good, nsteps=1000, nwalkers=32, burnin_frac=0.25
):
    """Fit a phase-only (delay + constant offset) model with emcee.

    The model has unit amplitude and does not describe any reflection
    structure, which makes it a fast first-pass estimate of the linear
    delay and the phase offset.

    Parameters
    ----------
    gain_inp : np.ndarray[nfreq,]
        Complex gain for a single input.
    freqs : np.ndarray[nfreq,]
        Frequency in MHz.
    good : np.ndarray[nfreq,] of bool
        True indicates an unflagged channel.  Only these channels
        are used in the fit.
    nsteps : int
        Total number of MCMC steps.
    nwalkers : int
        Number of emcee walkers.
    burnin_frac : float
        Fraction of the steps to discard as burn-in.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        best_fit_theta : np.ndarray[2,]
            The median posterior parameters [delay, const].
        flat_samples : np.ndarray[nsample, 2]
            The full chain, after the burn-in has been discarded.
        mcmc_model : np.ndarray[nfreq,]
            The best-fit model evaluated at all frequencies.
        acc_frac : float
            The mean acceptance fraction.
    """
    freqs_valid = freqs[good]
    data_valid = gain_inp[good]

    # Initialise the walkers.  The legacy global random state is seeded
    # so that the starting positions are reproducible.
    np.random.seed(42)  # noqa: NPY002
    pos = np.zeros((nwalkers, 2))
    pos[:, 0] = np.random.uniform(-0.005, 0.005, nwalkers)  # delay  # noqa: NPY002
    pos[:, 1] = np.random.uniform(0.01, 0.99, nwalkers)  # const  # noqa: NPY002

    burnin = int(burnin_frac * nsteps)

    sampler = emcee.EnsembleSampler(
        nwalkers, 2, log_probability_simple, args=(freqs_valid, data_valid)
    )
    sampler.run_mcmc(pos, nsteps, progress=False)

    acc_frac = float(np.mean(sampler.acceptance_fraction))
    flat_samples = sampler.get_chain(discard=burnin, flat=True)
    best_fit_theta = np.median(flat_samples, axis=0)
    mcmc_model = generate_model_simple(freqs, *best_fit_theta)

    return {
        "best_fit_theta": best_fit_theta,
        "flat_samples": flat_samples,
        "mcmc_model": mcmc_model,
        "acc_frac": acc_frac,
    }
