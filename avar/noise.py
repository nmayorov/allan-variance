"""Noise generation."""

import numpy as np
from scipy.signal import fftconvolve
from scipy._lib._util import check_random_state


def generate_noise(alpha, shape, dt=1.0, scale=1.0, rng=None):
    """Generate Gaussian noise with a power law spectrum.

    The function generates a noise sequence with the following
    power spectral density (PSD) function::

        PSD(f) = magnitude**2 / (2 * np.pi * f)**alpha

    Where ``magnitude`` and ``alpha`` are function arguments.

    The noise is generated using a "fractional differencing" approach [1]_ which does
    digital convolution of a white noise sequence with an impulse response
    corresponding to the following transfer function::

        H(z) = 1 / (1 - z**-1)**(alpha/2)

    With ``alpha = 2`` this transfer function is rational and corresponds to a
    well known process with independent increments or Browninan motion.
    For other alphas (for example ``alpha = 1`` -- flicker noise) it is non-rational
    and has to be expanded into a Taylor series to implement a convolution.
    The size of expansion is equal to the size of the desired output.

    Parameters
    ----------
    alpha : float
        Exponent in the denominator of the power law.
    shape : array_like
        Required shape of the output. The time is assumed to vary along the 0-th
        dimension. For example, to generate 100 signals of length 1000 pass
        `shape` as (1000, 100).
    dt : float, optional
        Time period between samples in seconds. Default is 1.0.
    scale : float, optional
        Input unit white noise is multiplied by `scale`. Default is 1.0.
    rng : None, int or `numpy.random.RandomState`, optional
        Seed to create or already created RandomState. None (default) corresponds to
        nondeterministic seeding.

    Returns
    -------
    ndarray
        Generated noise with the given `shape`.

    References
    ----------
    .. [1] J. N. Kasdin "Discrete Simulation of Colored Noise and Stochastic Processes
           and 1/f^alpha Power Law Noise Generation"
    """
    rng = check_random_state(rng)
    shape = np.atleast_1d(shape)
    n_samples = shape[0]
    h = np.empty(n_samples)
    h[0] = 1
    for i in range(1, n_samples):
        h[i] = h[i - 1] * (0.5 * alpha + i - 1) / i
    h = h.reshape((-1,) + (1,) * (len(shape) - 1))
    w = rng.randn(*shape)
    return scale * dt ** (-0.5 * (1 - alpha)) * fftconvolve(h, w)[:n_samples]
