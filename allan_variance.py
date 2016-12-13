# Author: Nikolay Mayorov <nikolay.mayorov@zoho.com>

from __future__ import division
from collections import OrderedDict
import numpy as np
from scipy.optimize import nnls


def allan_variance(x, dt=1, min_cluster_size=1, min_cluster_count='auto',
                   n_clusters=100, input_type="increment"):
    """Compute Allan variance (AVAR).

    Consider an underlying measurement y(t). Our sensors output integrals of
    y(t) over successive time intervals of length dt. These measurements
    x(k * dt) form the input to this function.

    Allan variance is defined for different averaging times tau = m * dt as
    follows::

        AVAR(tau) = 1/2 * <(Y(k + m) - Y(k))>,

    where Y(j) is the time average value of y(t) over [k * dt, (k + m) * dt]
    (call it a cluster), and < ... > means averaging over different clusters.
    If we define X(j) being an integral of x(s) from 0 to dt * j,
    we can rewrite the AVAR as  follows::

        AVAR(tau) = 1/(2 * tau**2) * <X(k + 2 * m) - 2 * X(k + m) + X(k)>

    We implement < ... > by averaging over different clusters of a given sample
    with overlapping, and X(j) is readily available from x.

    Parameters
    ----------
    x : ndarray, shape (n, ...)
        Integrating sensor readings, i. e. its cumulative sum gives an
        integral of a signal. Assumed to vary along the 0-th axis.
    dt : float, optional
        Sampling period. Default is 1.
    min_cluster_size : int, optional
        Minimum size of a cluster to use. Determines a lower bound on the
        averaging time as ``dt * min_cluster_size``. Default is 1.
    min_cluster_count : int or 'auto', optional
        Minimum number of clusters required to compute the average. Determines
        an upper bound of the averaging time as
        ``dt * (n - min_cluster_count) // 2``. If 'auto' (default) it is taken
        to be ``min(1000, n - 2)``
    n_clusters : int, optional
        Number of clusters to compute Allan variance for. The averaging times
        will be spread approximately uniform in a log scale. Default is 100.
    input_type : 'increment' or 'mean', optional
        How to interpret the input data. If 'increment' (default), then
        `x` is assumed to contain integral increments over successive time
        intervals (as described above). If 'mean', then `x` is assumed to
        contain mean values of y over successive time intervals, i.e.
        increments divided by `dt`.

    Returns
    -------
    tau : ndarray
        Averaging times for which Allan variance was computed, 1-d array.
    avar : ndarray
        Values of AVAR. The 0-th dimension is the same as for `tau`. The
        trailing dimensions match ones for `x`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Allan_variance
    .. [2] IEEE Standart 952-1997 "IEEE Standard Specification Format Guide and
        Test Procedure for Single-Axis Interferometric Fiber Optic Gyros", 1998
    """
    if input_type not in ('increment', 'mean'):
        raise ValueError("`input_type` must be either 'increment' or 'mean'.")

    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    X = np.cumsum(x, axis=0)

    if min_cluster_count == 'auto':
        min_cluster_count = min(1000, n - 2)

    log_min = np.log2(min_cluster_size)
    log_max = np.log2((n - min_cluster_count) // 2)

    cluster_sizes = np.logspace(log_min, log_max, n_clusters, base=2)
    cluster_sizes = np.unique(np.round(cluster_sizes)).astype(np.int64)

    avar = np.empty(cluster_sizes.shape + X.shape[1:])
    for i, k in enumerate(cluster_sizes):
        c = X[2*k:] - 2 * X[k:-k] + X[:-2*k]
        avar[i] = np.mean(c**2, axis=0) / k / k

    if input_type == 'increment':
        avar *= 0.5 / dt**2
    elif input_type == 'mean':
        avar *= 0.5

    return cluster_sizes * dt, avar


def params_from_avar(tau, avar, output_type='array', empiric_mode=False):
    """Estimate noise parameters from Allan variance.

    The parameters being estimated are typical for inertial sensors:
    quantization noise, additive white noise, flicker noise (long term bias
    instability), random walk and linear ramp (this is a deterministic effect).

    The parameters are estimated using linear least squares with weights
    inversely proportional to the values of Allan variance. That is the sum of
    relative error is minimized. This approach is approximately equivalent of
    doing estimation in the log-log scale.

    Parameters
    ----------
    tau : ndarray, shape (n,)
        Values of averaging time.
    avar : ndarray, shape (n,)
        Values of Allan variance corresponding to `tau`.
    output_type : 'array' or 'dict', optional
        How to return the computed parameters. If 'array' (default), then
        ndarray is returned. If 'dict', then OrderedDict is returned. See
        Returns section for more details.
    empiric_mode : bool (Default is False)
        If empiric mode is enabled (True), then if `flicker` is 0, then
        estimate it with `min(avar)`, but consider the right part of the
        predicted curve. Should be used carefully.

    Returns
    -------
    params : ndarray, shape (5,) or OrderedDict
        Either ndarray with estimated parameters, ordered as quantization,
        additive white, flicker, random walk, linear ramp. Or OrderedDict with
        the keys 'quantization', 'white', 'flicker', 'walk', 'ramp' and the
        estimated parameters as the values.
    prediction : ndarray, shape (n,)
        Predicted values of Allan variance using the estimated parameters.

    References
    ----------
    .. [1] IEEE Standart 952-1997 "IEEE Standard Specification Format Guide and
        Test Procedure for Single-Axis Interferometric Fiber Optic Gyros", 1998
    """
    if output_type not in ('array', 'dict'):
        raise ValueError("`output_type` must be either 'array' or 'dict'.")

    n = tau.shape[0]
    A = np.empty((n, 5))
    A[:, 0] = 3 / tau**2
    A[:, 1] = 1 / tau
    A[:, 2] = 2 * np.log(2) / np.pi
    A[:, 3] = tau / 3
    A[:, 4] = tau**2 / 2
    A /= avar[:, np.newaxis]
    b = np.ones(n)

    x = nnls(A, b)[0]
    # If flicker is not estimated, then let's estimate it as min of avar.
    # Consider the right part of the curve (either rate random walk or rate
    # ramp needs to be estimated)
    if empiric_mode and x[2] == 0 and (x[3] != 0 or x[4] != 0):
        x[2] = np.min(avar)**2.
    prediction = A.dot(x) * avar
    params = np.sqrt(x)

    if output_type == 'dict':
        params = OrderedDict(
            zip(('quantization', 'white', 'flicker', 'walk', 'ramp'), params))

    return params, prediction
