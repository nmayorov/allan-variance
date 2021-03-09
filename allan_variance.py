# Author: Nikolay Mayorov <nikolay.mayorov@zoho.com>

from __future__ import division
import numpy as np
import pandas as pd
from scipy.optimize import nnls


ALLOWED_INPUT_TYPES = ['mean', 'increment', 'integral']


def _compute_cluster_sizes(n_samples, dt, tau_min, tau_max, n_clusters):
    if tau_min is None:
        min_size = 1
    else:
        min_size = int(tau_min / dt)

    if tau_max is None:
        max_size = n_samples // 10
    else:
        max_size = int(tau_max / dt)

    result = np.logspace(np.log2(min_size), np.log2(max_size),
                         num=n_clusters, base=2)

    return np.unique(np.round(result)).astype(int)


def allan_variance(x, dt=1, tau_min=None, tau_max=None,
                   n_clusters=100, input_type='mean'):
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
    tau_min : float or None, optional
        Minimum averaging time to use. If None (default), it is assigned equal
        to `dt`.
    tau_max : float or None, optional
        Maximum averaging time to use. If None (default), it is chosen
        automatically such that to averaging is done over 10 *independent*
        clusters.
    n_clusters : int, optional
        Number of clusters to compute Allan variance for. The averaging times
        will be spread approximately uniform in a log scale. Default is 100.
    input_type : {'mean', 'increment', 'integral'}, optional
        How to interpret the input data:

            - 'mean': `x` is assumed to contain mean values of y over
              successive time intervals.
            - 'increment': `x` is assumed to contain integral increments over
              successive time intervals.
            - 'integral': `x` is assumed to contain cumulative integral of
              y from the time start.

        Note that 'mean' can be used when passing filtered and sampled value
        of y. But in this case generally the filtering process affects
        properties of the underlying signal y and "effective" Allan variance
        is computed, i.e. you can use obtained parameters when working with
        the sampled signal.

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
    """
    if input_type not in ALLOWED_INPUT_TYPES:
        raise ValueError("`input_type` must be one of {}."
                         .format(ALLOWED_INPUT_TYPES))

    x = np.asarray(x, dtype=float)
    if input_type == 'integral':
        X = x
    else:
        X = np.cumsum(x, axis=0)

    cluster_sizes = _compute_cluster_sizes(len(x), dt, tau_min, tau_max,
                                           n_clusters)

    avar = np.empty(cluster_sizes.shape + X.shape[1:])
    for i, k in enumerate(cluster_sizes):
        c = X[2*k:] - 2 * X[k:-k] + X[:-2*k]
        avar[i] = np.mean(c**2, axis=0) / k / k

    if input_type == 'mean':
        avar *= 0.5
    else:
        avar *= 0.5 / dt**2

    return cluster_sizes * dt, avar


def params_from_avar(tau, avar, effects=None, sensor_names=None):
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
    avar : ndarray, shape (n,) or (n, m)
        Values of Allan variance corresponding to `tau`.
    effects : list or None, optional
        Which effects to estimate. Allowed effects are 'quantization', 'white',
        'flicker', 'walk', 'ramp'. If None (default), estimate all of the
        mentioned above effects.
    sensor_names : list or None, optional
        How to name sensors in the output. If None (default), use integer
        values as names.

    Returns
    -------
    params : pandas DataFrame or Series
        Estimated parameters.
    prediction : ndarray, shape (n,) or (n, m)
        Predicted values of Allan variance using the estimated parameters.
    """
    ALLOWED_EFFECTS = ['quantization', 'white', 'flicker', 'walk', 'ramp']

    avar = np.asarray(avar)
    single_series = avar.ndim == 1
    if single_series:
        avar = avar[:, None]

    if effects is None:
        effects = ALLOWED_EFFECTS
    elif not set(effects) <= set(ALLOWED_EFFECTS):
        raise ValueError("Unknown effects are passed.")

    n = len(tau)

    A = np.empty((n, 5))
    A[:, 0] = 3 / tau**2
    A[:, 1] = 1 / tau
    A[:, 2] = 2 * np.log(2) / np.pi
    A[:, 3] = tau / 3
    A[:, 4] = tau**2 / 2
    mask = ['quantization' in effects,
            'white' in effects,
            'flicker' in effects,
            'walk' in effects,
            'ramp' in effects]

    A = A[:, mask]
    effects = np.asarray(ALLOWED_EFFECTS)[mask]

    params = []
    prediction = []

    for column in range(avar.shape[1]):
        avar_single = avar[:, column]
        A_scaled = A / avar_single[:, None]
        x = nnls(A_scaled, np.ones(n))[0]
        prediction.append(A_scaled.dot(x) * avar_single)
        params.append(np.sqrt(x))

    params = np.asarray(params)
    prediction = np.asarray(prediction).T

    params = pd.DataFrame(params, index=sensor_names, columns=effects)

    if single_series:
        params = params.iloc[0]
        prediction = prediction[:, 0]

    return params, prediction
