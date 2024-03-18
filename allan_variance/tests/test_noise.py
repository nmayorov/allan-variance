import numpy as np
from numpy.testing import assert_allclose
from allan_variance import generate_noise


def test_generate_noise_smoke():
    w = generate_noise(0.0, 100000, dt=0.01, scale=0.5, rng=0)
    assert_allclose(np.std(w), 5.0, rtol=1e-2)

    b = generate_noise(2.0, (10000, 1000), dt=2.0, scale=0.3, rng=0)
    std = np.std(b, axis=1)
    assert_allclose(std, 0.3 * (2.0 * np.arange(1, 10001)) ** 0.5, rtol=1e-1)
