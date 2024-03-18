import numpy as np
import avar


def test_generate_avar_smoke():
    x = np.ones(1000)
    tau, av = avar.compute_avar(x)
    assert np.all(av == 0.0)

    x = np.arange(1000)
    tau, av = avar.compute_avar(x)
    assert np.all(av == 0.5 * tau ** 2)


def test_estimate_parameters_smoke():
    rng = np.random.RandomState(0)
    x = rng.randn(1000)
    tau, av = avar.compute_avar(x)
    parameters, _ = avar.estimate_parameters(tau, av)
    assert np.abs(parameters.white - 1.0) < 0.1

    effects = ['quantization', 'white', 'flicker', 'walk', 'ramp']
    for exclude in ['quantization', 'flicker', 'walk', 'ramp']:
        effects.remove(exclude)
        parameters, _ = avar.estimate_parameters(tau, av, effects)
        assert np.abs(parameters.white - 1.0) < 0.1
