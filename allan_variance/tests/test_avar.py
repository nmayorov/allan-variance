import numpy as np
import allan_variance


def test_generate_avar_smoke():
    x = np.ones(1000)
    tau, avar = allan_variance.compute_avar(x)
    assert np.all(avar == 0.0)

    x = np.arange(1000)
    tau, avar = allan_variance.compute_avar(x)
    assert np.all(avar == 0.5 * tau ** 2)


def test_estimate_parameters_smoke():
    rng = np.random.RandomState(0)
    x = rng.randn(1000)
    tau, avar = allan_variance.compute_avar(x)
    parameters, _ = allan_variance.estimate_parameters(tau, avar)
    assert np.abs(parameters.white - 1.0) < 0.1

    effects = ['quantization', 'white', 'flicker', 'walk', 'ramp']
    for exclude in ['quantization', 'flicker', 'walk', 'ramp']:
        effects.remove(exclude)
        parameters, _ = allan_variance.estimate_parameters(tau, avar, effects)
        assert np.abs(parameters.white - 1.0) < 0.1
