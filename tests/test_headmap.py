"""Tests for the exploratory head filter map: stats on synthetic curves, stability, and the noise correction."""
import numpy as np

from dialedin import headmap as H

GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def test_head_stats_on_known_curves():
    x = np.asarray(GRID, dtype=float)
    gain = np.ones((len(GRID), 1, 3))
    gain[:, 0, 0] = x ** 0.2                                   # pure power law: tilt 0.2
    gain[:, 0, 1] = 1 + 9 / (1 + (64 / x) ** 4)                # a shelf rising around N = 64
    gain[:, 0, 2] = 1.0                                        # flat: no corner
    snr = np.full_like(gain, 10.0)
    tilt, corner = H.head_stats(gain, snr, GRID, 3.0)
    assert abs(tilt[0, 0] - 0.2) < 1e-9 and abs(tilt[0, 2]) < 1e-12 and np.isnan(corner[0, 2])
    assert 32 <= corner[0, 1] <= 128
    snr[3, 0, 0] = 2.0                                          # one unresolved period drops the head
    tilt, _ = H.head_stats(gain, snr, GRID, 3.0)
    assert np.isnan(tilt[0, 0]) and not np.isnan(tilt[0, 1])


def test_pooled_noise_correction_and_stability():
    sig = np.full((4, 2, 1, 1), 5.0); noi = np.full((4, 2, 1, 1), 1.0)
    g, snr = H.pooled(sig, noi, [0, 1, 2, 3])
    assert np.allclose(g, 2.0) and np.allclose(snr, np.sqrt(5.0))
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 8)); b = a + 0.01 * rng.normal(size=(3, 8)); b[0, 0] = np.nan
    st = H.stability(a, b)
    assert st["n"] == 23 and st["r"] > 0.99 and st["r_within_layer"] > 0.99
