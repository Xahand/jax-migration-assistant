import numpy as np

from original_numpy import normalize_and_score as original
from migrated_jax import normalize_and_score as migrated


def test_normalize_and_score_matches_numpy_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 3)).astype(np.float32)
    weights = rng.normal(size=(3, 2)).astype(np.float32)
    bias = rng.normal(size=(2,)).astype(np.float32)

    expected = original(x, weights, bias)
    actual = np.asarray(migrated(x, weights, bias))

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
