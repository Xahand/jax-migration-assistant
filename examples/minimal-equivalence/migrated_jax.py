import jax.numpy as jnp


def normalize_and_score(x, weights, bias):
    mean = jnp.mean(x, axis=0)
    std = jnp.std(x, axis=0) + 1e-6
    normalized = (x - mean) / std
    return normalized @ weights + bias
