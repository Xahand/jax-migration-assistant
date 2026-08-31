# Migration Guide

JAX migration is a program-understanding task. Start by finding the computation
that matters, then move it carefully.

## 1. Inspect The Original Code

Identify:

- framework boundaries: NumPy, PyTorch, Python loops, custom kernels;
- inputs and outputs: shapes, dtypes, units, randomness, side effects;
- state: model parameters, optimizer state, environment state, caches;
- control flow: batch loops, time loops, conditionals, early exits;
- existing tests, examples, checkpoints, or expected outputs.

Do not edit first. Write down what the original program computes and where JAX
could plausibly help.

## 2. Choose The Migration Boundary

Good JAX candidates usually include:

- pure numerical kernels;
- differentiable loss functions;
- model forward passes;
- training steps with explicit parameters and optimizer state;
- batched computation suitable for `vmap`;
- fixed-length recurrent computation suitable for `lax.scan`.

Weak candidates usually include:

- plotting;
- file I/O;
- pandas preprocessing;
- logging and experiment tracking;
- configuration;
- Python orchestration;
- dataloaders that only feed compiled computation.

Keep non-numerical code in ordinary Python unless moving it has a measurable
benefit.

## 3. Migrate For Correctness First

Create the simplest JAX version that preserves behavior:

- use `jax.numpy` for array computations;
- replace mutation with functional updates;
- pass state explicitly;
- pass PRNG keys explicitly and split before independent random operations;
- represent nested parameters or environment state as PyTrees;
- keep shapes and dtypes visible.

Avoid adding `jit`, `vmap`, or `scan` in the first pass unless the original
boundary is already simple and well-tested.

## 4. Verify Before Optimizing

Compare the original and migrated implementations on deterministic inputs.
Check shapes, dtypes, values, gradients when relevant, and state transitions.
Use tolerances appropriate to the dtype and backend.

## 5. Apply JAX Transformations Deliberately

Use JAX transformations when they match the computation:

- `jit` for stable-shape numerical functions called repeatedly;
- `vmap` for independent batch axes;
- `lax.scan` for fixed-length recurrences with carried state;
- `grad` or `value_and_grad` for scalar differentiable objectives;
- `jax.tree` utilities for structured parameters and state.

After each transformation, run equivalence tests again.

## 6. Benchmark Only After Equivalence

JAX performance depends on compilation, shape stability, host/device transfers,
backend, dtype, and asynchronous dispatch. Do not claim speedups from code shape
alone. Benchmark with warmup, block on results, and compare the original and JAX
implementations under realistic workload sizes.

## References

- JAX documentation: <https://docs.jax.dev/>
- JAX random numbers: <https://docs.jax.dev/en/latest/jax.random.html>
- JAX control flow: <https://docs.jax.dev/en/latest/201/control-flow.html>
- JAX PyTrees: <https://docs.jax.dev/en/latest/pytrees.html>
- Phan et al., "Learning Bug Context for PyTorch-to-JAX Translation with LLMs":
  <https://arxiv.org/abs/2510.09898>
