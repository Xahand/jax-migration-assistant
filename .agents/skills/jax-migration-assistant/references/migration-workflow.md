# Migration Workflow

Use this workflow for nontrivial migrations.

## 1. Inspect

Read the relevant code before editing. Identify the original framework, entry
points, tests, data shapes, state, randomness, and side effects.

Produce an internal inventory like:

```text
Frameworks: NumPy, PyTorch
Numerical kernels: ...
State: parameters, optimizer state, environment state
Randomness: torch.manual_seed, np.random, implicit sampling
Control flow: batch loop, time loop, early termination
Candidate JAX boundary: ...
Leave outside JAX: ...
```

## 2. Preserve A Reference

Keep the original implementation importable when possible. If code must move,
copy the original behavior into tests, fixtures, or a clearly named reference
module before replacing it.

## 3. Build The Straightforward JAX Version

First prefer clarity:

- use `jax.numpy` for array operations;
- pass parameters and state explicitly;
- replace mutation with returned values;
- use explicit PRNG keys;
- keep dataloading and orchestration in Python.

Avoid optimizing in the same edit unless the function is tiny and already has a
reference test.

## 4. Verify

Run or create equivalence tests. Compare shapes, dtypes, outputs, state updates,
and gradients when relevant. Use deterministic inputs and explicit tolerances.

## 5. Introduce Transformations

Only after equivalence:

- add `jit` to repeated, shape-stable numerical functions;
- use `vmap` for independent batch axes;
- use `lax.scan` for fixed-length recurrent loops;
- use `lax.cond` or `lax.while_loop` for compiled dynamic control flow;
- use `value_and_grad` for training steps that need both loss and gradients.

Run equivalence tests again after each transformation.

## 6. Benchmark When Relevant

If performance is part of the goal, benchmark with warmup, stable shapes,
realistic sizes, and `block_until_ready()` on JAX results. Report whether timing
includes compilation.
