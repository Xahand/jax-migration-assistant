# Migration Decisions

Do not migrate everything. Choose boundaries that make the program easier to
test and maintain.

## Strong Candidates

- Pure numerical functions.
- Large array operations currently expressed with NumPy or PyTorch.
- Differentiable loss functions and model forward passes.
- Training steps where parameters and optimizer state can be explicit.
- Independent computations over a leading batch axis.
- Fixed-length recurrent computations.
- Simulation transition functions with array state and stable shapes.

## Weak Candidates

- Plotting and visualization.
- Filesystem, network, and database I/O.
- pandas preprocessing and tabular cleaning.
- Logging, checkpoint bookkeeping, experiment tracking.
- CLI argument parsing and configuration.
- Object-heavy orchestration code.
- Dataloaders whose job is to feed arrays to the numerical core.

## Decision Questions

Ask:

- Is this part of the hot numerical computation?
- Is it pure or can its state be made explicit?
- Are shapes stable enough for compilation?
- Can we write a reference equivalence test?
- Will migrating this reduce complexity or enable useful transformations?
- Would ordinary Python be clearer?

If the answer is unclear, migrate a smaller boundary first.

## Common Mappings

```text
NumPy array code        -> jax.numpy at numerical kernels
PyTorch tensor ops      -> jax.numpy or JAX primitives
nn.Module forward pass  -> Flax, Equinox, or pure function
torch.optim             -> Optax or explicit optimizer update
implicit RNG            -> explicit jax.random keys
batch loop              -> vmap when iterations are independent
time loop               -> lax.scan when fixed-length with carried state
mutable state           -> explicit input/output state PyTree
```

Do not choose a neural-network library mechanically. Match the target project:
Flax for common module/train-state patterns, Equinox for lightweight PyTree
modules, or pure functions for small research kernels.
