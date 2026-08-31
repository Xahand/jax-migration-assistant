# PyTorch To JAX Checklist

Use this when the original code uses PyTorch models, tensors, autograd, or
optimizers.

## Before Editing

- Locate `nn.Module` definitions, `forward` methods, losses, optimizers, and
  training loops.
- Identify tensor shapes at module boundaries.
- Identify mutable state: parameters, buffers, optimizer state, metrics, RNG,
  dataloader state, hidden recurrent state.
- Identify training/eval mode differences.
- Find existing tests, checkpoints, fixtures, or deterministic scripts.

## Model Translation

- Verify all linear, convolution, embedding, normalization, and recurrent layer
  dimensions.
- Preserve activation order and residual paths.
- Confirm broadcasting semantics.
- Decide between Flax, Equinox, or pure functions based on project conventions.
- Keep initialization explicit and testable.

## Training Step

Prefer a pure training step shaped like:

```python
def train_step(state, batch, key):
    key, subkey = jax.random.split(key)

    def loss_fn(params):
        predictions = apply_fn(params, batch["x"], subkey)
        loss = compute_loss(predictions, batch["y"])
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, key, loss
```

Adapt this pattern to the chosen library. Do not hide parameters, optimizer
state, or PRNG state in global objects.

## Dataloaders

Keep PyTorch `DataLoader` or Python input pipelines outside `jit` unless there
is a specific reason to replace them. Convert batches at the boundary.

## Random Layers

Dropout, stochastic policies, exploration, sampling, data augmentation, and
random resets need explicit key flow. Split keys before independent operations.

## Verification

For PyTorch migrations, compare:

- forward outputs on controlled parameters and inputs;
- loss values;
- gradients on a tiny deterministic batch;
- one optimizer update;
- recurrent hidden-state transitions;
- train/eval behavior when applicable.

Expect small numerical differences. Investigate shape, dtype, and semantic
differences before loosening tolerances.
