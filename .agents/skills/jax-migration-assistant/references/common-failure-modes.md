# Common Failure Modes

This checklist is informed by observed PyTorch-to-JAX translation failures,
including Phan et al., "Learning Bug Context for PyTorch-to-JAX Translation with
LLMs" (<https://arxiv.org/abs/2510.09898>). Treat the paper as evidence that
these categories recur; do not claim the checklist guarantees correctness.

## Model Semantics

Check:

- weight matrix orientation;
- input, hidden, and output dimensions;
- broadcasting behavior;
- flatten, reshape, transpose, and batch-axis placement;
- train versus eval behavior;
- parameter initialization scale and dtype.

## Autodiff

Check:

- the differentiated function returns a scalar when required;
- `grad` or `value_and_grad` differentiates the intended argument;
- nondifferentiated state is not accidentally treated as a parameter;
- auxiliary outputs are handled explicitly;
- gradients match a tiny reference case.

## Randomness

Check:

- implicit NumPy or PyTorch RNG is not translated as hidden global state;
- each stochastic operation receives a PRNG key;
- independent random operations use split keys;
- keys are threaded through training or simulation state when needed;
- tests do not assume NumPy, PyTorch, and JAX share identical random streams.

## Training State

Check:

- parameters and optimizer state are explicit;
- updates return new state instead of mutating in place;
- optimizer updates use the computed gradients;
- batch normalization, dropout, recurrent state, and environment state are
  represented intentionally;
- checkpoint loading and saving still match the chosen parameter structure.

## JIT And Tracing

Check:

- `jit` is not applied blindly;
- static arguments are marked only when genuinely static;
- Python branches do not depend on traced array values;
- changing shapes do not cause excessive recompilation;
- host callbacks, printing, logging, or data loading are outside compiled code.

## Control Flow

Check:

- a Python loop is not converted to `scan` unless it is part of the numerical
  computation;
- `vmap` is used only for independent batch axes;
- `scan` carry structure and shapes are stable across iterations;
- early termination semantics are preserved or explicitly changed;
- RL episode boundaries and reset behavior remain correct.

## Incomplete Translation

Check:

- no leftover `torch` tensors flow into JAX computations unintentionally;
- no NumPy arrays are mutated expecting JAX-style behavior, or vice versa;
- device transfers are explicit where they matter;
- tests exercise more than a single happy-path forward pass.
