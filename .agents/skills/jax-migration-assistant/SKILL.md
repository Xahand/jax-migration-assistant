---
name: jax-migration-assistant
description: Analyze, plan, migrate, and verify existing NumPy, PyTorch, Python simulation, ML training, recurrent/sequence-model, or reinforcement-learning code for idiomatic JAX. Use when asked to convert, port, refactor, accelerate, audit, or decide whether existing Python numerical or ML code should move to JAX. Do not use for ordinary JAX questions that do not involve migration from an existing implementation.
---

# JAX Migration Assistant

## Overview

Use this skill for correctness-first JAX migrations. Treat migration as a
diagnostic workflow: understand the original computation, choose the migration
boundary, preserve a reference, migrate selectively, verify equivalence, then
optimize only when there is evidence.

## Required Workflow

1. Inspect the repository or files before editing.
2. Summarize the original computation internally: inputs, outputs, state,
   randomness, control flow, shapes, dtypes, and side effects.
3. Identify what should and should not move to JAX.
4. Preserve the original implementation where practical as a reference for
   tests.
5. Create the simplest correct JAX implementation before adding transformations.
6. Verify equivalence against the reference implementation.
7. Add `jit`, `vmap`, `lax.scan`, gradients, or device placement only when the
   computation and tests justify them.
8. Verify again after each transformation.
9. Benchmark before making performance claims.
10. Report what changed, what stayed outside JAX, verification performed, and
    remaining caveats.

## Migration Boundaries

Usually consider migrating:

- pure numerical kernels;
- differentiable model forward passes and losses;
- training steps with explicit parameters and optimizer state;
- independent batch computations suitable for `vmap`;
- fixed-length recurrences suitable for `lax.scan`;
- simulation or RL environment transitions when they are numerical, pure, and
  shape-stable.

Usually leave outside JAX:

- plotting and visualization;
- file I/O and dataset download code;
- pandas preprocessing;
- logging, configuration, and experiment tracking;
- high-level orchestration;
- dataloading code that only feeds compiled computations.

Explain any exception.

## JAX Issues To Make Explicit

- Functional state instead of hidden mutation.
- Immutable arrays and `.at[...]` updates.
- Explicit PRNG keys and key splitting.
- PyTrees for parameters, optimizer state, environment state, and batches.
- Scalar loss requirements and differentiated argument positions for gradients.
- Static versus traced values under `jit`.
- Shape stability and recompilation risk.
- Host/device transfer boundaries.
- Control-flow choices: Python loop, `vmap`, `lax.scan`, `lax.cond`,
  `lax.while_loop`.

## Reference Files

Read these only when relevant:

- `references/migration-workflow.md`: detailed step-by-step procedure.
- `references/migration-decisions.md`: what to migrate and what to leave alone.
- `references/common-failure-modes.md`: recurring bugs and diagnostics,
  including lessons from PyTorch-to-JAX translation research.
- `references/pytorch-to-jax-checklist.md`: focused checklist for PyTorch
  models, training loops, optimizers, gradients, randomness, and shape issues.

## Script

Use `scripts/compare_equivalence.py` as a lightweight helper for array
comparison when a project does not already have better test utilities. Prefer
the target repository's own test framework when available.

## Reporting Format

When finishing a migration, report:

- migrated components;
- intentionally unmigrated components;
- reference implementation preserved;
- tests or comparisons run;
- performance benchmarks run, if any;
- unresolved semantic differences or risks.
