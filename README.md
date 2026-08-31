# JAX Migration Assistant

A small research-group guide and Codex skill for migrating existing numerical
and machine-learning Python code to idiomatic JAX.

This project is for researchers who already know Python, NumPy, PyTorch, or ML
training code, but have little or moderate JAX experience. It is intentionally
not a syntax-translation prompt. The assistant first tries to understand the
original computation, decides which parts should actually move to JAX, preserves
the original implementation where practical, and verifies functional
equivalence before making performance claims.

## What It Helps With

- NumPy numerical kernels
- PyTorch models and training steps
- Python simulation loops
- Neural-network training code
- Recurrent and sequence models
- Reinforcement-learning environments and training loops

## What It Avoids

The assistant should not convert plotting, file I/O, pandas preprocessing,
configuration, logging, experiment tracking, or orchestration code to JAX unless
there is a concrete reason. JAX is most valuable at numerical computation
boundaries, not as a replacement for every Python library in a research codebase.

## Repository Contents

- `docs/migration-guide.md`: human-readable migration workflow.
- `docs/verification.md`: equivalence testing and benchmarking guidance.
- `examples/minimal-equivalence/`: a tiny original-versus-JAX example.
- `.agents/skills/jax-migration-assistant/`: repository-local Codex skill.

## Local Checks

The repository has no required runtime dependency. To run the example test:

```bash
python -m pip install -e ".[examples]"
python -m pytest examples/minimal-equivalence -q
```

To run the Codex skill validator, install the optional skill-development
dependency first:

```bash
python -m pip install -e ".[skill-dev]"
python /path/to/quick_validate.py .agents/skills/jax-migration-assistant
```

## Using The Skill

When Codex is working in this repository, the local skill is available at:

```text
.agents/skills/jax-migration-assistant/
```

For another project, copy or install that skill into the target repository's
`.agents/skills/` directory, then ask Codex to use `$jax-migration-assistant`.

Example prompt:

```text
Use $jax-migration-assistant to inspect this repository and migrate only the
numerical kernels that benefit from JAX. Preserve the original implementation
for equivalence tests and report what you intentionally left outside JAX.
```

## Research Context

This resource is partly motivated by Phan et al., "Learning Bug Context for
PyTorch-to-JAX Translation with LLMs" ([arXiv:2510.09898](https://arxiv.org/abs/2510.09898)).
The paper provides evidence that LLM-generated PyTorch-to-JAX translations often
contain recurring framework-specific bugs, and that known bug/fix context can
improve translation quality. This project applies that lesson as a diagnostic
workflow: inspect, migrate selectively, test, compare, and fix.

For technical JAX behavior, prefer the official JAX documentation:
<https://docs.jax.dev/>.
