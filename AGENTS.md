# Repository Instructions

Use the repository-local skill at `.agents/skills/jax-migration-assistant/` for
JAX migration work.

Principles:

- Understand the original computation before editing.
- Migrate selectively; do not treat migration as syntax translation.
- Preserve the original implementation where practical as a reference.
- Keep plotting, I/O, pandas preprocessing, logging, configuration, and
  orchestration outside JAX unless there is a clear reason.
- Make state, randomness, parameters, optimizer state, and device transfers
  explicit.
- Verify functional equivalence for every migration.
- Treat optimization as a second phase after correctness.
- Require benchmarks before making performance claims.

Keep this repository small. Put durable human guidance in `docs/`, and put
agent-specific operational knowledge in the local skill references.
