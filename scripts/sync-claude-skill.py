#!/usr/bin/env python3
"""Sync the canonical Codex skill into Claude's project-skill location."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEX_SKILL = ROOT / ".agents" / "skills" / "jax-migration-assistant"
CLAUDE_SKILL = ROOT / ".claude" / "skills" / "jax-migration-assistant"


def ignore_codex_only(_directory: str, names: list[str]) -> set[str]:
    return {"agents"} & set(names)


def main() -> int:
    if not CODEX_SKILL.exists():
        raise SystemExit(f"missing canonical skill: {CODEX_SKILL}")

    if CLAUDE_SKILL.exists():
        shutil.rmtree(CLAUDE_SKILL)

    shutil.copytree(CODEX_SKILL, CLAUDE_SKILL, ignore=ignore_codex_only)
    print(f"synced {CODEX_SKILL} -> {CLAUDE_SKILL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
