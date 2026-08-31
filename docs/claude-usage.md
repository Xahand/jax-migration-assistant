# Using The Skill With Claude

This repository supports both Codex and Claude:

- Codex project skill: `.agents/skills/jax-migration-assistant/`
- Claude project skill: `.claude/skills/jax-migration-assistant/`

The Codex copy is the canonical source. After changing it, run:

```bash
python3 scripts/sync-claude-skill.py
```

This refreshes the Claude copy and omits Codex-only UI metadata.

## Claude Code

Claude Code discovers repository skills at:

```text
.claude/skills/<skill-name>/SKILL.md
```

For this repository, that means:

```text
.claude/skills/jax-migration-assistant/SKILL.md
```

To use the skill in another project, copy the `jax-migration-assistant` folder
under that project's `.claude/skills/` directory.

## Claude.ai Custom Skill

For Claude.ai, package the skill folder as a ZIP whose root contains the skill
directory:

```text
jax-migration-assistant.zip
└── jax-migration-assistant/
    ├── SKILL.md
    ├── references/
    └── scripts/
```

One way to create that ZIP locally:

```bash
mkdir -p dist
cd .claude/skills
zip -r ../../dist/jax-migration-assistant-claude.zip jax-migration-assistant
```

Upload the ZIP through Claude's custom skill interface. Keep code execution
enabled if you want Claude to run bundled scripts.

## Maintenance Notes

Keep the `description` field short and specific. Claude uses it to decide when
to load the skill, and its public skill guidance recommends concise descriptions.

Do not add Codex-specific files such as `agents/openai.yaml` to the Claude skill
copy. Claude only needs `SKILL.md` plus relevant supporting resources.

## References

- Claude repository skill discovery:
  <https://platform.claude.com/docs/en/managed-agents/skills>
- Claude custom skill structure and packaging:
  <https://claude.com/docs/skills/how-to>
