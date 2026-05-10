# Skill Registry — forecasting-caja

Generated: 2026-05-09
Mode: engram

---

## Project Conventions

| Source | Status |
|--------|--------|
| AGENTS.md | ❌ Not found |
| CLAUDE.md | ❌ Not found (project-level) |
| .cursorrules | ❌ Not found |
| GEMINI.md | ❌ Not found |
| copilot-instructions.md | ❌ Not found |

> **Greenfield project**: No conventions established yet. All standards are pending definition during the first change.

---

## Compact Rules

*No project-specific conventions established. Default agent behaviors apply.*

---

## Available Skills

| Skill | Source | Trigger |
|-------|--------|---------|
| branch-pr | `~/.config/opencode/skills/branch-pr` | When creating a pull request, opening a PR, or preparing changes for review |
| doc-extractor | `~/.claude/skills/doc-extractor` | When reading an academic PDF to extract its content for study notes |
| doc-verifier | `~/.claude/skills/doc-verifier` | When verifying study notes against source documents |
| doc-writer | `~/.claude/skills/doc-writer` | When creating study notes from multiple extracted PDF sources |
| go-testing | `~/.config/opencode/skills/go-testing` | When writing Go tests, using teatest, or adding test coverage |
| humanities-research | `~/.config/opencode/skills/humanities-research` | Cuando se trabaja con búsqueda bibliográfica, análisis de fuentes primarias, gestión de citas académicas, archivos históricos o patrimonio cultural |
| issue-creation | `~/.config/opencode/skills/issue-creation` | When creating a GitHub issue, reporting a bug, or requesting a feature |
| judgment-day | `~/.config/opencode/skills/judgment-day` | When user says "judgment day", "judgment-day", "review adversarial", "dual review", "doble review", "juzgar", "que lo juzguen" |
| openmanus | `~/.config/opencode/skills/openmanus` | OpenManus agent framework — browser automation, web search, complex actions |
| skill-creator | `~/.config/opencode/skills/skill-creator` | When user asks to create a new skill, add agent instructions, or document patterns for AI |

### SDD Skills (internal — not for direct invocation)

| Skill | Purpose |
|-------|---------|
| sdd-init | Initialize SDD context in a project |
| sdd-explore | Explore and investigate ideas before committing to a change |
| sdd-propose | Create a change proposal with intent, scope, and approach |
| sdd-spec | Write specifications with requirements and scenarios |
| sdd-design | Create technical design document with architecture decisions |
| sdd-tasks | Break down a change into implementation tasks |
| sdd-apply | Implement tasks from the change specification |
| sdd-verify | Validate implementation matches specs, design, and tasks |
| sdd-archive | Sync delta specs to main specs and archive a completed change |

---

## Registered

- **Date**: 2026-05-09
- **Total skills**: 10 user-available + 9 SDD internal
