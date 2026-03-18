# SPML for Codex

Guide for using SPML with OpenAI Codex via native skill discovery.

SPML extends `superpowers`; it does not replace it. Install `superpowers`
first, then install SPML alongside it so Codex can discover both skill sets.

## Quick Install

Tell Codex:

```
Fetch and follow instructions from https://raw.githubusercontent.com/obra/superpowers/refs/heads/main/.codex/INSTALL.md, then fetch and follow instructions from https://raw.githubusercontent.com/qqhard/superpowers-ML/refs/heads/main/.codex/INSTALL.md
```

## Manual Installation

### Prerequisites

- OpenAI Codex CLI
- Git

### Steps

1. Install `superpowers` first by following the upstream Codex install guide:
   <https://raw.githubusercontent.com/obra/superpowers/refs/heads/main/.codex/INSTALL.md>

   That should leave you with:
   ```bash
   git clone https://github.com/obra/superpowers.git ~/.codex/superpowers
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/superpowers/skills ~/.agents/skills/superpowers
   ```

2. Clone the SPML repo:
   ```bash
   git clone https://github.com/qqhard/superpowers-ML.git ~/.codex/spml
   ```

3. Create the skills symlink:
   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/spml/skills ~/.agents/skills/spml
   ```

4. Restart Codex.

### Windows

Use junctions instead of symlinks:

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.agents\skills"
cmd /c mklink /J "$env:USERPROFILE\.agents\skills\superpowers" "$env:USERPROFILE\.codex\superpowers\skills"
cmd /c mklink /J "$env:USERPROFILE\.agents\skills\spml" "$env:USERPROFILE\.codex\spml\skills"
```

## How It Works

Codex has native skill discovery. Keep each skill collection visible under
`~/.agents/skills/` and Codex can discover the `SKILL.md` frontmatter at startup.

```
~/.agents/skills/superpowers/ -> ~/.codex/superpowers/skills/
~/.agents/skills/spml/ -> ~/.codex/spml/skills/
```

`superpowers` provides the general software-development workflow.
SPML adds ML experiment workflow on top of it and continues to rely on
`superpowers:*` skills for shared discipline such as TDD, planning,
verification, code review, and development-branch handoff.

## Usage

Skills are discovered automatically. Codex activates them when:
- You mention a skill by name (e.g., "use brainstorming")
- The task matches a skill's description
- A bootstrap skill such as `using-superpowers` or `using-superpowers-ml` directs Codex to use one

### Personal Skills

Create your own skills in `~/.agents/skills/`:

```bash
mkdir -p ~/.agents/skills/my-skill
```

Create `~/.agents/skills/my-skill/SKILL.md`:

```markdown
---
name: my-skill
description: Use when [condition] - [what it does]
---

# My Skill

[Your skill content here]
```

The `description` field is how Codex decides when to activate a skill automatically — write it as a clear trigger condition.

## Updating

```bash
cd ~/.codex/superpowers && git pull
cd ~/.codex/spml && git pull
```

Skills update instantly through the symlinks.

## Uninstalling

```bash
rm ~/.agents/skills/spml
```

**Windows (PowerShell):**
```powershell
Remove-Item "$env:USERPROFILE\.agents\skills\spml"
```

Optionally delete the clone: `rm -rf ~/.codex/spml`

## Troubleshooting

### Skills not showing up

1. Verify the symlinks: `ls -la ~/.agents/skills/superpowers ~/.agents/skills/spml`
2. Check skills exist: `ls ~/.codex/superpowers/skills ~/.codex/spml/skills`
3. Restart Codex — skills are discovered at startup

### Windows junction issues

Junctions normally work without special permissions. If creation fails, try running PowerShell as administrator.

## Getting Help

- SPML issues: https://github.com/qqhard/superpowers-ML/issues
- Superpowers issues: https://github.com/obra/superpowers/issues
- SPML repository: https://github.com/qqhard/superpowers-ML
