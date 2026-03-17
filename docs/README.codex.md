# SPML for Codex

Guide for using SPML with OpenAI Codex via native skill discovery.

SPML extends `superpowers`; it does not replace it. Install `superpowers`
first, then install SPML alongside it.

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

1. Install `superpowers` first:
   ```bash
   git clone https://github.com/obra/superpowers.git ~/.codex/superpowers
   mkdir -p ~/.codex/skills
   ln -s ~/.codex/superpowers/skills ~/.codex/skills/superpowers
   ```

2. Clone the SPML repo:
   ```bash
   git clone https://github.com/qqhard/superpowers-ML.git ~/.codex/spml
   ```

3. Create the skills symlink:
   ```bash
   mkdir -p ~/.codex/skills
   ln -s ~/.codex/spml/skills ~/.codex/skills/spml
   ```

4. Restart Codex.

### Windows

Use junctions instead of symlinks:

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.codex\skills"
cmd /c mklink /J "$env:USERPROFILE\.codex\skills\superpowers" "$env:USERPROFILE\.codex\superpowers\skills"
cmd /c mklink /J "$env:USERPROFILE\.codex\skills\spml" "$env:USERPROFILE\.codex\spml\skills"
```

## How It Works

Codex has native skill discovery. Keep each skill collection visible under
`~/.codex/skills/` and Codex can discover the `SKILL.md` frontmatter at startup.

```
~/.codex/skills/superpowers/ -> ~/.codex/superpowers/skills/
~/.codex/skills/spml/ -> ~/.codex/spml/skills/
```

`superpowers` provides the general software-development workflow.
SPML provides the ML experiment workflow on top of that.

## Usage

Skills are discovered automatically. Codex activates them when:
- You mention a skill by name (e.g., "use brainstorming")
- The task matches a skill's description
- A bootstrap skill such as `using-superpowers` or `using-spml` directs Codex to use one

### Personal Skills

Create your own skills in `~/.codex/skills/`:

```bash
mkdir -p ~/.codex/skills/my-skill
```

Create `~/.codex/skills/my-skill/SKILL.md`:

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
rm ~/.codex/skills/spml
```

**Windows (PowerShell):**
```powershell
Remove-Item "$env:USERPROFILE\.codex\skills\spml"
```

Optionally delete the clone: `rm -rf ~/.codex/spml`

## Troubleshooting

### Skills not showing up

1. Verify the symlinks: `ls -la ~/.codex/skills/superpowers ~/.codex/skills/spml`
2. Check skills exist: `ls ~/.codex/superpowers/skills ~/.codex/spml/skills`
3. Restart Codex — skills are discovered at startup

### Windows junction issues

Junctions normally work without special permissions. If creation fails, try running PowerShell as administrator.

## Getting Help

- SPML issues: https://github.com/qqhard/superpowers-ML/issues
- Superpowers issues: https://github.com/obra/superpowers/issues
- SPML repository: https://github.com/qqhard/superpowers-ML
