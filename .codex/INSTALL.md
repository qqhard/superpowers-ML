# Installing SPML for Codex

Enable SPML in Codex through native skill discovery. SPML extends
`superpowers`; it does not replace it.

## Prerequisites

- Git

## Installation

### 1. Install Superpowers first

SPML depends on `superpowers` for shared software-engineering workflow. Install
`superpowers` first:

```bash
git clone https://github.com/obra/superpowers.git ~/.codex/superpowers
mkdir -p ~/.agents/skills
ln -s ~/.codex/superpowers/skills ~/.agents/skills/superpowers
```

If you prefer to let Codex do that step for you, tell it:

```text
Fetch and follow instructions from https://raw.githubusercontent.com/obra/superpowers/refs/heads/main/.codex/INSTALL.md
```

### 2. Install SPML skills

1. **Clone the SPML repository:**
   ```bash
   git clone https://github.com/qqhard/superpowers-ML.git ~/.codex/spml
   ```

2. **Create the skills symlink:**
   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/spml/skills ~/.agents/skills/spml
   ```

   **Windows (PowerShell):**
   ```powershell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.agents\skills"
   cmd /c mklink /J "$env:USERPROFILE\.agents\skills\spml" "$env:USERPROFILE\.codex\spml\skills"
   ```

3. **Restart Codex** (quit and relaunch the CLI) to discover the skills.

SPML relies on `superpowers:*` skills for shared software-engineering workflow.
After installation, both of these should exist:

```bash
~/.agents/skills/superpowers -> ~/.codex/superpowers/skills
~/.agents/skills/spml -> ~/.codex/spml/skills
```

## Verify

```bash
ls -la ~/.agents/skills/spml
ls -la ~/.agents/skills/superpowers
```

You should see symlinks (or junctions on Windows) pointing to both skills
directories.

## Updating

```bash
cd ~/.codex/spml && git pull
```

Skills update instantly through the symlink.

## Uninstalling

```bash
rm ~/.agents/skills/spml
```

Optionally delete the clone: `rm -rf ~/.codex/spml`.
