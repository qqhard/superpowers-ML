# Installing SPML for Codex

Enable SPML in Codex through native skill discovery. SPML is an add-on to
`superpowers`, so install both skill sets into Codex's skills directory.

## Prerequisites

- Git

## Installation

### 1. Install Superpowers core skills

If you have not installed `superpowers` yet:

```bash
git clone https://github.com/obra/superpowers.git ~/.codex/superpowers
mkdir -p ~/.codex/skills
ln -s ~/.codex/superpowers/skills ~/.codex/skills/superpowers
```

### 2. Install SPML skills

1. **Clone the SPML repository:**
   ```bash
   git clone https://github.com/qqhard/superpowers-ML.git ~/.codex/spml
   ```

2. **Create the skills symlink:**
   ```bash
   mkdir -p ~/.codex/skills
   ln -s ~/.codex/spml/skills ~/.codex/skills/spml
   ```

   **Windows (PowerShell):**
   ```powershell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.codex\skills"
   cmd /c mklink /J "$env:USERPROFILE\.codex\skills\spml" "$env:USERPROFILE\.codex\spml\skills"
   ```

3. **Restart Codex** (quit and relaunch the CLI) to discover the skills.

## Migrating from old bootstrap

If you installed superpowers before native skill discovery, you need to:

1. **Update the repo:**
   ```bash
   cd ~/.codex/spml && git pull
   ```

2. **Create the skills symlink** (step 2 above) — this is the discovery mechanism.

3. **Remove the old bootstrap block** from `~/.codex/AGENTS.md` — any block referencing `spml-codex bootstrap` is no longer needed.

4. **Restart Codex.**

## Verify

```bash
ls -la ~/.codex/skills/spml
ls -la ~/.codex/skills/superpowers
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
rm ~/.codex/skills/spml
```

Optionally delete the clone: `rm -rf ~/.codex/spml`.
