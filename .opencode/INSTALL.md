# Installing SPML for OpenCode

## Prerequisites

- [OpenCode.ai](https://opencode.ai) installed
- Git installed

## Installation Steps

### 1. Clone Superpowers

```bash
git clone https://github.com/qqhard/superpowers-ML.git ~/.config/opencode/spml
```

### 2. Register the Plugin

Create a symlink so OpenCode discovers the plugin:

```bash
mkdir -p ~/.config/opencode/plugins
rm -f ~/.config/opencode/plugins/superpowers.js
ln -s ~/.config/opencode/spml/.opencode/plugins/superpowers.js ~/.config/opencode/plugins/superpowers.js
```

The plugin auto-registers the SPML skills directory at startup, so no separate
skills symlink is required.

### 3. Restart OpenCode

Restart OpenCode. The plugin will automatically inject the SPML bootstrap
context and register the skills directory.

Verify by asking: "do you have SPML?"

## Usage

### Finding Skills

Use OpenCode's native `skill` tool to list available skills:

```
use skill tool to list skills
```

### Loading a Skill

Use OpenCode's native `skill` tool to load a specific skill:

```
use skill tool to load spml/ml-brainstorming
```

### Personal Skills

Create your own skills in `~/.config/opencode/skills/`:

```bash
mkdir -p ~/.config/opencode/skills/my-skill
```

Create `~/.config/opencode/skills/my-skill/SKILL.md`:

```markdown
---
name: my-skill
description: Use when [condition] - [what it does]
---

# My Skill

[Your skill content here]
```

### Project Skills

Create project-specific skills in `.opencode/skills/` within your project.

**Skill Priority:** Project skills > Personal skills > Superpowers skills

## Updating

```bash
cd ~/.config/opencode/spml
git pull
```

## Troubleshooting

### Plugin not loading

1. Check plugin symlink: `ls -l ~/.config/opencode/plugins/superpowers.js`
2. Check source exists: `ls ~/.config/opencode/spml/.opencode/plugins/superpowers.js`
3. Check OpenCode logs for errors

### Skills not found

1. Use `skill` tool to list what's discovered
2. Verify the plugin is loading (see above) — the plugin auto-registers
   `~/.config/opencode/spml/skills` at startup

### Tool mapping

When skills reference Claude Code tools:
- `TodoWrite` → `todowrite`
- `Task` with subagents → `@mention` syntax
- `Skill` tool → OpenCode's native `skill` tool
- File operations → your native tools

## Getting Help

- Report issues: https://github.com/qqhard/superpowers-ML/issues
- Full documentation: https://github.com/qqhard/superpowers-ML/blob/main/docs/README.opencode.md
