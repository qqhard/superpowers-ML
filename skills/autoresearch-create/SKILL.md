---
name: autoresearch-create
description: Use when the user wants to start an Auto Research workflow - explicit entry point that activates autoresearch mode and routes into brainstorming for protocol definition
---

# Auto Research — Create

Explicit entry point for creating an Auto Research project.

Auto Research is an automated iteration loop that designs a research protocol, builds validated baseline code, then iterates autonomously — Agent A modifies + trains, Agent B evaluates, Supervisor manages git state — accumulating experience until the target is reached or max rounds complete.

## Full Flow

```
spml:autoresearch-create (you are here)
  -> spml:ml-brainstorming (autoresearch mode — 8 protocol questions)
    -> spml:experiment-planning
      -> spml:ml-subagent-dev
        -> spml:autoresearch-handoff (Post-Completion Gate: "Research")
          -> [new session] spml:autoresearch-run
```

## Action

<HARD-GATE>
Autoresearch mode is now ACTIVE. Invoke `spml:ml-brainstorming`.

When ml-brainstorming loads, go DIRECTLY to the autoresearch protocol definition flow (the 8 questions starting with Question 0: Experiment directory). Do NOT perform keyword detection — autoresearch intent is already confirmed by this command.

Do NOT explore the project first. Do NOT dispatch Explore agents or read code before asking the user. Ask the user questions first.
</HARD-GATE>
