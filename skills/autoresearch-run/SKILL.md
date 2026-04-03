---
name: autoresearch-run
description: Use when the user wants to execute an Auto Research loop - explicit entry point that locates the protocol and starts the autonomous iteration
---

# Auto Research — Run

Explicit entry point for executing an Auto Research loop. Expects a protocol to already exist (created by the `spml:autoresearch-create` → handoff flow).

## Prerequisite

The creation flow (`spml:autoresearch-create`) must have completed, producing:
- `autoresearch-protocol.md` — research protocol
- `experiences.md` — experience log (may have prior rounds if resuming)
- Base code committed to git

## Action

<HARD-GATE>
1. **Locate the protocol.** Ask the user for the experiment directory if not obvious from context. Verify `autoresearch-protocol.md` exists there.
2. **Invoke `spml:autoresearch`.** It handles startup verification, resume detection, the main loop, and final reporting.

If `autoresearch-protocol.md` does not exist, STOP and tell the user to run `spml:autoresearch-create` first to create the protocol.
</HARD-GATE>
