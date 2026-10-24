# Work Handoff

## Objective

Continue the work described in a `docs/handoffs/` note. The note supplies the goal, the repro, the constraints, and the finish line.

Use it for an open numerical bug, a partial kernel or API change, or the next phase of a feature.

## Invocation

```text
/work-handoff docs/handoffs/20260905_noise-generator-normalization.md
/work-handoff
```

If the path is missing, ask which handoff, and list the 3–5 newest files in `docs/handoffs/`.

Optional modifiers:

- `investigate only` — diagnosis and plan, no code
- `implement` — fix after a short orientation
- `update handoff only` — refresh the note from the tree, no code

Default: investigate, then implement, unless the note says to stop for a convention choice or the user asked for investigation only.

## Steps

### 1. Read the note

Read it end to end, then the code and tests it cites. When the note and `speckit/` disagree, follow the code and record the disagreement.

Skip a note that marks itself superseded, and ask whether to switch files.

### 2. Working brief

| Field | Take it from |
|-------|----------------|
| Goal | Problem / Goal |
| In scope | Scope, Context |
| Out of scope | Scope, non-goals |
| Status | Current status |
| Settled | Settled decisions, “do not re-litigate” |
| Done when | Success criteria |
| Check with | Repro, tests named in the note |

If a required section from `.cursor/rules/handoff-documents.mdc` is missing, name it, then continue with what is there.

### 3. Tree

`git status -sb`, `git branch --show-current`, `git log -8 --oneline`. Note dirty files that already touch this topic.

### 4. Guardrails

Read the rules that match the work before editing:

| Topic | Rule |
|-------|------|
| Units, kernels, module map | `.cursor/rules/project-context.mdc` |
| Scale factors, tolerances | `.cursor/rules/no-hacky-workarounds.mdc` |
| Editing the note itself | `.cursor/rules/no-delete-docs.mdc` |
| Commits | `.cursor/rules/git-workflow.mdc` |

### 5. Plan

Turn **Recommended next steps** into an ordered plan. Keep out-of-scope items out. For a bug, name the cause or the remaining hypothesis before changing production code.

State the plan before coding when a convention is still open or more than one fix would satisfy the note.

### 6. Execute

The smallest change that meets the success criteria. Update both Numba and NumPy paths when the statistic changes. Add the test the note’s gap section calls for.

If the only remaining patch is a fudge factor or a looser tolerance, stop, update this handoff, and say so.

### 7. Verify

Use the note’s success criteria as the checklist.

| Touch | Minimum check |
|-------|----------------|
| `schedulers`, `flattop`, `utils` | matching file in `tests/unit/` |
| `noise`, `dsp`, `systems` | matching file in `tests/unit/` |
| `analysis` API, errors, level | `tests/integration/test_analysis_api.py`, `test_errors.py`, `test_accuracy.py` as relevant |
| `core` kernels | the integration file that covers that estimate, plus a direct helper comparison when one exists |
| Docs only | re-read the snippet against the function; pytest optional |

Run `ruff check` on Python you edited.

### 8. Update the handoff

Refresh **Current status**, append **What was tried**, adjust **Recommended next steps**, and note the pytest command and result. Keep the old repro.

## Output

```markdown
## Handoff work: [title]

**Document:** …
**Mode:** investigate / implement / update only
**Branch / tree:** …

### Brief
- **Goal:** …
- **In scope:** …
- **Out of scope:** …
- **Settled:** …

### Plan
1. …

### Execution
…

### Verification
- `python -m pytest …` — pass/fail

### Handoff updates
- …
- **Remaining:** …
```

## Constraints

- The handoff sets scope. The rules set engineering constraints. The code is the implementation authority.
- Do not commit, push, or create a branch unless the user asks.
- Do not delete files under `docs/`.
