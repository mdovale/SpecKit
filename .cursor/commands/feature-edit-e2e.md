# Feature Edit End-to-End

## Objective

Change an existing public behavior with a written review first. Cover the API, the units, and the numerical claim. **Do not edit code until the analysis is delivered and the user confirms, or they already said to proceed.**

## Invocation

The user names a feature. If it is ambiguous, ask: "Which behavior should I analyze?"

Treat `speckit/` and `tests/` as the authority for current behavior. README, notebooks, and `docs/paper/` describe intent and can be behind the code.

## Phase 1: Analysis

### Scope

- **Changing:** functions, attributes, kernels, and conventions this edit will touch
- **Unchanged:** adjacent quantities, schedulers, and plot helpers outside that boundary
- **Boundary:** the smallest surface that makes the requested behavior true

### Review

| Dimension | Question |
|-----------|----------|
| **Call shape** | Who calls this, with what array shapes (`1d` auto vs `2 x N` cross)? |
| **Units** | One-sided or two-sided? Per Hz? ENBW included? What does the docstring promise? |
| **Definition** | Which formula (analytic error, `M2`, window, detrend order)? Does the NumPy path match Numba? |
| **Defaults** | Do `Jdes`, `Kdes`, `order`, `olap`, `win`, or `scheduler` stay stable? |
| **Failure** | What happens on bad `fs`, short `N`, empty band, or missing `fres`/`L`? |
| **Plots** | Does `SpectrumResult.plot` or `plotting.py` change scale, labels, or error bars? |

### Gap list

**High** — wrong level, wrong sidedness, kernel disagreement, or a public default change.

**Medium** — docstring, return shape, or missing test for a claim the code already makes.

**Low** — naming, plot styling, notebook drift.

### Recommended edits

For each: what, which file, and which formula or test defines done.

## Phase 2: Plan

Files, ordered steps, and risks (API break, tolerance, Numba compile, runtime of integration tests).

## Phase 3: Implementation (after confirmation)

- Minimal edit inside the feature boundary
- Update the paired Numba and NumPy paths together
- Docstring states the convention the code implements
- Add a regression test that fails on the old behavior
- No new dependencies

## Phase 4: Verification

- `python -m pytest` on the narrowest covering test, then the relevant file under `tests/unit/` or `tests/integration/`
- `ruff check` on the touched files
- If a notebook or the manuscript states the old claim, list it as a follow-up unless the user asked to update it

## Output before coding

```markdown
## Feature Edit: [name]

### Scope
- **Changing:** …
- **Unchanged:** …
- **Boundary:** …

### Review
[table]

### Gaps
**High:** …
**Medium:** …
**Low:** …

### Recommended edits
1. …

### Plan
**Files:** …
**Steps:** …
**Risks:** …

Proceed with implementation?
```

After confirmation, implement and report the test command and result.
