# Onboard

## Objective

Orient at the start of a session. Learn what SpecKit is, which files define behavior, which constraints apply, and what the tree is doing.

This is a read-and-orient pass. Do not change code unless the user also asked for work in the same message.

## Steps

### 1. Session context

Run in parallel when useful:

- `git status -sb` and `git branch --show-current`
- `git log -8 --oneline`
- Skim the latest `docs/handoffs/` files (3–5 by date prefix)

Record the branch, whether the tree is dirty, and any handoff that matches the task.

### 2. Source of truth

Use this order:

1. `speckit/` implementation
2. `tests/` that lock a numerical claim or API behavior
3. Docstrings on the public functions
4. `README.md` and `notebooks/` (they can lag the code)
5. `.cursor/rules/`
6. `docs/handoffs/` and `docs/reference/`
7. `docs/paper/` (manuscript narrative)

`ltpda/` is a vendored MATLAB reference. It is not the Python package.

A known open discrepancy is `docs/handoffs/20260905_noise-generator-normalization.md` (generator level and one-sided vs two-sided wording). Check it before editing `speckit/noise.py` or spectral scaling.

### 3. Repo map

SpecKit estimates spectra with logarithmic frequency scheduling, analytic errors (Bendat & Piersol), and empirical scatter (`M2`). Python 3.9+. Numba kernels with NumPy fallbacks.

| Area | Location |
|------|----------|
| Public spectrum API | `speckit/analysis.py`, `speckit/__init__.py` |
| Bin kernels | `speckit/core.py` |
| Frequency plans | `speckit/schedulers.py` |
| Windows / ENBW | `speckit/flattop.py` |
| DSP helpers | `speckit/dsp.py` |
| Noise generators | `speckit/noise.py` |
| SISO / MISO | `speckit/systems.py` |
| Figures | `speckit/plotting.py` |
| Unit tests | `tests/unit/` |
| Accuracy, errors, API | `tests/integration/` |
| Shared fixtures | `tests/conftest.py` |
| Tutorials | `notebooks/` |
| Manuscript | `docs/paper/` |

Spectral densities from `compute_spectrum` are one-sided. Noise-generator `psd` arguments are two-sided. CSD is `Pxy = X * conj(Y)`.

### 4. Constraints that always apply

| Constraint | Where |
|------------|--------|
| Conventions, layout, Python 3.9, ruff | `.cursor/rules/project-context.mdc` |
| No tolerance or scale-factor patches | `.cursor/rules/no-hacky-workarounds.mdc` |
| Keep authored docs | `.cursor/rules/no-delete-docs.mdc` |
| Commit only when asked; linear trunk | `.cursor/rules/git-workflow.mdc` |

### 5. Commands and checks

Dev install: `pip install -e '.[dev]'`.

| Check | Command |
|-------|---------|
| One test | `python -m pytest tests/unit/test_noise.py::test_white_noise_statistics -q` |
| One file | `python -m pytest tests/unit/test_schedulers.py -q` |
| Integration | `python -m pytest tests/integration -q` |
| Style | `ruff check speckit tests` and `ruff format --check speckit tests` |

| Command | Use when |
|---------|----------|
| `onboard` | Session start |
| `implement-feature` | Add a behavior in the owning module |
| `feature-edit-e2e` | Change a public API or numerical claim; analysis first |
| `add-test` | Lock a behavior in pytest |
| `code-review` | Review the current diff |
| `type-annotations` | Check and fix annotations in the named scope |
| `numpy-docstrings` | Check and fix NumPy docstrings in the named scope |
| `write-feature-doc` | Write `docs/reference/` from the code |
| `create-handoff` / `work-handoff` | Pause or resume an investigation |
| `plan-status` | Score a plan document against the tree |
| `commit-message` | Draft a message |
| `commit-step` / `commit-all-focused` | User asked to commit |

### 6. Read before editing

- Spectrum API or errors: `speckit/analysis.py`, `tests/integration/`
- Kernels: `speckit/core.py` (Numba and `_np` together)
- Plans: `speckit/schedulers.py`, `notebooks/0.6_schedulers.ipynb`
- Noise level: `speckit/noise.py`, `tests/unit/test_noise.py`, the noise handoff
- Windows: `speckit/flattop.py`, `notebooks/0.7_windows.ipynb`

Ask one question only when the task still has two unrelated targets after this pass.

## Output

```markdown
## Session Onboard

### Repo snapshot
- **Branch:** …
- **Working tree:** …
- **Recent focus:** …

### What SpecKit is
[2–3 sentences]

### Source of truth
- Code and tests, then docstrings. README, notebooks, and the paper can lag.

### Touchpoints
| Concern | Files |
|---------|-------|
| … | … |

### Constraints
- …

### First reads
- …

### Verification
- …

### Assumptions
- …
```

Keep the brief under about 60 lines.

## Constraints

- Do not modify code, docs, or git state during onboard unless the user combined it with another task.
- Do not commit, push, or create a branch.
