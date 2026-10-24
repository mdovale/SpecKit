# Code Review

## Objective

Review the current diff against SpecKit’s numerical contracts, module boundaries, and tests.

## Steps

1. **Scope.** `git diff` and `git diff --cached`. If the tree is clean, review the file the user has open.
2. **Review.**

### API and conventions
- Public behavior stays on `compute_spectrum` / `SpectrumAnalyzer` / `SpectrumResult`, or on the module that already owns it (`noise`, `dsp`, `schedulers`, `systems`, `plotting`)
- One-sided spectral densities vs two-sided generator PSDs are stated where the quantity is defined
- CSD remains `Pxy = X * conj(Y)` unless the change is explicitly about that convention
- Defaults (`scheduler="vectorized_ltf"`, detrend `order`, window, `Jdes`, `Kdes`) change only when the diff says so

### Numerics
- Analytic errors and empirical `M2` errors are not mixed silently
- A kernel edit updates the Numba function and the `_np` fallback
- Window scaling uses the ENBW / overlap tables in `flattop.py`
- New branches handle short records, `fs <= 0`, and empty bands the way neighboring code does

### Performance
- Segment loops stay in the existing kernels (`numba.prange` or the chunked NumPy path)
- Windows and `_build_Q` bases stay cached per segment length
- `__init__.py` thread pinning (workqueue, BLAS threads = 1) stays intact

### Tests and packaging
- A behavior change has a seeded pytest that would fail on the old result
- Tolerances match the statistic, not a single lucky draw
- Python 3.9. No new dependency unless `pyproject.toml` is part of the change
- Ruff: line length 88, double quotes
- `ltpda/` is untouched unless the review is about the MATLAB reference

3. **Order findings:** incorrect results, API or convention breaks, missing NumPy/Numba pair, missing tests, then optional cleanups.

## Output

```markdown
## Code Review Summary

**Scope:** …

### Issues
| Location | Issue | Suggestion |
|----------|-------|------------|
| file:line | … | … |

### Contracts
- [sidedness, kernel parity, defaults]

### Strengths
- …

### Optional
- …
```

Be specific. Prefer a wrong scale or a missing fallback over style notes.
