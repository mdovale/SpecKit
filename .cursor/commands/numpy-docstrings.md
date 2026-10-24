# NumPy Docstrings

## Objective

Check and fix NumPy-style docstrings in the scope the user named. The docstring describes the code and the tests. It does not invent a formula or a convention.

## Invocation

The user names the scope: a function, a class, a file, or the current diff. If they name nothing, use the file they have open. Do not walk `speckit/` unless they asked for the package or a directory.

Public functions, classes, and methods in that scope get a full docstring. Private functions in scope get one when they implement a statistic, a plan, or a window quantity. Skip local closures.

## Form

Triple double quotes. One summary line, capital letter, period at the end. Keep a summary that already describes the function. SpecKit summaries are descriptive (`Performs ...`, `Calculates ...`). Do not rewrite them into imperative mood.

Then the sections the function actually needs, in this order: extended description, `Parameters`, `Returns`, `Yields`, `Raises`, `Notes`, `References`, `Examples`. Omit empty sections.

```python
def example(x: np.ndarray, fs: float, order: int = 0) -> np.ndarray:
    """Estimate the one-sided spectrum of ``x``.

    Parameters
    ----------
    x : np.ndarray
        Input series.
    fs : float
        Sampling frequency in Hz. Must be positive.
    order : int, optional
        Detrend order. Defaults to 0.

    Returns
    -------
    psd : np.ndarray
        One-sided power spectral density.

    Raises
    ------
    ValueError
        If ``fs`` is not positive.
    """
```

- A blank line separates the summary from the body, and each section from the next. No blank line between a section name and its dashed underline, or between the underline and the first entry.
- Match the file on blank lines between parameter entries. `analysis.py`, `core.py`, and `noise.py` pack them.
- Every signature parameter appears under `Parameters`, except `self` and `cls`. Document the default in the description.
- The docstring type agrees with the annotation (`array-like` may pair with `ArrayLike`). Write shapes the code guarantees, as in `Q : (L, order + 1) ndarray`.
- `Raises` lists exceptions the body raises on purpose. Keep an exception callers can hit. Do not add exceptions that only escape from NumPy or Numba.
- `**kwargs` says what they are forwarded to when that is what the code does.
- Line length 88. Double quotes.

## Spectral claims

State the convention in the docstring of the function that returns or consumes the quantity.

- `compute_spectrum` spectral densities are one-sided.
- Noise-generator `psd` arguments are two-sided.
- CSD is `Pxy = X * conj(Y)`.
- A kernel docstring names the statistic (`MXX`, `MYY`, `mu_r`, `mu_i`, `M2`) and that the `_np` fallback computes the same one.

If the docstring and the code disagree, follow `speckit/` and `tests/`. If the disagreement is a convention choice or an unchecked formula, stop and write a handoff (`.cursor/rules/no-hacky-workarounds.mdc`) instead of editing the formula text.

## Steps

1. **Read the scope.** Compare each docstring with its signature and body.
2. **Check, without editing `pyproject.toml`.**

```bash
ruff check <paths> --select D,DOC --ignore D100,D104,D401 --preview --target-version py39 \
  --config "lint.pydocstyle.convention = \"numpy\""
```

`D401` stays ignored. `D100` and `D104` matter only when the user asked for a module or package docstring that does not exist. `DOC` is a preview aid: apply a finding when it matches the body, and leave a documented exception that callers can still hit.

3. **Edit docstrings only.** Do not change annotations or executable code in this command.
4. **Check the edit.** `ruff check` and `ruff format --check` on the touched files with the repo config. Re-run the NumPy docstring command above on those files. A docstring-only edit does not need pytest. If a code line changed, run the narrowest test that covers it.

## Output

```markdown
## NumPy Docstrings

**Scope:** …

| Symbol | Change |
|--------|--------|
| `function` | sections added or corrected, or left unchanged and why |

### Convention
- One-sided, two-sided, or CSD wording touched, if any

### Check
- `ruff check` result

### Left open
- A formula or convention that needs a handoff, if any
```

## Constraints

- Do not delete an accurate `Notes`, `References`, or `Examples` section.
- Do not add a dependency.
- Do not commit unless the user asks.
