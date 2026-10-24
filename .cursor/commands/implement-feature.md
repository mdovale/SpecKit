# Implement Feature

## Objective

Implement a behavior the user names (for example a scheduler option, an error quantity, a noise parameter, a plot helper). Keep the current module boundaries.

## Steps

1. **Name the behavior.** If the request is ambiguous, state the assumption and the boundary in a few lines, then proceed when the owning module is clear. Ask only when two modules could own it.
2. **Read the owner.** Match the surrounding style and the existing public entry point (`compute_spectrum` kwargs, a `SpectrumResult` attribute, or a function in `noise`, `dsp`, `systems`, `schedulers`, `plotting`).
3. **Implement in that module.**
   - Hot bin statistics go in `speckit/core.py` as a Numba kernel and a NumPy `_np` fallback that return the same statistic.
   - Frequency grids go in `speckit/schedulers.py`. Window ENBW and overlap go in `speckit/flattop.py`.
   - State the PSD convention in the docstring (NumPy style, as in `analysis.py`).
   - Stay on Python 3.9 and the dependencies already in `pyproject.toml`.
   - Copy the SpecKit BSD header onto a new Python file. Leave the PyTDI header on `speckit/lagrange.py` in place.
4. **Wire it.** Reachable from the existing API. A new public function is exported from `speckit/__init__.py` only when users should import it from the package root.
5. **Lock it.** Add or extend a pytest in `tests/unit/` or, for end-to-end spectral claims, `tests/integration/`. Use a seeded generator. The test must fail if the scale, sidedness, or kernel choice regresses.
6. **Check.** `python -m pytest` on the new test, then `ruff check` on the touched files.

If the only way to make the numbers match is a local fudge factor or a looser tolerance, stop and write a handoff (`.cursor/rules/no-hacky-workarounds.mdc`).

## Output

- What was added, in which module, and the convention it follows
- The test command and its result
- Follow-ups still open (docs, notebook, manuscript) if the user did not ask for them
