# Type Annotations

## Objective

Check and fix type annotations in the scope the user named. Annotations describe the code that is already there. They do not change defaults, control flow, or numeric results.

## Invocation

The user names the scope: a function, a class, a file, or the current diff. If they name nothing, use the file they have open. Do not walk `speckit/` unless they asked for the package or a directory.

Stay inside that scope. A function named in the request includes its signature and the imports that signature needs. It does not include neighboring functions.

## Style

Python 3.9. Match the annotations already used in the file.

- `Optional`, `Union`, `Tuple`, `List`, `Dict`, `Callable`, and `Any` come from `typing`. Keep `Union` and `Optional`. Do not rewrite them to `X | Y`.
- Concrete arrays are `np.ndarray`. Inputs that the body passes through `np.asarray` and that accept lists are `ArrayLike` from `numpy.typing`.
- Use `NDArray` only in a file that already imports it.
- `self` and `cls` stay unannotated. `__init__` returns `None`.
- `*args` and `**kwargs` that forward an open call are `Any`. Annotate a fixed signature parameter by parameter.
- Heavy imports (`pandas`, Matplotlib types) stay under `TYPE_CHECKING`, as in `speckit/analysis.py`.
- If the module needs postponed annotations, put `from __future__ import annotations` after the BSD header and before the other imports.

`Any` is acceptable for a genuinely open value. Narrow it when the body only accepts one shape.

## Numba

Files with `@njit`, `@jit`, or `numba.jit` must postpone annotations before any kernel signature is annotated. Numba reads evaluated annotations as a signature.

Annotate a kernel and its `_np` fallback with the same types. Do not add an explicit Numba signature string.

After editing a jitted function, import the module. If Numba rejects the annotation, remove it from the jitted function and report that.

## Steps

1. **Read the scope.** List missing annotations and annotations that disagree with the body (optional value typed as required, array-like typed as `ndarray`, one return typed as a tuple).
2. **Check, without editing `pyproject.toml`.**

```bash
ruff check <paths> --select ANN --target-version py39
```

ANN is a list of candidates. The package does not select ANN in `pyproject.toml`. Do not turn that on here. Skip local closures, and skip a jitted function when an annotation would become a Numba signature.

3. **Edit signatures and the imports those signatures need.** Leave docstrings to `numpy-docstrings`. Leave formulas, defaults, and branches alone.
4. **Check the edit.** `ruff check` on the touched files with the repo config (no `--select ANN`). Import a module whose jitted functions changed.

## Output

```markdown
## Type Annotations

**Scope:** …

| Symbol | Change |
|--------|--------|
| `function` | parameter and return types added, or left unchanged and why |

### Check
- `ruff check` result
- Import or Numba result, when a kernel was touched

### Left unannotated
- …
```

## Constraints

- Do not change runtime behavior to make an annotation easier.
- Do not add `mypy` or `pyright`.
- Do not commit unless the user asks.
