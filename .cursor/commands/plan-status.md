# Plan Status

## Objective

Score a planning document against the current tree. Report what is done, partial, or not started.

## Steps

1. **Find the plan.** Use the path the user gave. Otherwise look, in order, at a `docs/reference/` note, an open `docs/handoffs/` file, or the feature list in `README.md`. If several could apply, ask which document.
2. **Read its deliverables and exit criteria.** Use the document’s own sections. Do not invent a roadmap the repo does not have.
3. **Check the tree.** Match each item to `speckit/`, `tests/unit/`, `tests/integration/`, `notebooks/`, and `docs/paper/` as appropriate. A README claim without a test or an implementation is partial.
4. **Report** with the document’s own rows:

```markdown
## Plan Status

**Document:** `path`

| Item | Status | Notes |
|------|--------|-------|
| Logarithmic scheduling | ✅ / 🚧 / ⬜ | Where it lives, or what is missing |
| … | … | … |

### Next
- [The next unfinished exit criterion]
```

Use ✅ when the exit criterion is met in code and, where the plan requires it, in tests. Use 🚧 when the module exists but the criterion is open. Use ⬜ when it is absent.

Keep the table to the plan’s items. A one-line note is enough.
