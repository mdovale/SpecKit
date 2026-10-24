# Commit Message

## Objective

Inspect the repo and draft a vim-ready commit message. Do not stage or commit.

## Steps

1. Run `git status -sb` and `git diff --cached` (or `git diff` if nothing is staged).
2. Summarize only that diff.
3. Output one plaintext fenced block:

```
type(scope): brief description

- Change 1
- Change 2
```

## Line width

- Subject: one line, about 50 characters, at most 72.
- Body: hard-wrap at 72 characters. Indent wrapped bullet text.

`scope` is a SpecKit area: `analysis`, `core`, `schedulers`, `dsp`, `noise`, `systems`, `plotting`, `flattop`, `utils`, `tests`, `docs`, `benchmarks`.

If the diff continues a handoff, cite `docs/handoffs/<file>.md` in the body.

Follow `.cursor/rules/commit-message.mdc`.
