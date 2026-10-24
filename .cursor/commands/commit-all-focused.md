# Commit All (Focused)

Use when the user has asked to commit and the tree holds more than one logical change. Inspect, plan several small commits, then create them.

## Scope

- **In:** tracked edits and untracked files that are not ignored.
- **Out:** `git push`. Build and cache directories, `.venv/`, data products (`*.npy`, `*.npz`, `*.h5`, `*.pkl`), notebook checkpoints, TeX intermediates, and secrets.
- If a path looks wrong to track, say why and leave it unstaged.

## Phase 1 — Inspect

1. `git status -sb`
2. `git diff --cached`, then `git diff`
3. `git ls-files --others --exclude-standard`

If there is nothing to commit, say so and stop.

## Phase 2 — Plan

Before `git add`, write a short plan: order, paths, and one line of rationale per commit.

- One concern per commit.
- Separate docs, tests, and behavior when they can land independently.
- Order so each commit is coherent (helper before the caller that needs it).
- Cite a handoff in the body when a commit continues one.

## Phase 3 — Execute

For each planned commit, follow `.cursor/commands/commit-step.md`:

1. Stage only that commit’s paths.
2. Run the pytest and ruff checks that cover those paths.
3. Write the message in one plaintext block (`.cursor/rules/commit-message.mdc`).
4. `git commit -F - <<'EOF' … EOF`
5. `git status -sb`, then the next commit.

After the last commit, run `python -m pytest -q` if the session changed library behavior and the full suite was not already run. Report the final `git status -sb`.

## If stuck

- Mixed edits in one file: separate them, then stage the file in the commit it belongs to.
- Unsure grouping: an extra small commit is better than a mixed one.
- Use a `wip` subject only when the user wants incomplete work recorded.

## Anti-patterns

- One commit that bundles an unrelated notebook, kernel change, and packaging fix
- A message that describes work that is not staged
- Claiming pytest or ruff passed without running them
