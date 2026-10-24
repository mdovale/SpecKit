# Commit Step

Use at a breakpoint while implementing, after the user has asked to commit. Produces one small commit.

## Steps

1. Run `git status -sb`, `git diff --cached`, and `git diff`. The staged result must be one logical concern. If a file mixes concerns, split the edits and stage explicit paths.
2. Run the checks for this change:
   - `python -m pytest` on the narrowest covering test or file
   - `ruff check` and `ruff format --check` on the touched Python
   Stop if a required check fails.
3. Stage only this commit’s paths: `git add <paths>`.
4. Summarize the staged diff. Do not add files or behavior that are not in it.
5. Output one plaintext fenced block with the full message, per `.cursor/rules/commit-message.mdc`. Cite a handoff path in the body when this commit continues one.
6. Commit that text:

```bash
git commit -F - <<'EOF'
type(scope): subject

- bullet
EOF
```

7. Run `git status -sb` and report what remains unstaged.

## Rules

- Follow `.cursor/rules/git-workflow.mdc`.
- Do not push.
- Do not stage virtualenvs, caches, data products, or TeX intermediates.
