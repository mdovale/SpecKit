# No Git Mutations

Hold this constraint for the rest of the session. It stays in force until the user says to lift it, including after other slash commands.

## Allowed

Read-only inspection of the repo:

- `git status`, `git diff`, `git log`, `git show`, `git blame`
- `git rev-parse`, `git branch --show-current`, `git ls-files` (no flags that write)

Editing, creating, and deleting working-tree files is allowed when the task asks for it. Tests and formatters (`pytest`, `ruff`) are allowed.

## Forbidden

Any command that writes the index, refs, config, stash, or a remote. That includes:

- `git add`, `git commit`, `git commit --amend`, `git restore`, `git checkout`, `git switch`, `git reset`, `git revert`, `git clean`
- `git merge`, `git rebase`, `git cherry-pick`, `git stash`
- creating, renaming, or deleting a branch or tag
- `git pull`, `git push`, `git fetch`
- `git config` and any hook skip (`--no-verify`)
- `gh` commands that open, edit, or merge a pull request

Do not stage “just this file,” commit a WIP, or stash to get a clean tree. If a later step would need a git write, stop that step and say the constraint is still on.

## Acknowledgement

On the turn this command is invoked, confirm in one line that git writes are off and that file edits remain allowed. Then continue the user’s task.
