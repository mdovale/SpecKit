# Write Reference Doc

## Objective

Create or revise an implementation note in `docs/reference/` that matches the code.

The note records behavior, units, and where the algorithm lives. It does not invent a formula to fill a section.

## Invocation

The user may provide a topic (`noise normalization`, `ltf scheduler`), a path (`docs/reference/noise.md`), and whether to create or update.

If the topic could mean two modules, ask once which one.

## Resolve first

1. Target path: `docs/reference/<kebab-case-topic>.md`
2. The behavior this note owns
3. Whether it is a single algorithm note or a short hub that links to others

## Process

1. Read the owning module and the tests that lock the claim. When they disagree with `README.md`, a notebook, or `docs/paper/`, describe the code and mention the drift.
2. Updating: keep accurate content, drop claims the code no longer implements, and add a superseded line rather than deleting a file (`.cursor/rules/no-delete-docs.mdc`).
3. Creating: use the section order below.
4. Link related handoffs, notebooks, and other reference notes instead of pasting them.

## Sections, in order

1. `# <Title>`
2. One-paragraph intro (what a caller gets)
3. `## Goal`
4. `## Scope` (in / out)
5. `## Behavior and conventions` (shapes, units, one-sided vs two-sided, defaults)
6. `## Algorithm` (the formula or scheduling rule the code uses)
7. `## Key files` (module, tests, notebooks)
8. `## Related docs`

Add `## Numerical checks`, `## Performance`, or `## Known limitations` only when they carry a concrete fact.

## Output

```markdown
## Reference Doc

**Target:** `docs/reference/<file>.md`
**Mode:** create | update

### What changed
- …

### Notes
- …
```

Then write the file.

## Constraints

- Every quantitative claim is visible in `speckit/` or `tests/`.
- Keep the note short enough to maintain.
- Follow `.cursor/rules/project-context.mdc` for sidedness and kernel layout.
