# Create Handoff

## Objective

Write one handoff into `docs/handoffs/` for a later session.

## Naming

- `docs/handoffs/YYYYMMDD_<topic-slug>.md`
- If that file exists, use `YYYYMMDDb_`, `YYYYMMDDc_`, and so on

Use today’s date. Slug is lowercase and specific (`noise-generator-normalization`, `ltf-plan-bin-edges`).

## Steps

1. List `docs/handoffs/` and pick a free name.
2. From the conversation and the code, collect the repro (seed, `N`, `fs`, expected vs actual), formulas tried, files, PSD convention, and what must be true when the work is done.
3. Write the file with the required sections in `.cursor/rules/handoff-documents.mdc`.
4. Reply with the path.

## Output

One markdown file. Confirm the path. Do not commit unless the user asks.
