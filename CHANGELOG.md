# Changelog

Notable changes to SpecKit. Versions follow semantic versioning. A change to
the numbers a seeded generator or estimator returns is a behavioral change
and gets at least a minor version bump.

## Unreleased

### Changed (behavioral)

- `speckit.noise.alpha_noise` and `pink_noise` now meet their documented
  two-sided level, `S2(f) = f**-alpha`, 1.0 at 1 Hz. The scale now comes from
  the filter cascade's own transfer function instead of `1 / f_max**(alpha/2)`.
  The old output was high by up to 13% in PSD (about 6% in amplitude): about
  13% at `alpha = 1`, 10% at `alpha = 0.5` and `1.5`, and correct at
  `alpha = 2`. For a given seed, realizations change by that factor. The
  spectral shape is unchanged.

### Documentation

- `white_noise`: the docstring RMS is now `sqrt(psd * f_sample)`, matching
  the code. It previously read `sqrt(psd * f_sample / 2)`.
- `speckit.noise` and `compute_spectrum` now state the PSD conventions. Generators
  are two-sided, `compute_spectrum` is one-sided, and `Gxx = 2 * S2` for a
  real signal. Both include a round-trip example.

Details: `docs/handoffs/20260905_noise-generator-normalization.md`.

## 1.0.2

Last release before this changelog was started.
