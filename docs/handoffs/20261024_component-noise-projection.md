# Handoff: component noise projection

## Scope

Item 7 of the LTPDA feature set in
`docs/handoffs/20261024_ltpda-feature-gaps.md`. Project one-sided input
spectra through transfer functions and sum the output contributions in
power. Coherent cross-terms are out of scope. This note does not cover
`new_ltf_plan`.

## Goal

A function in `speckit/systems.py` that takes one-sided input PSDs and
complex or magnitude transfer functions on the same frequency grid and
returns:

- the one-sided output contribution of each input, `S_out,i(f) = |H_i(f)|² S_in,i(f)`;
- the incoherent sum of those contributions.

Phase between outputs is out of scope. A later time-domain projection that
keeps phase must be a different function and must say so.

## Current status

Not started. `speckit/systems.py` has SISO and MISO optimal subtraction of
measured inputs from a measured output. It does not project a noise budget.
`python-control` is imported in `speckit/analysis.py` for `mag2db` only.

## Context

**LTPDA.** `ltpda/classes/@ssm/projectNoise.m`. The frequency-domain branch
evaluates the state-space Bode response from each noise input to each output
and stores `sqrt(sum |H|²)`. The header says this branch does **not** keep
phase between outputs. The time-domain branch simulates. Commented code in
the frequency branch shows an abandoned one-sided white-noise normalization
(`sqrt(bodeOut*2./(fs*ncols))`). That comment is not the running path. Do
not revive it.

**Owner.** `speckit/systems.py`.

**PSD convention.** One-sided in, one-sided out. Both `S_in` and `S_out` are
power per Hz on positive frequencies, the same convention as
`SpectrumResult.psd`. `|H|²` is dimensionless when `H` is a transfer function
between the same units. If the caller passes an ASD, they convert it. This
function does not take an ASD and silently square it.

**State space.** Do not port `@ssm`. If the caller has a state-space model,
they use `python-control` to obtain `H(f)`, then call this function. Ask
before adding a new model object. Do not grow a second SSM class beside
`python-control`.

**Coherent inputs.** A pair with a cross-spectrum is not accepted here. The
cross-term needs the N-channel matrix
(`docs/handoffs/20261024_nchannel-cross-spectral-matrix.md`). Do not add the
cross-term under a flag that defaults to off.

## Settled decisions

- Incoherent sum only.
- No new model class in this change.
- The abandoned `sqrt(2/fs)` comment in `projectNoise.m` is not a
  normalization to implement.

## Success criteria

Analytic, noise-free, so the tolerance is float64 rather than a spectral
error bar. Record `f`, `H`, and `S_in` in the test.

- One input, `H = 1`: `S_out == S_in`.
- One input, `H = 1 / (1 + 1j * f / f0)`, `S_in` white and one-sided:
  `S_out(f) = |H(f)|² S_in`.
- Two incoherent inputs: the sum is in power, not in amplitude.
  `sqrt(S1) + sqrt(S2)` must not be the result.
- A complex `H` and a real `|H|` with the same modulus give the same `S_out`.
- `ruff check` and `ruff format --check` on `speckit/systems.py` and the new
  test. Existing SISO/MISO tests stay green.

## Recommended next steps

1. Implement the broadcast `|H|² S` and the sum along inputs.
2. Reject a cross-spectral input with a clear error that points at the
   N-channel work.
3. Do not call `MISO_*_optimal_spectral_analysis` from this function. That
   path subtracts measured inputs. This path propagates a budget.

This item is step 7 of the cross-item order.

## References

- `docs/handoffs/20261024_ltpda-feature-gaps.md`
- `docs/handoffs/20261024_nchannel-cross-spectral-matrix.md`
- `speckit/systems.py`
- `ltpda/classes/@ssm/projectNoise.m`
- `docs/paper/main.tex` — optimal spectral analysis is the subtraction
  problem, not this projection
