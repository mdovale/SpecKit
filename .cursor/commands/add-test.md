# Add Test

## Objective

Add a pytest that locks a SpecKit behavior, then run it until it passes for the right reason.

## Layout

- `tests/unit/` — one module’s math or API: schedulers, noise, dsp, systems, utils
- `tests/integration/` — `compute_spectrum` accuracy, error bars, and the public result object
- `tests/conftest.py` — seeded series (`reproducible_rng`, `short_white_noise_data`, `long_white_noise_data`, `siso_data`)

Prefer extending the existing file for that module. Integration tests are the place for a claim about PSD level, coherence, or analytic vs empirical error.

## Steps

1. **Name the claim.** If the user did not specify the behavior, ask once.
2. **Write the test.**
   - Name: `test_<behavior>_<expected_outcome>`
   - One claim per test when practical
   - Seed with `np.random.default_rng` or a fixture. Set `N`, `fs`, and the expected formula in the test docstring
   - Use `np.testing.assert_allclose` or `pytest.approx` with a tolerance tied to sample size or a known analytic result
   - For a kernel change, assert Numba and NumPy helpers agree when both exist
3. **Run the narrowest test first.**

```bash
python -m pytest tests/unit/test_noise.py::test_white_noise_statistics -q
```

4. If it fails because the production code is wrong, fix the code. If it fails because the expected value was guessed, correct the expectation from the formula. Do not widen the tolerance to hide a bias (`.cursor/rules/no-hacky-workarounds.mdc`).

## Output

- The test name and the claim it locks
- The pytest command and pass/fail
- Any production fix required to make the claim true
