# PGF source and test organization

The constant-rate PGF implementation is split under `src/pgf/` and included by
`src/pgf.jl` in dependency order.

- `primitives.jl`: closed-form constant-rate `bd_coefficients`, `gamma_bd`/`γ`,
  `alpha_bd`/`α`, `beta_bd`/`β`, and `pn_birthdeath`/`pₙ`.
- `series.jl`: truncated power-series arithmetic and
  `constant_rate_pgf_series`.
- `original_process.jl`: original-process joint `N,S` probabilities,
  marginals, diagnostics, and truncation/tail utilities.
- `reconstructed.jl`: unsampled probability, transformed rates, raw
  reconstructed `α/β/γ` wrappers, `reconstructed_pgf`, `reconstructed_xi`,
  `reconstructed_eta`, and raw reconstructed count PMFs.
- `conditioned.jl`: conditioned reconstructed scalar wrappers and count PMFs.
- `likelihood_helpers.jl`: grouped sampling-time filtering kernels,
  `grouped_sampling_time_likelihood`, and terminal sampling-time likelihoods.
- `joint_probabilities.jl`: raw and conditioned reconstructed PGF series,
  joint `A,S` PMFs/tables, marginals, tails, and truncation helpers.

Future likelihood-forward-algorithm code should live in a separate source file
under `src/pgf/` or in its own likelihood module, depending on whether it shares
the PGF kernels directly or grows into a larger inference surface. Keep the
current PGF primitives stable and add new wrappers beside the smallest existing
dependency they need.

The PGF and reconstructed-process machinery is protected by the split test
files `test_pgf_primitives.jl`, `test_original_process_probabilities.jl`,
`test_ode_invariants.jl`, `test_reconstructed_process.jl`,
`test_conditioned_kernels.jl`, `test_reconstructed_ode_invariants.jl`, and
`test_stress.jl`. Simulation-facing validation lives in
`test_original_process_validation.jl` and `test_reconstructed_simulation.jl`.
`test_sampling_time_likelihood.jl` keeps direct coefficient, brute-force
latent-count, and simulation-backed checks for terminal sampling-time
likelihoods separate from the conditioned kernel tests.

For manual research/development validation, run
`julia --project=. scripts/validation/sampling_time_likelihood_validation.jl`.
That script compares `sampling_time_likelihood` with an independent
latent-count brute-force calculation for one and two grouped serial sampling
times, fixed-seed `simulate_bd` Monte Carlo estimates, Monte Carlo standard
errors and approximate confidence intervals, and `diagnostics=true` summaries.
Increase the simulation budget with
`BDUTILS_SAMPLING_TIME_VALIDATION_NSIMS=<n>`.

## Sampling-time likelihood API notes

`sampling_time_likelihood` is the intended public entry point for the new
forward-filtering likelihood. It computes a constant-rate, removal-sampling,
single-initial-lineage likelihood conditioned on `A_t0^ℓ = 1`. Serial sampling
times are unique grouped checkpoints in `(t0,tℓ)`, with unlabelled grouped
samples by default and `labelled_samples=true` switching the grouped jump from
`binomial(b,c)` to the falling factorial `(b)_c`. A zero-count checkpoint is an
observed no-sample time and therefore constrains the path; it is not identical
to omitting that time.

With `terminal_sampling=true`, `terminal_count` is the observed count at
exactly `tℓ`, including the present-day Bernoulli sampling probability `ρ₀`
through the reconstruction and terminal count transition. With
`terminal_sampling=false`, `terminal_count` must be zero; `terminal_condition`
then controls whether the final interval is censored for no samples or whether
the final state is summed without conditioning.

`no_sample_probability_conditioned` and `terminal_count_transition` are
currently exported because tests and downstream validation code inspect these
kernels directly. They are lower-level helpers rather than the preferred user
API. If there is no downstream dependence, consider making them internal in a
future breaking-change window. `sampling_time_likelihood` should remain
exported.
