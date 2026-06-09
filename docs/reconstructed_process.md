# Reconstructed Process

Up: [`index.md`](index.md)

This page documents retained-lineage quantities and analytical helpers for the
single-type reconstructed process. The reconstructed process is derived from
the original birth-death-sampling process by keeping only lineages that lead to
future sampled observations.

For original-process `N(t)` and `S(t)` quantities, see
[`original_process_counts.md`](original_process_counts.md). For simulation
basics, see [`constant_rate_core.md`](constant_rate_core.md).

## Shared Notation

Times are forward process times.

- `t_i`: start time for an analytical interval.
- `t_j`: query time.
- `t_k`: observation horizon, with `t_i <= t_j <= t_k`.
- `N(t)`: number of active lineages in the original process.
- `S(t)`: cumulative number of sampled lineages observed up to time `t`.
- `A(t_j)`: number of lineages active at `t_j` that have at least one sampled
  descendant in `(t_j, t_k]`.

The window `(t_j, t_k]` is strict on the left and closed on the right. A sample
exactly at `t_j` is not a future descendant for `A(t_j)`, while a sample exactly
at `t_k` is included.

> Original vs reconstructed: `N(t_j)` counts all active lineages in the original
> process. `A(t_j)` counts only active lineages that are retained by a later
> sample in `(t_j, t_k]`. A lineage can contribute to `N(t_j)` but not to
> `A(t_j)`.

## Why Reconstruction Is Needed

Observed trees usually omit lineages that leave no sampled descendants. The
reconstructed process describes the lineage process after those unobserved side
branches have been removed. This is why reconstructed quantities depend on the
future observation horizon `t_k`, while original-process quantities at `t_j`
do not.

From simulated logs:

- `retained_lineages_at(log, t_j, t_k)` returns retained lineage ids.
- `A_at(log, t_j, t_k)` returns the retained count.
- `A_over_time(log, times; t_k=...)` evaluates retained counts over times.
- `reconstructed_counts_A` and `reconstructed_pmf_A` summarize `A(t_j)` across
  many logs.
- `reconstructed_joint_counts_AS` and `reconstructed_joint_pmf_AS` summarize
  `(A(t_j), S(t_j))` across logs.

## Unsampled Probability And Transformed Rates

`unsampled_probability(t_j, t_k, pars)` is the probability that one lineage
extant at `t_j` has no sampled descendant in `(t_j, t_k]`.

The transformed-rate helpers describe the effective reconstructed process over
the same observation horizon:

- `transformed_birth_rate(t_j, t_k, pars)`
- `transformed_death_rate(t_j, t_k, pars)`
- `transformed_sampling_rate(t_j, t_k, pars)`
- `reconstructed_effective_rates(t_j, t_k, pars)`

These rates are not the original rates. They are horizon-dependent quantities
after conditioning on future observation.

## Reconstructed PGFs And Count Distributions

The reconstructed PGF helpers describe `A(t_j)` and related sampled-count
quantities from one lineage at `t_i` over the observation horizon `t_k`.

There are two related API prefixes:

| API prefix | Conditioning / interpretation |
| --- | --- |
| `reconstructed_*` | raw thinning of hidden lineage, `G_ij(p_j^k + q_j^k z,w)` |
| `conditioned_reconstructed_*` | conditioned on `A_i^k = 1`, normalized by `q_i^k` |

The older `reconstructed_*` functions are preserved for backward
compatibility. They evaluate the raw thinned hidden-lineage PGF and are not
conditioned on the initial lineage at `t_i` being reconstructed. The newer
`conditioned_reconstructed_*` functions evaluate

```text
Gtilde_ij^k(z,w) = (G_ij(p_j^k + q_j^k z,w) - p_i^k) / q_i^k
```

and represent

```text
E[z^(A_j^k) w^(S_ij) | A_i^k = 1, S_i = 0].
```

Common entry points:

- `reconstructed_pgf(z, w, t_i, t_j, t_k, pars)`
- `reconstructed_pgf_series(smax, t_i, t_j, t_k, pars)`
- `reconstructed_count_pmf(a, t_i, t_j, t_k, pars)`
- `reconstructed_joint_pmf(a, s, t_i, t_j, t_k, pars)`
- `reconstructed_joint_pmf_table(amax, smax, t_i, t_j, t_k, pars)`
- `reconstructed_sampling_marginal_pmf(s, t_i, t_j, t_k, pars)`
- `reconstructed_count_tail` and `reconstructed_sampling_tail`
- `reconstructed_count_truncation` and `reconstructed_sampling_truncation`
- `conditioned_reconstructed_pgf(z, w, t_i, t_j, t_k, pars)`
- `conditioned_reconstructed_pgf_series(smax, t_i, t_j, t_k, pars)`
- `conditioned_reconstructed_count_pmf(a, t_i, t_j, t_k, pars)`
- `conditioned_reconstructed_joint_pmf(a, s, t_i, t_j, t_k, pars)`
- `conditioned_reconstructed_joint_pmf_table(amax, smax, t_i, t_j, t_k, pars)`

The `S(t_j)` component in reconstructed joint helpers is still the cumulative
sample count up to `t_j`; the reconstructed part is the retained lineage count
`A(t_j)`.

For probabilities of the form `P(A_j^k = a | A_i^k = 1)`,
`conditioned_reconstructed_count_pmf(a, t_i, t_j, t_k, pars)` is the preferred
entry point.

## Exact Sampling Times With Terminal Counts

`sampling_time_likelihood(t0, sampling_times, sample_counts, terminal_count,
pars; tℓ)` evaluates grouped exact serial sampling observations together with
an endpoint terminal sampling count. The calculation is conditioned on
`A(t0, tℓ) = 1`. Serial samples are point-time observations in `(t0, tℓ)`,
while terminal sampling is a separate Bernoulli endpoint observation at exactly
`tℓ` with probability `pars.ρ₀` per eligible lineage. Terminal samples should
therefore be supplied through `terminal_count`, not inserted into
`sampling_times`.

The forward filter tracks `A(t)^ell`, the number of reconstructed lineages at
the current time that have sampled descendants by `tℓ`. Between the final
serial observation and `tℓ`, the terminal-count transition marginalizes over
unobserved birth-death histories with no further serial samples and accounts
for the endpoint Bernoulli sampling through the same `ρ₀` convention used by
`unsampled_probability`. In `simulate_bd`, terminal samples generated by
`apply_ρ₀=true` are recorded as `SerialSampling` events at `tmax`, so validation
code counts events at exactly `tℓ` separately from serial samples with
times `< tℓ`.

The manual exact/grouped validation script covers terminal-only, serial plus
terminal, and grouped exact serial plus terminal observations with `ρ₀ > 0`:

```bash
julia --project=. scripts/validation/sampling_time_likelihood_validation.jl
```

## Rounded Or Binned Sampling Times

Rounded sampling dates should be treated as interval-censored observations, not
as exact point-time samples. If a sample is recorded as occurring in year `y`,
the observation is better interpreted as the event that the true continuous
sampling time fell inside the corresponding year bin, rather than as evidence
that the sample occurred exactly at the midpoint of the year.

The binned likelihood is an analytical correction for rounded sampling times. It
replaces exact point-time sampling factors with interval-count probabilities
computed from the `w`-marked reconstructed-process PGF, thereby summing over all
possible continuous sampling times inside each bin. For a bin from `t_i` to
`t_j`, define

```text
Q_{ij}^ell(a_j, m | a_i)
  = [z^a_j w^m] { R_{ij}^ell(z, w) }^a_i
```

where `a_i` is the reconstructed lineage count at the start of the bin, `a_j`
is the reconstructed lineage count at the end of the bin, `m` is the observed
number of serial samples in the bin, and `w` marks serial sampling events. The
forward filter over bin edges is

```text
f_j(a_j) =
    sum_{a_i} f_i(a_i) Q_{ij}^ell(a_j, m_{ij} | a_i)
```

where `m_{ij}` is the observed count in that bin.

The manual validation in
`scripts/validation/binned_sampling_time_likelihood_validation.jl` uses the
conditioned reconstructed process with `A(first_edge, t_ell) = 1`; the Monte
Carlo estimator is conditioned on the same event. It uses the endpoint
convention `(left, right]`, matching the PGF interval convention.

If the final bin edge is before `t_ell`, the validation distinguishes two final
interval conventions:

- `final_interval = :marginalized`: samples after the final bin edge are not
  part of the observation.
- `final_interval = :censored`: no additional samples are allowed after the
  final bin edge and before `t_ell`.

The current validation covers full-removal serial sampling (`r = 1.0`) and
partial-removal serial sampling (`r = 0.5`) with terminal sampling disabled
(`rho = 0.0`). For partial removal, the simulator records sampled-and-removed
events as `SerialSampling` and sampled-but-not-removed events as
`FossilizedSampling`; both are counted as serial sampling events in the binned
sample counts. Terminal sampling with `rho > 0` has not yet been validated in
this binned likelihood script because the final bin can ambiguously contain
both serial samples inside the interval and terminal samples exactly at
`t_ell`.

Run the validation from the package root with:

```bash
julia --project=. scripts/validation/binned_sampling_time_likelihood_validation.jl
```

Optional environment variables are
`BDUTILS_BINNED_VALIDATION_NSIMS`, `BDUTILS_BINNED_VALIDATION_SEED`,
`BDUTILS_BINNED_VALIDATION_ATOL`, and `BDUTILS_BINNED_VALIDATION_MAX_A`.

## Reconstructed Tree Statistics

The package also includes analytical summaries for reconstructed trees:

- `reconstructed_mean_lineages(t, t0, T, pars)`
- `reconstructed_internal_branch_density(ell, s, T, pars)`
- `reconstructed_external_branch_density(ell, s, T, pars)`
- `reconstructed_branch_length_intensity(kind, ell, t0, T, pars)`
- `reconstructed_node_depth_intensity(x, t0, T, pars)`
- `reconstructed_node_depth_density(x, t0, T, pars)`
- `reconstructed_one_tip_probability(t, T, pars)`
- `expected_reconstructed_cherries(t0, T, pars)`
- `reconstructed_tree_stat_counts(tree)`
- `reconstructed_forest_stat_counts(forest)`

These helpers use reconstructed-process conventions. Branch lengths, node
depths, and cherries are summaries of reconstructed trees, not of the full
unpruned event history.

## Example: Simulation-Derived A(t) And Analytical Probability

```julia
using Random
using BDUtils

pars = ConstantRateBDParameters(1.2, 0.5, 0.7, 0.6)
t_i = 0.0
t_j = 0.8
t_k = 2.0

rng = MersenneTwister(20260420)
logs = [simulate_bd(rng, pars, t_k; apply_ρ₀=false) for _ in 1:5_000]

empirical_A = reconstructed_pmf_A(logs, t_j, t_k)
analytic_A0 = reconstructed_count_pmf(0, t_i, t_j, t_k, pars)
analytic_A1 = reconstructed_count_pmf(1, t_i, t_j, t_k, pars)

println("empirical P(A=0) = ", get(empirical_A, 0, 0.0))
println("analytic  P(A=0) = ", analytic_A0)
println("empirical P(A=1) = ", get(empirical_A, 1, 0.0))
println("analytic  P(A=1) = ", analytic_A1)

p_unsampled = unsampled_probability(t_j, t_k, pars)
println("P(one lineage at t_j is unsampled by t_k) = ", p_unsampled)
println("effective rates at t_j = ", reconstructed_effective_rates(t_j, t_k, pars))
```

The empirical `A` distribution is estimated by simulating the original process
to `t_k` and then retaining only lineages with sampled descendants in
`(t_j, t_k]`. The analytical values shown here use the backward-compatible raw
`reconstructed_count_pmf` convention from one lineage at `t_i`; use
`conditioned_reconstructed_count_pmf` when the target probability is conditioned
on `A_i^k = 1`.

## Simulation Validation Scripts

The conditioned reconstructed validation scripts under `scripts/validation/`
are manually runnable checks, not part of the default test suite. Their Monte
Carlo summaries simulate histories to `t_k`, keep only replicates satisfying
the documented conditioning event, and compare the retained empirical
distribution to the conditioned analytical probabilities.

Those scripts default to `apply_ρ₀ = false` because the analytical
probabilities are for sampling over the simulated interval `(t_i, t_k]`, not an
additional terminal present-day sampling pass at `t_k`.
