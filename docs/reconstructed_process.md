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
quantities from one lineage at `t_i`, conditioned on the observation horizon
`t_k`.

Common entry points:

- `reconstructed_pgf(z, w, t_i, t_j, t_k, pars)`
- `reconstructed_pgf_series(smax, t_i, t_j, t_k, pars)`
- `reconstructed_count_pmf(a, t_i, t_j, t_k, pars)`
- `reconstructed_joint_pmf(a, s, t_i, t_j, t_k, pars)`
- `reconstructed_joint_pmf_table(amax, smax, t_i, t_j, t_k, pars)`
- `reconstructed_sampling_marginal_pmf(s, t_i, t_j, t_k, pars)`
- `reconstructed_count_tail` and `reconstructed_sampling_tail`
- `reconstructed_count_truncation` and `reconstructed_sampling_truncation`

The `S(t_j)` component in reconstructed joint helpers is still the cumulative
sample count up to `t_j`; the reconstructed part is the retained lineage count
`A(t_j)`.

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
`(t_j, t_k]`. The analytical values are for the same reconstructed process from
one lineage at `t_i`.
