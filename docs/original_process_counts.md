# Original-Process Counts

Up: [`index.md`](index.md)

This page documents analytical and empirical count tools for the original
single-type constant-rate process. It does not describe reconstructed counts;
for retained-lineage quantities, see
[`reconstructed_process.md`](reconstructed_process.md).

Related pages:

- [`constant_rate_core.md`](constant_rate_core.md)
- [`reconstructed_process.md`](reconstructed_process.md)
- [`overview.md`](overview.md)

## Shared Notation

Times are forward process times.

- `t_i`: start time for an analytical interval.
- `t_j`: query time, with `t_i <= t_j`.
- `N(t)`: number of active lineages at time `t`.
- `S(t)`: cumulative number of sampled lineages observed up to time `t`.

Analytical original-process functions describe the distribution at `t_j` from
one lineage present at `t_i`, with `S(t_i)=0`.

> Original vs reconstructed: this page is about the original process. It tracks
> all simulated lineages through birth, death, and sampling. It does not prune
> away unobserved side branches and it does not condition on future observation.

## Empirical Counts From Simulation

For simulated `BDEventLog` values:

- `NS_at(log, t)` returns `(N=N(t), S=S(t))`.
- `N_at(log, t)` returns `N(t)`.
- `S_at(log, t)` returns `S(t)`.
- `joint_counts_NS(logs, t)` counts `(N(t), S(t))` outcomes across logs.
- `joint_pmf_NS(logs, t)` normalizes those empirical counts.
- `marginal_counts_NS` and `marginal_pmf_NS` split a joint table into `N` and
  `S` marginals.

## Analytical PGF And PMF Helpers

The original-process analytical helpers are based on a probability generating
function for the joint distribution of `N(t_j)` and `S(t_j)`.

Common entry points:

- `constant_rate_pgf_series(smax, t_i, t_j, pars)` returns coefficient series
  used to build probabilities through sampled-count order `smax`.
- `joint_pmf_NS(n, s, t_i, t_j, pars)` returns
  `P(N(t_j)=n, S(t_j)=s)`.
- `joint_pmf_NS_table(nmax, smax, t_i, t_j, pars)` returns a rectangular table
  with rows `n=0:nmax` and columns `s=0:smax`.
- `n_marginal_pmf(n, t_i, t_j, pars)` and
  `s_marginal_pmf(s, t_i, t_j, pars)` return marginal probabilities.

The overloaded name `joint_pmf_NS` is used for both analytical probabilities and
empirical normalization. The argument types determine which method is called.

## Tails And Truncation

Analytical count distributions are infinite in general, so table helpers use
finite cutoffs.

- `n_marginal_tail(nmax, t_i, t_j, pars)` returns omitted mass
  `P(N(t_j) > nmax)`.
- `s_marginal_tail(smax, t_i, t_j, pars)` returns omitted mass
  `P(S(t_j) > smax)`.
- `n_truncation(t_i, t_j, pars; atol=...)` chooses an `nmax` with small
  omitted `N` mass.
- `s_truncation(t_i, t_j, pars; atol=..., max_smax=...)` chooses an `smax` with
  small omitted `S` mass.
- `joint_pmf_NS_table(...; diagnostics=true)` returns the table plus retained
  and omitted mass accounting.

## Example: Simulation Compared With Analytical PMF

```julia
using Random
using BDUtils

pars = ConstantRateBDParameters(1.1, 0.4, 0.5, 0.7)
t_i = 0.0
t_j = 1.0

rng = MersenneTwister(20260420)
logs = [simulate_bd(rng, pars, t_j; apply_ρ₀=false) for _ in 1:5_000]

empirical = joint_pmf_NS(logs, t_j)
analytical_1_0 = joint_pmf_NS(1, 0, t_i, t_j, pars)
analytical_0_1 = joint_pmf_NS(0, 1, t_i, t_j, pars)

println("empirical P(N=1,S=0) = ", get(empirical, (1, 0), 0.0))
println("analytic  P(N=1,S=0) = ", analytical_1_0)
println("empirical P(N=0,S=1) = ", get(empirical, (0, 1), 0.0))
println("analytic  P(N=0,S=1) = ", analytical_0_1)

nmax = n_truncation(t_i, t_j, pars; atol=1e-8)
smax = s_truncation(t_i, t_j, pars; atol=1e-8)
diagnostic = joint_pmf_NS_table(nmax, smax, t_i, t_j, pars; diagnostics=true)

println("table size = ", size(diagnostic.table))
println("retained table mass = ", diagnostic.retained_mass)
println("omitted mass = ", diagnostic.missing_mass)
```

Simulation estimates will vary with the random seed and number of replicates.
The analytical probabilities are for the original process from one lineage at
`t_i` to `t_j`.
