# API Status

Up: [`index.md`](index.md)

The exported API is broader than the documented API. Prefer the documented
surfaces below unless you are working on package internals or validation.

Status labels:

- `stable`: intended as a core public surface within the documented domain.
- `stable but advanced`: intended for public use, but requires careful attention
  to model and time conventions.
- `provisional / experimental`: exported or script-backed, but narrow,
  underdocumented, or not yet a general inference surface.

## Constant-Rate Core

Status: `stable`

Defines `ConstantRateBDParameters`, single-type event logs, event kinds,
simulation with `simulate_bd`, and empirical queries such as `N_at`, `S_at`,
`A_at`, and time-series variants.

Limitations: constant rates only; present-day sampling and sampling-removal
conventions should be chosen explicitly.

## Original-Process Analytics

Status: `stable but advanced`

Provides constant-rate PGF coefficients, `(N, S)` probabilities, marginal PMFs,
tail probabilities, and truncation helpers.

Limitations: describes the original full process, not the reconstructed process.
Some names have both empirical and analytical overloads, so check argument types.

## Reconstructed-Process Analytics

Status: `stable but advanced`

Provides retained-lineage probabilities, transformed reconstructed rates,
reconstructed PGFs, `(A, S)` distributions, truncation helpers, and reconstructed
tree-statistic summaries.

Limitations: depends on observation-window semantics. These helpers should not
be mixed with original-process `(N, S)` formulas without checking the
conditioning.

## Tree Likelihood: Single-Type

Status: `stable but advanced`

Scores admissible `TreeSim.Tree` inputs under a constant-rate single-type
birth-death-sampling likelihood and provides narrow fitting wrappers.

Limitations: unsupported tree topologies fail explicitly. Root, event, and
conditioning conventions differ from some external tools; TreePar comparisons
are validation records, not a promise of raw likelihood equivalence.

## Multitype Coloured Framework

Status: `stable but advanced`

Provides multitype constant-rate parameters, simulation, no-observation
analytics, coloured critical-event trees, pruned event-log extraction, typed
observed-tree likelihoods, and narrow fitting helpers.

Limitations: types are observed on the likelihood input. Hidden-colour
marginalisation, time-varying rates, broad `TreeSim` coloured-tree conversion,
and multi-root observed extraction are not documented public surfaces.

## Uncoloured MTBD-2

Status: `provisional / experimental`

Provides a two-type latent-state likelihood for uncoloured `TreeSim.Tree`
objects, including known, unknown, and partially constrained sampled-node state
observations. Also includes parameter-vector helpers, fixed/free specs,
transforms, batch scoring, and MLE wrappers. See
[`uncoloured_mtbd2.md`](uncoloured_mtbd2.md).

Limitations: two-state layer only. It is semantically separate from the
multitype coloured observed-tree likelihood. Fitting helpers are narrow and
should be treated as validation/scoring tools unless documented otherwise.

## Superspreader Layer

Status: `provisional / experimental`

Provides a constrained superspreader parameterisation that maps onto the native
uncoloured MTBD-2 parameters, with scoring and fitting wrappers. See
[`superspreader.md`](superspreader.md).

Limitations: this is a reparameterisation, not a separate likelihood model. The
coordinates may be weakly identified, especially with unknown sampled-node
states or constrained transition structure.
