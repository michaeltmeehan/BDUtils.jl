# BDUtils.jl Overview

Up: [`index.md`](index.md)

`BDUtils.jl` is organised around constant-rate birth-death-sampling calculations
and `TreeSim.jl` tree representations. It includes simulation utilities,
analytical count distributions, reconstructed-process formulas, and several
likelihood surfaces. These surfaces share vocabulary but do not all represent
the same observation model.

The documentation is organised into foundations, trees and likelihood,
multitype models, latent-state inference, reparameterisations, and examples. See
[`index.md`](index.md) for the full navigation list.

## Original Vs Reconstructed Process

The original process tracks the full birth-death-sampling process through time.
In this package, `N(t)` is the number of active lineages at time `t`, and `S(t)`
is the number of sampled lineages observed up to that time. Simulation event logs
and original-process analytical helpers work directly with these quantities. See
[`constant_rate_core.md`](constant_rate_core.md) and
[`original_process_counts.md`](original_process_counts.md).

The reconstructed process keeps only lineages that are retained by later
observation in a time window. In this package, `A(t)` is the retained lineage
count at time `t`, usually with respect to an observation horizon `t_k`. This
exists because an observed or reconstructed tree omits extinct or unsampled
side branches from the original process.

Do not treat `(N, S)` and `(A, S)` distributions as interchangeable. Original
process helpers describe the full process; reconstructed helpers describe the
process after conditioning on future observation. See
[`reconstructed_process.md`](reconstructed_process.md).

## Simulation, Analytics, And Likelihood

Simulation functions generate event histories from a model. For example,
`simulate_bd` and `simulate_multitype_bd` produce event logs that can be queried
or converted into trees.

Analytical functions evaluate probabilities, tails, truncations, transformed
rates, and reconstructed-process summaries without simulating. They are useful
for validation, benchmarking, and direct calculation.

Likelihood functions score an observed object under a model. In this package the
observed object may be a `TreeSim.Tree`, a multitype coloured critical-event
tree, or an uncoloured tree with latent type states. Each likelihood surface has
its own admissible input rules. For the single-type tree bridge and likelihood,
see [`constant_rate_trees.md`](constant_rate_trees.md) and
[`constant_rate_tree_likelihood.md`](constant_rate_tree_likelihood.md).

## Coloured Vs Uncoloured Tree Likelihoods

The multitype coloured-tree framework scores observed typed trees. The likelihood
input records segment types and typed critical events such as typed births,
typed transitions, samples, and hidden-birth events from pruning. The colours
are part of the observed input. See
[`multitype_simulation.md`](multitype_simulation.md) and
[`multitype_coloured_trees.md`](multitype_coloured_trees.md).

The uncoloured MTBD-2 framework scores `TreeSim.Tree` objects while marginalising
over latent two-state dynamics. Sampled-node states can be known, unknown, or
partially constrained, but the tree itself is not a fully coloured observed
tree. See [`uncoloured_mtbd2.md`](uncoloured_mtbd2.md).

These are different likelihood semantics. A coloured observed-tree likelihood is
not the same calculation as an uncoloured latent-state likelihood.

## Native MTBD-2 Vs Superspreader Parameterisation

The native uncoloured MTBD-2 layer is parameterised by two-type birth, death,
sampling, removal-probability, transition, and present-day sampling parameters.

The superspreader layer is a constrained mapping into that native MTBD-2
parameterisation. It is not a new likelihood model. Superspreader parameters are
converted to native MTBD-2 parameters before scoring. See
[`superspreader.md`](superspreader.md).

Because the mapping is constrained, fitted superspreader coordinates can be hard
to identify from limited or weakly typed observations.

## Workflow Map

Common workflows are:

1. Simulate a constant-rate history with `simulate_bd` or `simulate_multitype_bd`.
2. Analyse event logs with count queries such as `N_at`, `S_at`, `A_at`, or the
   multitype equivalents.
3. Compare simulation output with analytical original-process or
   reconstructed-process distributions.
4. Extract trees or observed coloured-tree representations from event logs.
5. Score trees with the appropriate likelihood surface.
6. Use the narrow fitting helpers only after the relevant likelihood semantics
   and fixed/free parameter choices are clear.

The safest way to navigate the package is to choose the observation model first:
single-type tree, multitype coloured observed tree, or uncoloured latent-state
MTBD-2 tree.
