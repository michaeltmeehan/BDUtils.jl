# Multitype Coloured Observed Trees

Up: [`index.md`](index.md)

This page documents the current multitype coloured observed-tree representation
and likelihood. It builds on the full-history simulator described in
[`multitype_simulation.md`](multitype_simulation.md) and on the single-type
tree/likelihood distinction in [`constant_rate_trees.md`](constant_rate_trees.md)
and [`constant_rate_tree_likelihood.md`](constant_rate_tree_likelihood.md).

Related pages:

- [`constant_rate_core.md`](constant_rate_core.md)
- [`constant_rate_tree_likelihood.md`](constant_rate_tree_likelihood.md)
- [`multitype_simulation.md`](multitype_simulation.md)
- [`uncoloured_mtbd2.md`](uncoloured_mtbd2.md)

## Full History Vs Observed Tree

A `MultitypeBDEventLog` is a full forward history. It may contain unobserved
deaths, unsampled survivors, and branches that leave no sampled descendants.

A `MultitypeColoredTree` is a likelihood input for an observed coloured tree. It
does not store the full event history. It stores typed no-observed-event
segments and observed critical events in likelihood age time, where age `0` is
the observation horizon and larger ages go backward in time.

> Full history vs observed tree: simulate first if you need a process history.
> Convert or prune only when you need an observed-tree likelihood input.

## Coloured Vs Latent-State Likelihood

The coloured-tree likelihood assumes the relevant lineage types are observed and
encoded in the `MultitypeColoredTree`. It does not integrate over unknown
colours.

The uncoloured latent-state likelihood is a different surface documented in
[`uncoloured_mtbd2.md`](uncoloured_mtbd2.md). It marginalises over latent states
on an uncoloured `TreeSim.Tree`. Do not treat these two likelihoods as
interchangeable.

## `MultitypeColoredTree`

`MultitypeColoredTree(origin_time, root_type; ...)` is a critical-event
representation. Its fields include:

- `segments`: `MultitypeColoredSegment(type, start_time, end_time)` intervals
  with no observed event.
- `births`: `MultitypeColoredBirth(time, parent_type, child_type)` where both
  sides of a birth are retained in the observed tree.
- `transitions`: `MultitypeColoredTransition(time, from_type, to_type)` for
  observed anagenetic type changes.
- `hidden_births`: `MultitypeColoredHiddenBirth(time, parent_type, child_type)`
  for pruned child-only retained births.
- `terminal_samples`: `MultitypeColoredSampling(time, type)` for terminal
  sampled nodes before or at the horizon.
- `ancestral_samples`: non-removing sampled ancestors with retained future.
- `present_samples`: type ids sampled at the observation horizon.

`validate_multitype_colored_tree(tree, pars)` checks dimensions, types, times,
and the presence of a root-type segment ending at `origin_time`.

## Direct Conversion

`multitype_colored_tree_from_eventlog(log; serial_at_tmax=:present)` directly
converts a clean fully observed event log.

This path is conservative:

- exactly one initial lineage is supported;
- unobserved death events are rejected;
- active lineages surviving to `tmax` without terminal or present samples are
  rejected;
- serial samples at exactly `tmax` are present samples by default;
- pass `serial_at_tmax=:terminal` to treat those samples as terminal samples at
  age zero.

Use this path for fully observed histories. Use pruning for histories with
unobserved branches.

## Pruning And Hidden Births

`pruned_multitype_colored_tree_from_eventlog(log; serial_at_tmax=:present)`
extracts an observed coloured tree from a full history.

Pruning removes:

- unobserved dead branches;
- unsampled survivors;
- branches with no serial or fossilized sample and no retained sampled
  descendant.

Pruning retains:

- observed sampled lineages;
- ancestors of sampled lineages;
- typed transitions and samples that lie on retained observed ancestry.

When a birth has a retained child side but the continuing parent side has no
observed descendants, the extractor emits
`MultitypeColoredHiddenBirth(time, parent_type, child_type)`.

A hidden birth is still a birth event. It contributes through the birth matrix
and a no-observation factor for the unobserved continuing parent side. It is not
a `MultitypeColoredTransition`, which represents an anagenetic type change along
one lineage.

## Likelihood

`multitype_colored_loglikelihood(tree, pars; steps_per_unit=256, min_steps=16)`
scores an admissible `MultitypeColoredTree` under the constant-rate multitype
coloured observed-tree likelihood. `multitype_colored_likelihood` returns the
exponentiated value.

Admissible inputs are typed critical-event trees whose event types, segment
times, and root type are compatible with `pars`. Positive rates are required for
events that appear in the tree. For example, an observed `birth[parent, child]`
event requires the corresponding birth rate to be positive.

The fitting layer for coloured trees is intentionally narrow and is documented
later. This page covers representation, conversion, pruning, and scoring.

## Example: Simulate, Prune, Score

```julia
using BDUtils
using Random

pars = MultitypeBDParameters(
    [0.0 0.8; 0.0 0.0],
    [0.1, 0.1],
    [0.05, 1.2],
    [1.0, 1.0],
    zeros(2, 2),
    [0.0, 0.0],
)

rng = MersenneTwister(20260420)
log = simulate_multitype_bd(rng, pars, 1.0; initial_types=[1], apply_ρ₀=false)

tree = pruned_multitype_colored_tree_from_eventlog(log)
ll = multitype_colored_loglikelihood(tree, pars)

println(log)
println("segments = ", tree.segments)
println("births = ", tree.births)
println("hidden births = ", tree.hidden_births)
println("terminal samples = ", tree.terminal_samples)
println("log likelihood = ", ll)
```

For this seeded example, the simulated history contains a sampled child lineage
while the continuing parent side is unobserved. Pruning therefore produces a
`MultitypeColoredHiddenBirth`.

## Example: Direct Conversion Of A Fully Observed Log

```julia
using BDUtils

log = MultitypeBDEventLog(
    [0.2, 0.8, 1.0],
    [2, 1, 2],
    [1, 0, 0],
    [MultitypeBirth, MultitypeSerialSampling, MultitypeSerialSampling],
    [1, 1, 2],
    [2, 1, 2],
    [1],
    1.0,
)

tree = multitype_colored_tree_from_eventlog(log; serial_at_tmax=:terminal)

println("segments = ", tree.segments)
println("births = ", tree.births)
println("terminal samples = ", tree.terminal_samples)
```

This direct conversion succeeds because the history has one initial lineage and
no unobserved deaths or unsampled survivors.
