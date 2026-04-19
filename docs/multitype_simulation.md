# Multitype Simulation

Up: [`index.md`](index.md)

This page documents the finite-type constant-rate simulation layer. It extends
the single-type event-log ideas from [`constant_rate_core.md`](constant_rate_core.md)
by assigning each active lineage a discrete type.

Related pages:

- [`constant_rate_core.md`](constant_rate_core.md)
- [`constant_rate_trees.md`](constant_rate_trees.md)
- [`multitype_coloured_trees.md`](multitype_coloured_trees.md)

## Full History, Not Observed Tree

`simulate_multitype_bd` returns a full forward event history. It records all
simulated births, deaths, samples, and type transitions that occur before the
horizon. It is not an observed coloured tree and it is not a likelihood input by
itself.

Observed coloured trees are built later by direct conversion or pruning; see
[`multitype_coloured_trees.md`](multitype_coloured_trees.md).

## Parameters

`MultitypeBDParameters(birth, death, sampling, removal_probability, transition, ρ₀)`
stores a `K`-type constant-rate process. Types are indexed as `1:K`.

- `birth[a,b]`: rate at which an active type-`a` lineage gives birth to a new
  type-`b` child. The parent lineage remains active and remains type `a`.
- `death[a]`: rate at which an active type-`a` lineage dies unobserved.
- `sampling[a]`: rate at which an active type-`a` lineage is sampled.
- `removal_probability[a]`: probability that sampling removes the sampled
  type-`a` lineage.
- `transition[a,b]`: anagenetic type-change rate from `a` to `b`; diagonal
  entries are ignored by simulation.
- `ρ₀[a]`: present-day sampling probability for an active type-`a` lineage at
  the simulation horizon when `apply_ρ₀=true`.

The constructor checks dimensions, non-negative finite rates, and probability
entries in `[0, 1]`.

## Event Kinds

`MultitypeBDEventLog` records:

- `MultitypeBirth`: `lineage` is the new child; `parent` is the continuing
  parent lineage; `type_before` is the parent type; `type_after` is the child
  type.
- `MultitypeDeath`: `lineage` dies unobserved; `type_before == type_after`.
- `MultitypeFossilizedSampling`: `lineage` is sampled but not removed;
  `type_before == type_after`.
- `MultitypeSerialSampling`: `lineage` is sampled and removed;
  `type_before == type_after`.
- `MultitypeTransition`: `lineage` changes type from `type_before` to
  `type_after`.

Birth and transition are distinct events. A birth creates a new lineage and the
parent continues. A transition changes the type of one existing lineage.

## Simulation

Use `simulate_multitype_bd([rng], pars, tmax; initial_types=[1], apply_ρ₀=true)`.

`initial_types` gives one type per initial lineage. For example,
`initial_types=[1, 2]` starts with two active lineages. The current observed-tree
conversion layer is narrower than the simulator and supports one initial lineage
for conversion.

The event log stores:

- `time`: forward event times.
- `lineage`: affected lineage id.
- `parent`: parent lineage for births, or `0` otherwise.
- `kind`: multitype event kind.
- `type_before`: type before the event.
- `type_after`: type after the event.
- `initial_types`: starting types.
- `tmax`: simulation horizon.

## Minimal Example

```julia
using BDUtils
using Random

pars = MultitypeBDParameters(
    [0.0 0.8; 0.0 0.2],
    [0.1, 0.2],
    [0.05, 1.0],
    [1.0, 0.7],
    [0.0 0.3; 0.1 0.0],
    [0.0, 0.2],
)

rng = MersenneTwister(20260420)
log = simulate_multitype_bd(rng, pars, 1.0; initial_types=[1], apply_ρ₀=false)

println(log)
println("events = ", collect(log))
println("N(1.0), S(1.0) by type = ", multitype_NS_at(log, 1.0))
println("validate event log = ", validate_multitype_eventlog(log))
```

The exact events depend on the random seed. The event meanings do not: births
create child lineages, transitions change the type of an existing lineage, and
sampling may or may not remove the sampled lineage depending on the type-specific
removal probability.

## Scope

This page covers simulation and full multitype histories only. The coloured
observed-tree likelihood assumes a pruned or directly converted observed object
with known lineage types. It does not marginalise over unknown colours.
