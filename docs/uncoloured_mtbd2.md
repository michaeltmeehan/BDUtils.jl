# Uncoloured MTBD-2 Latent-State Likelihood

Up: [`index.md`](index.md)

This page documents the native uncoloured MTBD-2 likelihood layer. It scores an
uncoloured `TreeSim.Tree` while integrating over latent two-state lineage
histories. It is conceptually separate from the coloured observed-tree
likelihood documented in [`multitype_coloured_trees.md`](multitype_coloured_trees.md).

Related pages:

- [`constant_rate_tree_likelihood.md`](constant_rate_tree_likelihood.md)
- [`multitype_coloured_trees.md`](multitype_coloured_trees.md)
- Reparameterisation notes: [`superspreader.md`](superspreader.md)

## Coloured Vs Uncoloured Likelihood

This distinction is central.

In the coloured observed-tree framework, lineage states are observed and encoded
directly in a `MultitypeColoredTree`. The likelihood scores those observed
colours.

In the uncoloured MTBD-2 framework, the input is an ordinary `TreeSim.Tree`.
Internal lineage states are latent. The likelihood integrates over the possible
two-state histories along branches and at internal nodes.

Sampled-node states may be:

- known;
- unknown;
- partially known by an allowed-state constraint.

Only sampled-node observations are supplied by the user. Internal states are
never observed in this likelihood.

## Model Overview

`UncolouredMTBD2ConstantParameters` stores a constant-rate two-state model:

- `birth[a,b]`: rate at which a latent type-`a` lineage gives birth to a
  type-`b` child.
- `death[a]`: unobserved death rate for type `a`.
- `sampling[a]`: sampling rate for type `a`.
- `removal_probability[a]`: probability that a sampling event removes a type-`a`
  lineage.
- `transition[a,b]`: anagenetic transition rate from latent type `a` to latent
  type `b`.
- `ρ₀[a]`: present-day sampling probability for type `a`.

The layer is currently fixed to two states. Parameter arrays must therefore
have length two or shape `2 x 2`.

## Tree Requirements

The main entry point is:

```julia
loglikelihood_uncoloured_mtbd2(tree, pars, tip_states; root_prior=[0.5, 0.5])
```

The `tree` must be a structurally valid `TreeSim.Tree` with:

- one reachable root;
- finite node times;
- at least one sampled node;
- supported node kinds:
  - `TreeSim.Root`
  - `TreeSim.Binary`
  - `TreeSim.SampledLeaf`
  - `TreeSim.SampledUnary`

Unsupported structures fail with `ArgumentError`. Unsupported cases include
empty trees, malformed trees, trees with no sampled node, and node kinds such as
`TreeSim.UnsampledUnary`.

This admissibility surface is similar to the single-type constant-rate tree
likelihood, but the likelihood semantics are different because states are
latent and two-type.

## Observation Model

`tip_states` must provide an observation for every sampled node:
`TreeSim.SampledLeaf` and `TreeSim.SampledUnary`.

Accepted containers:

- `Dict(node_id => observation)`, keyed by sampled node id.
- A node-indexed vector whose length is at least the largest sampled node id.

Accepted observations:

- `1` or `2`: known sampled-node state.
- `missing`, `nothing`, or `:`: fully unknown state.
- length-2 Boolean mask, such as `[true, false]` or `[true, true]`.
- allowed-state set as a tuple or vector, such as `(1,)`, `(2,)`, `(1, 2)`, or
  `[1, 2]`.

Observations apply only at sampled nodes. Do not provide internal-node states;
they are integrated out by the likelihood recursion.

## Likelihood Structure

At a high level, the likelihood performs a postorder recursion over the tree.

- Along branches, it propagates a two-entry likelihood vector through latent
  birth-death-transition dynamics.
- At binary nodes, it combines the two child likelihood vectors through the
  birth matrix.
- At sampled leaves, it applies the sampled-node observation mask and the
  sampling/removal/present-day sampling boundary.
- At sampled-unary nodes, it treats the node as an observed non-removing
  sampling event on a continuing lineage, then propagates through the child
  branch.
- At the root, it combines the root state likelihood vector with `root_prior`.

The implementation uses numerical ODE propagation controlled by
`steps_per_unit` and `min_steps`.

## Parameterisation Helpers

Native helpers:

- `uncoloured_mtbd2_parameter_vector(pars)` flattens native parameters in
  `UNCOLOURED_MTBD2_PARAMETER_ORDER`.
- `uncoloured_mtbd2_parameters_from_vector(θ)` reconstructs native parameters.
- `UncolouredMTBD2ParameterSpec(fixed; ...)` marks entries as fixed or free for
  free-vector scoring and fitting.
- `free_parameter_vector(...)` extracts free parameters from a spec.
- `uncoloured_mtbd2_parameters_from_free_vector(θ_free, spec)` rebuilds full
  native parameters from free entries and fixed entries.

Transform helpers:

- `uncoloured_mtbd2_unconstrained_from_free(θ_free, spec)` maps positive rates
  through logs and probabilities through logits.
- `uncoloured_mtbd2_free_from_unconstrained(η_free, spec)` maps back to the
  constrained parameter space.

The transformed layer is an interior transform. Free rate parameters must be
positive, and free probability parameters must lie strictly inside `(0, 1)` for
the log/logit transform.

The superspreader layer is a separate reparameterisation that maps onto these
native MTBD-2 parameters. See [`superspreader.md`](superspreader.md).

## Batch Scoring And Fitting

Batch scoring helpers:

- `loglikelihoods_uncoloured_mtbd2(trees, pars, tip_states_list)`
- `likelihoods_uncoloured_mtbd2(trees, pars, tip_states_list)`
- `total_loglikelihood_uncoloured_mtbd2(trees, pars, tip_states_list)`
- `score_uncoloured_mtbd2(trees, pars, tip_states_list)`

Each `tip_states_list[i]` supplies sampled-node observations for `trees[i]`.

Native fitting wrappers:

- `fit_uncoloured_mtbd2_mle`
- `fit_uncoloured_mtbd2_mle_transformed`

These wrappers are narrow optimisation utilities over selected free parameters.
They are not a general inference framework.

Superspreader scoring and fitting wrappers also exist, but they are a
reparameterised route into the native likelihood and are not documented here.

## Example: Known Sampled States

```julia
using BDUtils
using TreeSim

tree = Tree(
    [0.0, 0.6, 1.0, 1.4],
    [2, 3, 0, 0],
    [0, 4, 0, 0],
    [0, 1, 2, 2],
    [Root, Binary, SampledLeaf, SampledLeaf],
    [0, 0, 0, 0],
    [0, 0, 101, 102],
)

pars = UncolouredMTBD2ConstantParameters(
    [0.8 0.35; 0.25 0.7],
    [0.2, 0.3],
    [0.45, 0.55],
    [0.7, 0.4],
    [0.0 0.12; 0.08 0.0],
    [0.2, 0.3],
)

tip_states = Dict(3 => 1, 4 => 2)
ll = loglikelihood_uncoloured_mtbd2(
    tree,
    pars,
    tip_states;
    root_prior=[0.6, 0.4],
)

println("known-state log likelihood = ", ll)
```

Nodes `3` and `4` are sampled leaves, so the observation dictionary provides one
state constraint for each.

## Example: Unknown And Partially Known States

```julia
using BDUtils
using TreeSim

tree = Tree(
    [0.0, 0.6, 1.0, 1.4],
    [2, 3, 0, 0],
    [0, 4, 0, 0],
    [0, 1, 2, 2],
    [Root, Binary, SampledLeaf, SampledLeaf],
    [0, 0, 0, 0],
    [0, 0, 101, 102],
)

pars = UncolouredMTBD2ConstantParameters(
    [0.8 0.35; 0.25 0.7],
    [0.2, 0.3],
    [0.45, 0.55],
    [0.7, 0.4],
    [0.0 0.12; 0.08 0.0],
    [0.2, 0.3],
)

unknown = Dict(3 => missing, 4 => nothing)
partial = Dict(3 => [true, false], 4 => (1, 2))

println(loglikelihood_uncoloured_mtbd2(tree, pars, unknown; root_prior=[0.6, 0.4]))
println(loglikelihood_uncoloured_mtbd2(tree, pars, partial; root_prior=[0.6, 0.4]))
```

The first call integrates over both sampled-node states. The second fixes node
`3` to state `1` by mask and allows either state at node `4`.

## Limitations And Caveats

- The native uncoloured layer is currently MTBD-2: exactly two latent states.
- This likelihood is not the multitype coloured observed-tree likelihood.
- Internal lineage states are latent even when sampled-node states are known.
- Unknown or weak sampled-node observations can make parameters weakly
  identifiable.
- The transformed optimisation layer works on interior coordinates; zero rates
  and boundary probabilities cannot be log/logit transformed as free entries.
- `SampledUnary` nodes are supported as non-removing sampled ancestors, but
  their observation must still be supplied in `tip_states`.
- Superspreader parameters are a reparameterisation of the native MTBD-2 layer,
  not a separate likelihood model.
