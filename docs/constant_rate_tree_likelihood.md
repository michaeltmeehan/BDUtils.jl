# Constant-Rate Tree Likelihood

Up: [`index.md`](index.md)

This page documents the current single-type constant-rate tree likelihood
surface. It assumes the simulation and extraction concepts from
[`constant_rate_core.md`](constant_rate_core.md) and
[`constant_rate_trees.md`](constant_rate_trees.md).

Related pages:

- [`constant_rate_core.md`](constant_rate_core.md)
- [`constant_rate_trees.md`](constant_rate_trees.md)
- [`reconstructed_process.md`](reconstructed_process.md)
- [`original_process_counts.md`](original_process_counts.md)

## What `bd_loglikelihood_constant` Scores

`bd_loglikelihood_constant(tree, pars)` computes the log likelihood of an
admissible `TreeSim.Tree` under the currently supported single-type
constant-rate sampled birth-death core.

The parameter object is `ConstantRateBDParameters(λ, μ, ψ, r, ρ₀)`, with:

- `λ > 0`: birth rate.
- `μ >= 0`: death rate.
- `ψ > 0`: sampling rate for likelihood evaluation.
- `r in [0, 1]`: probability that a sampling event removes the sampled lineage.
- `ρ₀ in [0, 1]`: contemporaneous sampling probability used by the helper
  functions in the likelihood assembly.

The implementation uses a root contribution involving `log(1 - E(T))`, where
`T` is the maximum node time in the input tree and `E` is `E_constant`.

## Admissible Input Trees

The input must be a structurally valid `TreeSim.Tree` and must also satisfy the
current analytical likelihood restrictions:

- non-empty tree;
- single reachable root;
- finite node times;
- at least one sampled node;
- node kinds limited to:
  - `TreeSim.Root`
  - `TreeSim.Binary`
  - `TreeSim.SampledLeaf`
  - `TreeSim.SampledUnary`

Unsupported cases include:

- `TreeSim.UnsampledUnary`;
- empty trees;
- root-only trees with no sampled node;
- malformed or unreachable tree structures;
- `ψ == 0` for likelihood evaluation.

> Extraction vs likelihood admissibility: `TreeSim.forest_from_eventlog` returns
> reconstructed trees that collapse `UnsampledUnary` nodes and are closer to the
> current likelihood surface. `full_forest_from_eventlog` may return structurally
> valid trees with `TreeSim.UnsampledUnary`; those are not accepted by
> `bd_loglikelihood_constant`.

## Minimal Example

```julia
using BDUtils
using TreeSim

log = BDEventLog(
    [0.2, 0.8, 1.0],
    [2, 1, 2],
    [1, 0, 0],
    [Birth, SerialSampling, SerialSampling],
    1,
    1.0,
)

tree = TreeSim.tree_from_eventlog(log; tj=0.0, tk=1.0)
pars = ConstantRateBDParameters(1.2, 0.4, 0.6, 0.7)

ll = bd_loglikelihood_constant(tree, pars)
println("log likelihood = ", ll)
```

This example first extracts a reconstructed sampled tree from a deterministic
event log, then scores that tree. Extraction and likelihood evaluation are
separate operations.

## Limitations And Conventions

The likelihood is convention-sensitive. In particular:

- the root contribution is part of the current BDUtils convention;
- binary events contribute under the package's single-type birth convention;
- sampled leaves and sampled ancestors use the package's treatment of removing
  and non-removing sampling;
- node times are interpreted through the `TreeSim.Tree` representation and the
  likelihood's time-before-present calculations.

These choices are part of the likelihood definition. They should be documented
and kept separate from tree extraction, simulation, and empirical count
summaries.

## TreePar Comparisons

> TreePar convention sensitivity: direct raw comparison between
> `bd_loglikelihood_constant` and TreePar likelihood values is not automatic.
> TreePar-style calculations can differ by root handling, survival conditioning,
> event encoding, and factors such as whether transmission events use `λ` or
> `2λ`.

The files under `validation/` record restricted TreePar reconciliation work.
Those notes are validation records, not a guarantee that the public BDUtils
likelihood and a TreePar call will return identical raw values for the same
apparent tree.

Use TreePar comparisons only after matching the conditioning convention, root
encoding, event encoding, and sampling assumptions explicitly.

## Fitting

Fitting wrappers exist for selected constant-rate workflows, but they are not
documented here. This page is only about the tree likelihood input contract and
the convention used by `bd_loglikelihood_constant`.
