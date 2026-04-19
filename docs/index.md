# BDUtils.jl Documentation

`BDUtils.jl` provides constant-rate birth-death-sampling utilities, analytical
count distributions, reconstructed-process calculations, and tree likelihood
tools built around `TreeSim.jl` tree representations.

Start with [`overview.md`](overview.md) for the conceptual map, then use the
sections below to navigate to the relevant component.

## Foundations

- [`constant_rate_core.md`](constant_rate_core.md): single-type parameters,
  event logs, simulation, and `N(t)`, `S(t)`, `A(t)` queries.
- [`original_process_counts.md`](original_process_counts.md): original-process
  `(N, S)` analytical and empirical count distributions.
- [`reconstructed_process.md`](reconstructed_process.md): retained lineages,
  reconstructed-process distributions, and reconstructed tree statistics.
- [`api_status.md`](api_status.md): current public/provisional status by
  component.

## Trees And Likelihood

- [`constant_rate_trees.md`](constant_rate_trees.md): extraction from
  single-type event logs to `TreeSim` trees and forests.
- [`constant_rate_tree_likelihood.md`](constant_rate_tree_likelihood.md):
  single-type constant-rate tree likelihood conventions and admissibility.

## Multitype Models

- [`multitype_simulation.md`](multitype_simulation.md): multitype parameters,
  simulation, and full event logs.
- [`multitype_coloured_trees.md`](multitype_coloured_trees.md): coloured
  observed-tree representation, pruning, hidden births, and likelihood scoring.
- [`multitype_constant_rate.md`](multitype_constant_rate.md): older combined
  multitype note, retained for continuity.

## Latent-State Inference

- [`uncoloured_mtbd2.md`](uncoloured_mtbd2.md): uncoloured two-state MTBD
  likelihood with latent internal states and sampled-node observations.

## Reparameterisations

- [`superspreader.md`](superspreader.md): constrained superspreader
  reparameterisation of native uncoloured MTBD-2.

## Examples

- [`examples/index.md`](examples/index.md): runnable example, validation, and
  diagnostic scripts.
