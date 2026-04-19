# BDUtils.jl

`BDUtils.jl` provides birth-death utilities for constant-rate simulation,
analytical count distributions, reconstructed-process calculations, and
tree-likelihood scoring on `TreeSim.jl` trees. The package currently contains
single-type constant-rate tools, a multitype coloured observed-tree framework,
an uncoloured two-type latent-state MTBD-2 likelihood layer, and a constrained
superspreader parameterisation.

## Capability Areas

- Constant-rate birth-death-sampling simulation and empirical event-log queries.
- Original-process analytical count distributions for active and sampled
  lineages.
- Reconstructed-process analytics for retained lineages and reconstructed tree
  statistics.
- Constant-rate tree likelihoods for admissible `TreeSim.jl` trees.
- Multitype coloured-tree simulation, extraction, likelihood, and narrow fitting
  helpers.
- Uncoloured MTBD-2 latent-state likelihoods for two-state `TreeSim.jl` trees.
- Superspreader parameterisation mapped onto the native uncoloured MTBD-2 layer.

## Quick Start

From this package directory:

```julia
using Pkg
Pkg.activate(".")

using BDUtils
```

Run the test suite with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Documentation

- Start with the documentation index in [`docs/index.md`](docs/index.md) or the
  conceptual map in [`docs/overview.md`](docs/overview.md).
- Read the single-type foundations in
  [`docs/constant_rate_core.md`](docs/constant_rate_core.md),
  [`docs/original_process_counts.md`](docs/original_process_counts.md), and
  [`docs/reconstructed_process.md`](docs/reconstructed_process.md).
- See tree extraction and likelihood conventions in
  [`docs/constant_rate_trees.md`](docs/constant_rate_trees.md) and
  [`docs/constant_rate_tree_likelihood.md`](docs/constant_rate_tree_likelihood.md).
- Read the multitype simulation and coloured observed-tree notes in
  [`docs/multitype_simulation.md`](docs/multitype_simulation.md) and
  [`docs/multitype_coloured_trees.md`](docs/multitype_coloured_trees.md).
- Read the uncoloured latent-state likelihood notes in
  [`docs/uncoloured_mtbd2.md`](docs/uncoloured_mtbd2.md).
- Read the superspreader reparameterisation notes in
  [`docs/superspreader.md`](docs/superspreader.md).
- Find runnable scripts in [`docs/examples/index.md`](docs/examples/index.md).
- Check maturity notes in [`docs/api_status.md`](docs/api_status.md).

## Current Scope And Limitations

The main implemented surface is constant-rate birth-death-sampling. Time-varying
rates, broad Bayesian inference, and general workflow orchestration are not
documented public surfaces.

The uncoloured MTBD-2 layer is currently a two-type latent-state framework. It is
separate from the multitype coloured observed-tree framework, where types are
observed on the likelihood input.

The superspreader layer is a constrained reparameterisation of the native
uncoloured MTBD-2 parameters. It is useful for diagnostics and narrow fitting
tasks, but may be weakly identified, especially when sampled-node states are
unknown.
