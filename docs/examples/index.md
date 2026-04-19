# Example And Validation Scripts

Up: [`../index.md`](../index.md)

Run scripts from the package root with `julia --project=. path/to/script.jl`.

| Path | Category | Related Docs | Description |
|---|---|---|---|
| `scripts/compare_reconstructed_tree_statistics.jl` | validation | `docs/reconstructed_process.md` | Simulates single-type histories, extracts reconstructed forests, and compares selected reconstructed tree-statistic summaries with analytical quantities. |
| `validation/treepar_compare.jl` | validation | `docs/constant_rate_tree_likelihood.md`, `validation/TREEPAR_VALIDATION_SUMMARY.md` | Compares helper-level and restricted full-tree calculations against TreePar conventions. |
| `validation/treepar_compare.R` | validation | `docs/constant_rate_tree_likelihood.md`, `validation/TREEPAR_VALIDATION_SUMMARY.md` | R wrapper used by the TreePar comparison script. |
| `scripts/uncoloured_mtbd2_known_tips_toy.jl` | example | `docs/uncoloured_mtbd2.md` | Scores small hand-built `TreeSim.Tree` examples under the uncoloured MTBD-2 likelihood with known, unknown, and mixed sampled-node states. |
| `scripts/uncoloured_mtbd2_sim_validation.jl` | validation | `docs/uncoloured_mtbd2.md` | Simulates typed histories, converts admissible outputs to uncoloured trees, and checks that true parameters score better than perturbed parameters in sampled examples. |
| `scripts/uncoloured_mtbd2_superspreader_diagnostics.jl` | diagnostic | `docs/superspreader.md` | Evaluates likelihood slices for the superspreader parameterisation and reports weak-separation diagnostics across observation modes. |
| `scripts/multitype/worked_constant_rate_pipeline.jl` | example | `docs/multitype_simulation.md`, `docs/multitype_coloured_trees.md` | Simulates multitype histories, extracts pruned observed coloured trees, scores truth and initial parameters, and fits one selected birth-rate entry. |
| `scripts/multitype/bridge_eventlog_to_colored_tree.jl` | example | `docs/multitype_coloured_trees.md` | Converts a fully observed multitype event log into a coloured-tree likelihood input and scores it. |
| `scripts/multitype/pruned_eventlog_to_colored_tree.jl` | example | `docs/multitype_coloured_trees.md` | Extracts a pruned observed coloured tree from a multitype event log with unobserved branches. |
| `scripts/multitype/hidden_birth_pruned_eventlog.jl` | example | `docs/multitype_coloured_trees.md` | Demonstrates pruned extraction where a retained child-only branch becomes a hidden-birth likelihood event. |
| `scripts/multitype/fit_multitype_observed_trees.jl` | example | `docs/multitype_coloured_trees.md` | Scores hand-built coloured trees and fits a selected multitype birth-rate entry. |
| `scripts/multitype/validate_multitype_semantics.jl` | validation | `docs/multitype_simulation.md`, `docs/multitype_coloured_trees.md` | Compares simulated no-observation frequencies with `multitype_E` and checks a coloured likelihood factorisation. |
| `scripts/multitype/validate_multitype_fit_pipeline.jl` | validation | `docs/multitype_coloured_trees.md` | Runs small multitype simulation-fitting scenarios and checks that fitted likelihoods improve over initial templates. |

The validation scripts are useful for checking implementation assumptions, but
they should not be read as user-facing API contracts.
