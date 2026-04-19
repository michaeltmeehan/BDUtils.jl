# Constant-Rate Multitype Birth-Death-Sampling

Up: [`index.md`](index.md)

This page has been superseded by:

- [`multitype_simulation.md`](multitype_simulation.md)
- [`multitype_coloured_trees.md`](multitype_coloured_trees.md)

It is retained as an older combined note.

This note documents the current multitype layer in `BDUtils.jl`. It is a
constant-rate, fully typed framework for simulation, observed-tree likelihood
evaluation, narrow scoring/fitting, and validation scripts.

The implementation is deliberately conservative. It does not yet implement
hidden-colour marginalisation, time-varying rates, Bayesian inference, or broad
TreeSim conversion.

## Process Semantics

Types are finite discrete states `1, 2, ..., K`.

`MultitypeBDParameters` stores:

- `birth[a,b]`: rate that an active type-`a` lineage gives birth to a new
  type-`b` child. The parent lineage remains active and remains type `a`.
- `death[a]`: rate that an active type-`a` lineage dies without being sampled.
- `sampling[a]`: rate that an active type-`a` lineage is sampled.
- `removal_probability[a]`: probability that a sampling event removes the
  sampled lineage.
- `transition[a,b]`: anagenetic type-change rate from `a` to `b`. Diagonal
  entries are ignored by simulation/analysis.
- `rho0[a]` (`ρ₀` in code): present-day sampling probability for a type-`a`
  lineage active at `tmax`.

Sampling is always an observation. Sampling-removal affects whether the sampled
lineage continues after observation, but it does not enter the no-observation
probability `E_a(t)`.

## Event Logs

`simulate_multitype_bd` returns a `MultitypeBDEventLog`, a struct-of-arrays log
with event time, lineage id, parent id, event kind, and type-before/type-after.

Event kinds are:

- `MultitypeBirth`: `lineage` is the new child, `parent` is the continuing
  parent lineage, `type_before` is parent type, `type_after` is child type.
- `MultitypeDeath`: `lineage` dies unobserved.
- `MultitypeFossilizedSampling`: the lineage is sampled but not removed.
- `MultitypeSerialSampling`: the lineage is sampled and removed.
- `MultitypeTransition`: the lineage changes from `type_before` to
  `type_after`.

## Analytical Backbone

`multitype_E(t, pars)` evaluates the no-observation probabilities `E_a(t)`.
The current ODE convention matches the simulator:

- a hidden birth `a => b` contributes `birth[a,b] * E_a(t) * E_b(t)`;
- a hidden transition `a => b` contributes `transition[a,b] * E_b(t)`;
- death contributes to no-observation;
- sampling is an observation and therefore contributes as loss.

`multitype_log_flow(t, pars)` returns diagonal no-observed-event segment
transport factors for typed observed segments. These are used by the likelihood
layer.

## Observed Coloured Trees

`MultitypeColoredTree` is the current likelihood input representation. It is a
critical-event representation, not a general tree object.

It contains:

- `segments`: typed no-observed-event intervals in likelihood age time, where
  age `0` is the observation horizon and larger ages go backward in time.
- `births`: observed typed births where both the parent side and child side are
  retained in the observed tree.
- `transitions`: observed anagenetic type changes.
- `hidden_births`: collapsed unary birth events where the observed lineage
  follows the child side and the continuing parent side has no observed
  descendants.
- `terminal_samples`: sampled nodes before or at the horizon.
- `ancestral_samples`: non-removing sampled ancestors with retained observed
  future.
- `present_samples`: sampled lineages at the observation horizon.

`MultitypeColoredHiddenBirth(time, parent_type, child_type)` is not an
anagenetic transition. It represents a birth event under the birth matrix. Its
likelihood contribution is:

```text
birth[parent_type, child_type] * E_parent_type(time)
```

The `E_parent_type(time)` factor integrates over the unobserved continuing
parent side.

## Event-Log Conversion

Two conversion paths are available:

- `multitype_colored_tree_from_eventlog(log)`: narrow direct conversion for
  fully observed histories. It rejects unobserved deaths and unsampled
  survivors.
- `pruned_multitype_colored_tree_from_eventlog(log)`: observed/pruned
  extraction. It removes unobserved dead branches and unsampled survivors, emits
  hidden-birth events for child-only retained births, and returns a
  `MultitypeColoredTree` when the retained observed history is supported.

Both currently support one initial lineage. Multi-root extraction is deferred.

Serial samples at exactly `tmax` are treated as present-day samples by default.
Pass `serial_at_tmax=:terminal` to treat them as terminal serial samples at age
zero.

## Likelihood And Fitting

`multitype_colored_loglikelihood(tree, pars)` evaluates the fully typed
constant-rate observed-tree likelihood. `multitype_loglikelihood` is a scoring
wrapper for one tree or a vector of trees.

`MultitypeMLESpec` and `fit_multitype_mle` provide a narrow MLE-style wrapper.
The first supported fitting surface is intentionally modest:

- fit selected positive entries of `birth`, `death`, `sampling`, and
  off-diagonal `transition`;
- keep zero-rate support fixed;
- keep `removal_probability` and `ρ₀` fixed;
- optimize on log-rate coordinates with simple bound constraints.

This fitting wrapper is suitable for small validation and scoring tasks, not a
general inference framework.

## Current Limitations

The following are intentionally not supported yet:

- hidden-colour marginalisation;
- time-varying multitype rates;
- Bayesian inference;
- model selection over zero/nonzero rate support;
- fitting `removal_probability` or `ρ₀`;
- multi-root observed extraction;
- direct general TreeSim coloured-tree conversion;
- broad end-to-end workflow orchestration.

## Worked Example

Run:

```bash
julia --project=. scripts/multitype/worked_constant_rate_pipeline.jl
```

The script simulates histories, extracts pruned observed coloured trees, scores
truth and initial templates, fits a selected birth-rate entry, and prints a
compact summary.
