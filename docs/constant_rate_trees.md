# Constant-Rate Trees

Up: [`index.md`](index.md)

This page documents the single-type bridge from `BDEventLog` histories to
`TreeSim.jl` trees and forests. It covers extraction only. Scoring extracted
trees is a separate step documented in
[`constant_rate_tree_likelihood.md`](constant_rate_tree_likelihood.md).

Related pages:

- [`constant_rate_core.md`](constant_rate_core.md)
- [`original_process_counts.md`](original_process_counts.md)
- [`reconstructed_process.md`](reconstructed_process.md)

## Extraction Window

Tree extraction uses the same forward-time retained-lineage convention as the
reconstructed process:

- `t_j`: root/query time for extraction.
- `t_k`: observation horizon.
- The retained window is `(t_j, t_k]`.

Samples exactly at `t_j` are excluded from future retention. Samples exactly at
`t_k` are included. The roots of the extracted forest are the lineages active
immediately after events at `t_j` that have sampled descendants in
`(t_j, t_k]`.

## Event Logs And Trees

A `BDEventLog` is a forward process history. It records births, deaths,
fossilized samples, serial samples, lineage ids, and parent ids. A `TreeSim`
tree extracted from that log is an observed sampled-ancestry object derived
from the event history.

The extraction routines ignore unretained parts of the event log. Death events
do not become tree nodes unless they are on retained sampled ancestry, and
unobserved side branches are either collapsed or represented as unsampled unary
nodes depending on the extraction function.

> Extraction vs likelihood admissibility: tree extraction returns structurally
> valid `TreeSim.Tree` values for the requested extraction convention. A
> structurally valid extracted tree is not automatically admissible for every
> likelihood function. In particular, the full extraction can include
> `TreeSim.UnsampledUnary`, which the current constant-rate likelihood does not
> support.

## Bridge Helpers

The bridge methods are defined as extensions on `TreeSim` plus one exported
BDUtils helper:

- `TreeSim.forest_from_eventlog(log; tj=0.0, tk=log.tmax, validate=true)`
  extracts the reconstructed sampled-ancestry forest. Retained birth events
  with only the child side sampled are collapsed.
- `TreeSim.tree_from_eventlog(log; tj=0.0, tk=log.tmax, validate=true)` returns
  the strict single reconstructed tree. It returns an empty `TreeSim.Tree` if no
  component is retained and errors if more than one retained component exists.
- `TreeSim.reconstructed_tree_from_eventlog` is the explicit name behind
  `TreeSim.tree_from_eventlog`.
- `full_forest_from_eventlog(log; tj=0.0, tk=log.tmax, validate=true)` extracts
  a full sampled-ancestry forest and keeps child-only retained births as
  `TreeSim.UnsampledUnary`.
- `TreeSim.full_tree_from_eventlog(log; tj=0.0, tk=log.tmax, validate=true)`
  returns the strict single full tree, with the same empty/multiple-component
  behaviour as the strict reconstructed tree helper.

Use the forest helpers when multiple retained components are possible. Use the
strict tree helpers only when the chosen window is expected to retain exactly
zero or one component.

## Node Kinds

Extracted single-type trees may contain:

- `TreeSim.Root`: artificial root at `t_j` for a retained lineage.
- `TreeSim.Binary`: retained birth where both sides have sampled descendants.
- `TreeSim.SampledLeaf`: terminal sampled observation.
- `TreeSim.SampledUnary`: non-removing sampled ancestor with retained future.
- `TreeSim.UnsampledUnary`: only in full extraction, for a retained child side
  whose continuing parent side is unobserved.

Reconstructed extraction collapses `TreeSim.UnsampledUnary`; full extraction
keeps it.

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

forest = TreeSim.forest_from_eventlog(log; tj=0.0, tk=1.0)
tree = TreeSim.tree_from_eventlog(log; tj=0.0, tk=1.0)

println("components = ", length(forest))
println("node kinds = ", tree.kind)
println("node times = ", tree.time)
println("retained roots = ", retained_lineages_at(log, 0.0, 1.0))
```

The event log has one birth and two removing samples. The extracted tree has a
root at `t_j`, a binary node at the retained birth, and sampled leaves at the
sample times.

## Reconstructed And Full Extraction

The distinction matters when a birth creates a sampled child but the continuing
parent side has no sampled descendants.

```julia
using BDUtils
using TreeSim

log = BDEventLog(
    [0.2, 0.5, 1.0],
    [2, 1, 2],
    [1, 0, 0],
    [Birth, Death, SerialSampling],
    1,
    1.0,
)

reconstructed = TreeSim.forest_from_eventlog(log; tj=0.0, tk=1.0)
full = full_forest_from_eventlog(log; tj=0.0, tk=1.0)

println("reconstructed kinds = ", only(reconstructed).kind)
println("full kinds = ", only(full).kind)
```

The reconstructed forest collapses the unobserved parent side. The full forest
keeps an `UnsampledUnary` node to record the retained child-only birth.
