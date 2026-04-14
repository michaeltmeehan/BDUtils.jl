# LikShiftsSTT Reconciliation Notes

These notes describe the restricted full-likelihood diagnostic in
`validation/treepar_compare.jl`. They are validation notes only; they do not
define or change the `BDUtils.jl` public API.

## Restricted regime

The diagnostic currently restricts comparison to:

- constant rates
- `r = 1` in `BDUtils.jl`
- no `SampledUnary` nodes
- hand-built `TreeSim.Tree` inputs with only `Root`, `Binary`, and
  `SampledLeaf` nodes
- explicit conversion to time before present, `tau = Tfinal - node.time`

The TreePar parameter mapping used is:

- `par = (lambda, mu + psi)`
- `sprob = psi / (mu + psi)`
- `sampling = rho0`

TreePar then reconstructs:

- `mu = par[2] * (1 - sprob)`
- `psi = par[2] * sprob`

## Event encoding

For the restricted diagnostic:

- `Binary` nodes are encoded as `ttype = 1`
- `SampledLeaf` nodes are encoded as `ttype = 0`
- `times` are `tau = Tfinal - node.time`

For `root = 1`, TreePar internally appends an additional transmission event at
`max(transmission)`. For `root = 0`, TreePar expects the caller to have added a
root edge; the diagnostic probes this by optionally encoding the `Root` node as
an extra transmission event.

## TreePar terms observed in LikShiftsSTT

Inspection of TreePar `LikShiftsSTT` shows the following term structure in the
constant-rate path:

- `transmission = times[ttype == 1]`
- `sampling = times[ttype == 0]`
- if `root == 1`, TreePar appends `max(transmission)` to `transmission`
- initial normalization includes `-(root + 1) * log(2 * lambda)`
- if `survival == 1`, it also includes
  `-(root + 1) * log(1 - p0(max(transmission)))`
- each transmission contributes `qfuncskylog(time) + log(2 * lambda)`
- each sampling event contributes `-qfuncskylog(time) + log(psi)`
- after event terms, TreePar subtracts
  `(length(transmission) - 1 - root) * log(2)` from its internal negative
  log-likelihood accumulator before returning `-out`

This means raw comparison against `bd_loglikelihood_constant` is sensitive to
at least three convention choices:

- whether binary events use `lambda` or `2 * lambda`
- whether root/stem conditioning uses zero, one, or two survival factors
- whether the root is an observed split, an added stem edge, or an internal
  TreePar correction

## Current interpretation

The diagnostic is intended to isolate additive differences, not to certify full
likelihood equivalence. At the time of writing, the helper functions agree with
TreePar, but the restricted full-tree raw likelihoods still do not match under
the probed root/survival encodings. The remaining mismatch is therefore in
likelihood assembly conventions rather than in `E_constant` or `g_constant`.
