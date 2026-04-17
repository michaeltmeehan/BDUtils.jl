# TreePar Validation Summary

This note summarizes the current validation status against TreePar. It is a
validation record only; it does not define public API.

## Helper equivalence

In the present-day incomplete-sampling regime, the helper-level mapping is:

```text
With `pars = ConstantRateBDParameters(lambda, mu, 0.0, r, rho0)`:

BDUtils E_constant(t, pars) == TreePar p0(t, lambda, mu, rho)
rho0 * exp(BDUtils g_constant(t, pars)) == TreePar p1(t, lambda, mu, rho)
```

The validation harness checks this mapping across short, moderate, and larger
times, including supercritical, subcritical, near-critical, and `mu = 0`
parameter settings.

The same harness also checks the boundary conditions:

```text
E_constant(0, ...) == 1 - rho0
rho0 * exp(g_constant(0, ...)) == rho0
```

and finite-difference ODE identities:

```text
dE/dt = -(lambda + mu + psi)E + lambda E^2 + mu
dg/dt = -(lambda + mu + psi) + 2lambda E
```

## Restricted full-tree comparison

For the restricted regime:

- `r = 1`
- no `SampledUnary` nodes
- only `Root`, `Binary`, and `SampledLeaf` nodes
- hand-built small trees
- explicit time-before-present conversion, `tau = Tfinal - node.time`

the validation harness contains a TreePar-aligned likelihood assembly that
matches TreePar `LikShiftsSTT` to roundoff on the existing balanced 2-tip and
ladder 3-tip trees.

This assembly is validation-only. It mirrors TreePar's observed conventions:

- `Binary` events are TreePar transmissions
- `SampledLeaf` events are TreePar sampling events
- TreePar uses `log(2lambda)` for transmission events
- TreePar applies root/survival normalization and topology-dependent `log(2)`
  corrections
- for `root = 1`, TreePar appends an additional transmission at
  `max(transmission)`

## Current BDUtils likelihood convention

The current public `bd_loglikelihood_constant` intentionally remains unchanged.
In the same restricted full-tree cases, it does not raw-match TreePar
`LikShiftsSTT`. This is now an expected and tested distinction.

The mismatch is due to root/event/conditioning conventions, not helper-function
failure. Symbolic details are recorded in
`validation/LIKSHIFTSTT_SMALL_TREE_DERIVATIONS.md`.

## Design direction

Future public support for alternative conditioning conventions should be built
around a structured conditioning abstraction and a base tree-likelihood
assembly. The MacPherson appendix frames conditioning as a multiplicative
factor applied to a base likelihood; that direction is preferable to adding
ad hoc root/survival booleans to the public API.
