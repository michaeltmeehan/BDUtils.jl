# LikShiftsSTT Small-Tree Derivations

This note derives the exact additive differences between the current
`BDUtils.bd_loglikelihood_constant` assembly and TreePar `LikShiftsSTT` for the
two restricted diagnostic trees used in `validation/treepar_compare.jl`.

This is validation/diagnosis only. It does not propose a change to
`bd_loglikelihood_constant`.

## Shared restricted regime and notation

Restrictions:

- constant rates
- `r = 1`
- no `SampledUnary` nodes
- only `Root`, `Binary`, and `SampledLeaf` nodes
- `tau = Tfinal - node.time`

Definitions:

- `pars = ConstantRateBDParameters(lambda, mu, psi, r, rho0)`
- `E(t) = E_constant(t, pars)`
- `g(t) = g_constant(t, pars)`
- `S(t) = log(1 - E(t))`
- a sampled-leaf contribution at time-before-present `u` is `A(u) = log(psi) - g(u)`

For TreePar `LikShiftsSTT`, the diagnostic uses:

- `par = (lambda, mu + psi)`
- `sprob = psi / (mu + psi)`
- `sampling = rho0`
- `Binary => ttype = 1`
- `SampledLeaf => ttype = 0`

Inspection of TreePar `LikShiftsSTT` gives the constant-rate log-likelihood
accumulator, after converting its returned negative log-likelihood back to a log
likelihood:

- initial root normalization: `-(root + 1) * log(2lambda)`
- if `survival == 1`: additionally `-(root + 1) * S(max(transmission))`
- each transmission event: `g(t) + log(2lambda)`
- each sampling event: `-g(t) + log(psi)`
- topology correction: `-(length(transmission) - 1 - root) * log(2)`
- if `root == 1`, TreePar appends an additional transmission at
  `max(transmission)`

The important point is that TreePar's survival factor is evaluated at
`max(transmission)`, which may be the oldest binary time rather than the
TreeSim root age unless the diagnostic explicitly encodes the root as an added
transmission event.

## Balanced 2-tip tree

TreeSim nodes:

- root age `T`
- one binary event at age `b`
- two sampled leaves at ages `u1` and `u2 = 0`

For the concrete diagnostic tree:

- `T = 1.4`
- `b = 0.8`
- `u1 = 0.4`
- `u2 = 0.0`

### BDUtils expression

Under `r = 1`, the current implementation assembles:

```text
L_BD =
  S(T)
  + [log(lambda) + g(b)]
  + A(u1)
  + A(u2)
```

So:

```text
L_BD = S(T) + log(lambda) + g(b) + A(u1) + A(u2)
```

### TreePar expressions

#### root = 1, survival = 0

TreePar appends a second transmission at `b`.

```text
L_TP =
  -2log(2lambda)
  + 2[g(b) + log(2lambda)]
  + A(u1) + A(u2)
```

Simplified:

```text
L_TP = 2g(b) + A(u1) + A(u2)
```

Difference:

```text
L_BD - L_TP = S(T) + log(lambda) - g(b)
```

For the diagnostic parameters this is about `0.0047`, explaining why this mode
is close but not exactly equal.

#### root = 1, survival = 1

TreePar adds the survival factor at `max(transmission) = b`, not at `T`.

```text
L_TP = 2g(b) + A(u1) + A(u2) - 2S(b)
```

Difference:

```text
L_BD - L_TP = S(T) + log(lambda) - g(b) + 2S(b)
```

This is much farther from BDUtils because the conditioning age and multiplicity
do not match the current BDUtils root term.

#### root = 0, survival = 0, root encoded as transmission

The diagnostic encodes the TreeSim root as an added transmission at age `T`.
TreePar's validity condition for `root = 0` then holds.

```text
L_TP =
  -log(2lambda)
  + [g(T) + log(2lambda)]
  + [g(b) + log(2lambda)]
  + A(u1) + A(u2)
  - log(2)
```

Simplified:

```text
L_TP = g(T) + g(b) + log(lambda) + A(u1) + A(u2)
```

Difference:

```text
L_BD - L_TP = S(T) - g(T)
```

#### root = 0, survival = 1, root encoded as transmission

Now `max(transmission) = T`.

```text
L_TP = g(T) + g(b) + log(lambda) + A(u1) + A(u2) - S(T)
```

Difference:

```text
L_BD - L_TP = 2S(T) - g(T)
```

## Ladder 3-tip tree

TreeSim nodes:

- root age `T`
- two binary events at ages `b1` and `b2`
- three sampled leaves at ages `u1`, `u2`, and `u3 = 0`

For the concrete diagnostic tree:

- `T = 2.0`
- `b1 = 1.6`
- `b2 = 0.8`
- `u1 = 1.1`
- `u2 = 0.3`
- `u3 = 0.0`

### BDUtils expression

```text
L_BD =
  S(T)
  + [log(lambda) + g(b1)]
  + [log(lambda) + g(b2)]
  + A(u1) + A(u2) + A(u3)
```

Simplified:

```text
L_BD = S(T) + 2log(lambda) + g(b1) + g(b2) + A(u1) + A(u2) + A(u3)
```

### TreePar expressions

#### root = 1, survival = 0

TreePar appends an additional transmission at `b1`. There are then three
transmissions, so the topology correction contributes `-log(2)`.

```text
L_TP =
  -2log(2lambda)
  + 2[g(b1) + log(2lambda)]
  + [g(b2) + log(2lambda)]
  + A(u1) + A(u2) + A(u3)
  - log(2)
```

Simplified:

```text
L_TP = log(lambda) + 2g(b1) + g(b2) + A(u1) + A(u2) + A(u3)
```

Difference:

```text
L_BD - L_TP = S(T) + log(lambda) - g(b1)
```

This has the same symbolic form as the balanced-tree `root = 1, survival = 0`
case, with `b` replaced by the oldest binary age `b1`.

#### root = 1, survival = 1

TreePar survival is again evaluated at the oldest transmission age `b1`.

```text
L_TP = log(lambda) + 2g(b1) + g(b2) + A(u1) + A(u2) + A(u3) - 2S(b1)
```

Difference:

```text
L_BD - L_TP = S(T) + log(lambda) - g(b1) + 2S(b1)
```

#### root = 0, survival = 0, root encoded as transmission

There are three transmissions: root at `T`, then `b1`, then `b2`. The topology
correction contributes `-2log(2)`.

```text
L_TP =
  -log(2lambda)
  + [g(T) + log(2lambda)]
  + [g(b1) + log(2lambda)]
  + [g(b2) + log(2lambda)]
  + A(u1) + A(u2) + A(u3)
  - 2log(2)
```

Simplified:

```text
L_TP = 2log(lambda) + g(T) + g(b1) + g(b2) + A(u1) + A(u2) + A(u3)
```

Difference:

```text
L_BD - L_TP = S(T) - g(T)
```

This has the same symbolic form as the balanced-tree `root = 0, survival = 0`
case.

#### root = 0, survival = 1, root encoded as transmission

Now `max(transmission) = T`.

```text
L_TP = 2log(lambda) + g(T) + g(b1) + g(b2) + A(u1) + A(u2) + A(u3) - S(T)
```

Difference:

```text
L_BD - L_TP = 2S(T) - g(T)
```

## Interpretation

The symbolic differences are simple once TreePar's event counting is expanded.
The mismatch is not a helper-function problem, and it is not a universal
`lambda` versus `2lambda` problem.

For `root = 1` TreePar effectively duplicates the oldest transmission event.
That changes the comparison by replacing BDUtils' root-age conditioning
`S(T)` and root split factor with a duplicated oldest-binary propagation
`g(b1)` plus, when `survival = 1`, two survival factors at `b1`.

For `root = 0` with an explicitly encoded root transmission, TreePar replaces
BDUtils' root-conditioning term `S(T)` with a root-propagation term `g(T)`.
With `survival = 1`, TreePar additionally subtracts `S(T)`, giving the
difference `2S(T) - g(T)`.

TreePar also has topology-dependent powers of `log(2)`, but in these restricted
trees those combine with the `log(2lambda)` transmission factors to yield the
simplified expressions above. The remaining difference is therefore best
described as a root/stem/crown conditioning and root-event convention mismatch,
not as a simple missing constant.
