# Trust Criteria

## Role of this package

`BDUtils.jl` is the standalone analytical utilities package for birth-death calculations in the recovered outbreak-modelling ecosystem.

Its current role is to provide trustworthy constant-rate analytical utilities and tree likelihood calculations using `TreeSim.jl` as the canonical tree layer.

---

## Trust goal

`BDUtils.jl` is trustworthy when a user can rely on its currently supported constant-rate calculations as scientifically and numerically valid within the documented domain.

---

## What trust does mean here

Trust in this phase means:

- model assumptions are explicit
- conditioning assumptions are explicit
- admissible tree requirements are explicit
- likelihood calculations agree with trusted references or derivations
- supported parameter regimes behave numerically stably
- invalid inputs fail clearly rather than silently producing misleading values

---

## What trust does not mean here

Trust in this phase does **not** imply:

- support for all birth-death model variants
- MLE or inference pipelines
- ODE-based analytical layers
- plotting or workflow tools
- broad ecosystem integration
- support for undocumented conditioning conventions

---

## Stable trust boundary

The current trust boundary includes:

- the currently recovered constant-rate analytical core
- public functions intentionally exposed for that core
- documented use of `TreeSim.jl` trees as inputs

Everything else should be treated as provisional unless explicitly documented as part of the stable API.

---

## Conditions required for trust

### 1. Mathematical clarity

The exact supported model and conditioning regime must be written down.

### 2. Reference agreement

Core calculations must agree with trusted independent benchmarks.

### 3. Numerical reliability

Supported parameter regimes must not silently fail due to instability.

### 4. Domain clarity

The package must clearly define which trees and parameters are admissible.

### 5. Failure safety

Unsupported cases and invalid domains must fail explicitly and intelligibly.

---

## Known trust risks

Current or likely risks include:

- subtle formula errors surviving recovery
- unclear conditioning assumptions
- hidden numerical instability near parameter boundaries
- conflation of structurally valid trees with analytically admissible trees
- silent acceptance of unsupported inputs

---

## Required evidence before calling this package trustworthy

The following evidence is required:

- a written supported-model note
- documented admissibility rules
- benchmark cases with trusted expected outputs
- numerical stress tests
- domain-enforcement tests
- regression tests for fixed reference examples

---

## Phase-2 completion standard

For the purposes of this project phase, `BDUtils.jl` is trustworthy when:

1. its current analytical scope is explicit
2. its formulas are benchmarked against trusted references
3. its numerical behaviour is stress-tested in the supported domain
4. its failure modes are explicit and safe
5. its outputs are reliable enough for research use in the currently supported constant-rate setting
