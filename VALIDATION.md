# Validation Plan

## Purpose

This document defines the validation work required for `BDUtils.jl` to be considered a trustworthy package for birth-death analytical utilities and constant-rate tree likelihoods.

`BDUtils.jl` depends on `TreeSim.jl` and is intended to provide a scientifically reliable analytical layer for currently supported birth-death calculations.

---

## Current scope

`BDUtils.jl` currently supports:

- standalone birth-death analytical utilities
- constant-rate tree likelihood calculations
- use of `TreeSim.jl` as the tree representation layer

---

## Out of scope

The following are explicitly out of scope for the current phase:

- ODE-based likelihood layers
- MLE frameworks
- ensemble fitting layers
- plotting and visualization
- large-scale workflow orchestration
- unsupported conditioning regimes not explicitly documented

---

## Core validation questions

Validation work must establish that:

1. the supported constant-rate analytical formulas are mathematically correct
2. likelihood calculations are numerically stable across the supported domain
3. unsupported inputs fail clearly rather than producing plausible nonsense
4. package assumptions about admissible trees are explicit
5. outputs agree with trusted references, derivations, or legacy validated code

---

## Required validation areas

### 1. Model and conditioning specification

Validation must begin by documenting:

- exactly which birth-death model is implemented
- what conditioning assumptions are used
- what sampling assumptions are used
- what tree forms are admissible
- what is explicitly unsupported

### 2. Reference-value validation

Tests must compare outputs against:

- hand-checked small examples where feasible
- independent derivations
- trusted legacy implementations
- external reference tools where appropriate and feasible

### 3. Numerical stability validation

Tests must probe regimes such as:

- very small rates
- very large rates
- closely spaced event times
- long time depths
- near-boundary parameter values
- parameter combinations likely to trigger underflow, overflow, cancellation, or invalid logarithms

### 4. Domain enforcement

Tests must verify that invalid inputs are rejected clearly, including:

- invalid parameter domains
- unsupported tree topologies
- unsupported conditioning assumptions
- malformed or semantically incompatible trees

### 5. Regression validation

A suite of fixed benchmark cases should be maintained to ensure:

- stable outputs over time
- no accidental drift in formulas
- no silent numerical regressions

---

## Canonical benchmark suite

A benchmark suite should include:

- very small trees with known likelihood values
- representative admissible trees
- edge cases near parameter boundaries
- cases expected to fail
- cases compared against independent references

Each benchmark should document:

- tree used
- parameter values
- expected outcome
- basis for expected outcome

---

## Downstream validation relevance

`BDUtils.jl` is intended to be a scientifically reliable analytical consumer of `TreeSim.jl` trees.

Therefore validation must explicitly define:

- what `TreeSim.jl` guarantees are assumed
- what additional admissibility assumptions are imposed by `BDUtils.jl`
- whether a structurally valid tree may still be analytically inadmissible

---

## Exit criteria for Phase 2

`BDUtils.jl` should not be considered fully validated for this phase until:

- the supported constant-rate model and conditioning are documented
- a benchmark suite of known cases exists
- outputs agree with trusted references on the benchmark suite
- numerical stress tests have been implemented
- invalid inputs fail clearly
- domain restrictions are explicit
- the stable current analytical API is clearly defined

---

## Evidence of successful validation

Evidence should include:

- benchmark tests with documented expected outputs
- numerical stress tests across supported regimes
- domain-failure tests
- comparison notes against reference tools or derivations
- code-level documentation aligning formulas with assumptions
