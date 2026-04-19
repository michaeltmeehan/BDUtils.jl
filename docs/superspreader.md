# Superspreader Parameterisation

Up: [`index.md`](index.md)

The superspreader layer is a constrained reparameterisation of the native
uncoloured MTBD-2 model documented in [`uncoloured_mtbd2.md`](uncoloured_mtbd2.md).
It does not define a new likelihood. Superspreader parameters are mapped into
`UncolouredMTBD2ConstantParameters`, and scoring then uses the existing
uncoloured MTBD-2 latent-state likelihood.

## Parameters

`UncolouredMTBD2SuperspreaderParameters` stores:

- `total_R0`: target total reproduction scale across the two latent types.
- `superspreader_fraction`: fraction `q` of new infections assigned to type 2.
- `relative_transmissibility`: type-2 transmissibility multiplier relative to
  type 1.
- `death[1]`, `death[2]`: unobserved death rates for types 1 and 2.
- `sampling[1]`, `sampling[2]`: sampling rates for types 1 and 2.
- `removal_probability[1]`, `removal_probability[2]`: probabilities that
  sampling removes a lineage of each type.
- `ρ₀[1]`, `ρ₀[2]`: present-day sampling probabilities for each type.

The exported parameter order is recorded in
`UNCOLOURED_MTBD2_SUPERSPREADER_PARAMETER_ORDER`.

## Mapping To Native MTBD-2

Use `uncoloured_mtbd2_native_parameters(pars)` to map superspreader coordinates
to native `UncolouredMTBD2ConstantParameters`.

The implemented mapping is:

1. Let `q = superspreader_fraction`.
2. Let the child-type probabilities be `p = (1 - q, q)`.
3. Let relative transmissibility be `τ = (1, relative_transmissibility)`.
4. Let `δᵢ = death[i] + sampling[i] * removal_probability[i]`.
5. Let `c = total_R0 / ((1 - q) * τ₁ + q * τ₂)`.
6. Let parent-type total birth rates be `λᵢ = c * τᵢ * δᵢ`.
7. Set `birth[i,j] = λᵢ * p[j]`.

This produces the native birth matrix, keeps the supplied death, sampling,
removal-probability, and `ρ₀` vectors, and fixes the native transition matrix to
zero.

## Structural Constraints

This layer is intentionally narrow:

- it is two-type only;
- anagenetic transitions are fixed to zero;
- child type probabilities are independent of parent type;
- the birth matrix has the constrained form `birth[i,j] = λᵢ * p[j]`;
- it maps into native MTBD-2 rather than replacing it.

It should not be treated as a fully general heterogeneity model.

## Likelihood Usage

Superspreader scoring wrappers call the native uncoloured MTBD-2 likelihood
after applying the mapping:

- `loglikelihood_uncoloured_mtbd2_superspreader`
- `likelihood_uncoloured_mtbd2_superspreader`
- `total_loglikelihood_uncoloured_mtbd2_superspreader`

Fitting wrappers also exist:

- `UncolouredMTBD2SuperspreaderSpec`
- `fit_uncoloured_mtbd2_superspreader_mle`

These wrappers optimise superspreader coordinates, but the likelihood being
evaluated is still the native uncoloured MTBD-2 likelihood after mapping.

## Identifiability And Limitations

The superspreader coordinates can be weakly identified.
`superspreader_fraction` and `relative_transmissibility` can trade off because
both affect the mixture of child types and the relative birth scale. Unknown or
weak sampled-node state observations can make this tradeoff stronger.

Use diagnostics to inspect local likelihood shape before interpreting fitted
coordinates. Do not read a fitted superspreader fraction or relative
transmissibility as a formal identifiability result from this package alone.

This layer is useful for constrained scoring and diagnostics. It is not a broad
inference framework and not a general model of transmission heterogeneity.

## Diagnostics

The script
`scripts/uncoloured_mtbd2_superspreader_diagnostics.jl` evaluates likelihood
slices over superspreader coordinates for known, unknown, and mixed sampled-node
observation modes. It is intended as a local diagnostic, not a proof of
identifiability.

## Minimal Example

```julia
using BDUtils
using TreeSim

tree = Tree(
    [0.0, 0.6, 1.0, 1.4],
    [2, 3, 0, 0],
    [0, 4, 0, 0],
    [0, 1, 2, 2],
    [Root, Binary, SampledLeaf, SampledLeaf],
    [0, 0, 0, 0],
    [0, 0, 101, 102],
)

pars = UncolouredMTBD2SuperspreaderParameters(
    1.8,
    0.2,
    6.0,
    [0.15, 0.25],
    [0.5, 0.7],
    [0.6, 0.4],
    [0.1, 0.2],
)

native = uncoloured_mtbd2_native_parameters(pars)
tip_states = Dict(3 => 1, 4 => missing)

ll_native = loglikelihood_uncoloured_mtbd2(
    tree,
    native,
    tip_states;
    root_prior=[0.6, 0.4],
)

ll_super = loglikelihood_uncoloured_mtbd2_superspreader(
    tree,
    pars,
    tip_states;
    root_prior=[0.6, 0.4],
)

println("native log likelihood = ", ll_native)
println("superspreader log likelihood = ", ll_super)
```

The two likelihood values should agree because the superspreader wrapper maps to
native MTBD-2 parameters before scoring.
