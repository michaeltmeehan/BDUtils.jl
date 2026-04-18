struct UncolouredMTBD2ConstantParameters{T<:AbstractFloat}
    birth::Matrix{T}
    death::Vector{T}
    sampling::Vector{T}
    removal_probability::Vector{T}
    transition::Matrix{T}
    ρ₀::Vector{T}

    function UncolouredMTBD2ConstantParameters{T}(birth::AbstractMatrix{T},
                                                 death::AbstractVector{T},
                                                 sampling::AbstractVector{T},
                                                 removal_probability::AbstractVector{T},
                                                 transition::AbstractMatrix{T},
                                                 ρ₀::AbstractVector{T}=zeros(T, 2)) where {T<:AbstractFloat}
        _check_uncoloured_mtbd2_parameters(birth, death, sampling, removal_probability, transition, ρ₀)
        return new{T}(Matrix{T}(birth), Vector{T}(death), Vector{T}(sampling),
                      Vector{T}(removal_probability), Matrix{T}(transition), Vector{T}(ρ₀))
    end
end

function UncolouredMTBD2ConstantParameters(birth::AbstractMatrix{<:Real},
                                           death::AbstractVector{<:Real},
                                           sampling::AbstractVector{<:Real},
                                           removal_probability::AbstractVector{<:Real},
                                           transition::AbstractMatrix{<:Real},
                                           ρ₀::AbstractVector{<:Real}=zeros(2))
    T = promote_type(eltype(birth), eltype(death), eltype(sampling),
                     eltype(removal_probability), eltype(transition), eltype(ρ₀), Float64)
    return UncolouredMTBD2ConstantParameters{T}(T.(birth), T.(death), T.(sampling),
                                               T.(removal_probability), T.(transition), T.(ρ₀))
end

Base.length(::UncolouredMTBD2ConstantParameters) = 2

const UNCOLOURED_MTBD2_PARAMETER_ORDER = (
    :birth_11, :birth_12, :birth_21, :birth_22,
    :death_1, :death_2,
    :sampling_1, :sampling_2,
    :removal_probability_1, :removal_probability_2,
    :transition_11, :transition_12, :transition_21, :transition_22,
    :ρ₀_1, :ρ₀_2,
)

struct UncolouredMTBD2ParameterSpec
    fixed::UncolouredMTBD2ConstantParameters{Float64}
    free::NamedTuple{(:birth, :death, :sampling, :removal_probability, :transition, :ρ₀),
                     Tuple{Matrix{Bool},Vector{Bool},Vector{Bool},Vector{Bool},Matrix{Bool},Vector{Bool}}}

    function UncolouredMTBD2ParameterSpec(fixed::UncolouredMTBD2ConstantParameters;
                                          birth=trues(2, 2),
                                          death=trues(2),
                                          sampling=trues(2),
                                          removal_probability=trues(2),
                                          transition=trues(2, 2),
                                          ρ₀=trues(2))
        free = _check_uncoloured_mtbd2_free_masks(; birth, death, sampling, removal_probability, transition, ρ₀)
        fixed64 = UncolouredMTBD2ConstantParameters(fixed.birth, fixed.death, fixed.sampling,
                                                    fixed.removal_probability, fixed.transition, fixed.ρ₀)
        return new(fixed64, free)
    end
end

struct UncolouredMTBD2SuperspreaderParameters{T<:AbstractFloat}
    total_R0::T
    superspreader_fraction::T
    relative_transmissibility::T
    death::Vector{T}
    sampling::Vector{T}
    removal_probability::Vector{T}
    ρ₀::Vector{T}

    function UncolouredMTBD2SuperspreaderParameters{T}(total_R0::T,
                                                       superspreader_fraction::T,
                                                       relative_transmissibility::T,
                                                       death::AbstractVector{T},
                                                       sampling::AbstractVector{T},
                                                       removal_probability::AbstractVector{T},
                                                       ρ₀::AbstractVector{T}=zeros(T, 2)) where {T<:AbstractFloat}
        _check_uncoloured_mtbd2_superspreader_parameters(total_R0, superspreader_fraction,
                                                         relative_transmissibility, death,
                                                         sampling, removal_probability, ρ₀)
        return new{T}(total_R0, superspreader_fraction, relative_transmissibility,
                      Vector{T}(death), Vector{T}(sampling),
                      Vector{T}(removal_probability), Vector{T}(ρ₀))
    end
end

function UncolouredMTBD2SuperspreaderParameters(total_R0::Real,
                                                superspreader_fraction::Real,
                                                relative_transmissibility::Real,
                                                death::AbstractVector{<:Real},
                                                sampling::AbstractVector{<:Real},
                                                removal_probability::AbstractVector{<:Real},
                                                ρ₀::AbstractVector{<:Real}=zeros(2))
    T = promote_type(typeof(total_R0), typeof(superspreader_fraction),
                     typeof(relative_transmissibility), eltype(death), eltype(sampling),
                     eltype(removal_probability), eltype(ρ₀), Float64)
    return UncolouredMTBD2SuperspreaderParameters{T}(
        T(total_R0),
        T(superspreader_fraction),
        T(relative_transmissibility),
        T.(death),
        T.(sampling),
        T.(removal_probability),
        T.(ρ₀),
    )
end

const UNCOLOURED_MTBD2_SUPERSPREADER_PARAMETER_ORDER = (
    :total_R0,
    :superspreader_fraction,
    :relative_transmissibility,
    :death_1, :death_2,
    :sampling_1, :sampling_2,
    :removal_probability_1, :removal_probability_2,
    :ρ₀_1, :ρ₀_2,
)

struct UncolouredMTBD2SuperspreaderSpec
    fixed::UncolouredMTBD2SuperspreaderParameters{Float64}
    free::Vector{Bool}

    function UncolouredMTBD2SuperspreaderSpec(fixed::UncolouredMTBD2SuperspreaderParameters;
                                             total_R0::Bool=true,
                                             superspreader_fraction::Bool=true,
                                             relative_transmissibility::Bool=true,
                                             death=trues(2),
                                             sampling=trues(2),
                                             removal_probability=trues(2),
                                             ρ₀=trues(2))
        length(death) == 2 || throw(ArgumentError("death free mask must have length 2."))
        length(sampling) == 2 || throw(ArgumentError("sampling free mask must have length 2."))
        length(removal_probability) == 2 || throw(ArgumentError("removal_probability free mask must have length 2."))
        length(ρ₀) == 2 || throw(ArgumentError("ρ₀ free mask must have length 2."))
        free = Bool[
            total_R0,
            superspreader_fraction,
            relative_transmissibility,
            death[1], death[2],
            sampling[1], sampling[2],
            removal_probability[1], removal_probability[2],
            ρ₀[1], ρ₀[2],
        ]
        fixed64 = UncolouredMTBD2SuperspreaderParameters(
            fixed.total_R0,
            fixed.superspreader_fraction,
            fixed.relative_transmissibility,
            fixed.death,
            fixed.sampling,
            fixed.removal_probability,
            fixed.ρ₀,
        )
        return new(fixed64, free)
    end
end

function _check_uncoloured_mtbd2_parameters(birth, death, sampling, removal_probability, transition, ρ₀)
    size(birth) == (2, 2) || throw(ArgumentError("birth must be a 2 x 2 matrix."))
    size(transition) == (2, 2) || throw(ArgumentError("transition must be a 2 x 2 matrix."))
    length(death) == 2 || throw(ArgumentError("death must have length 2."))
    length(sampling) == 2 || throw(ArgumentError("sampling must have length 2."))
    length(removal_probability) == 2 || throw(ArgumentError("removal_probability must have length 2."))
    length(ρ₀) == 2 || throw(ArgumentError("ρ₀ must have length 2."))

    for x in birth
        isfinite(x) || throw(ArgumentError("birth rates must be finite."))
        x >= zero(x) || throw(ArgumentError("birth rates must be non-negative."))
    end
    for x in death
        isfinite(x) || throw(ArgumentError("death rates must be finite."))
        x >= zero(x) || throw(ArgumentError("death rates must be non-negative."))
    end
    for x in sampling
        isfinite(x) || throw(ArgumentError("sampling rates must be finite."))
        x >= zero(x) || throw(ArgumentError("sampling rates must be non-negative."))
    end
    for x in transition
        isfinite(x) || throw(ArgumentError("transition rates must be finite."))
        x >= zero(x) || throw(ArgumentError("transition rates must be non-negative."))
    end
    for x in removal_probability
        _check_probability("removal_probability", x)
    end
    for x in ρ₀
        _check_probability("ρ₀", x)
    end
    return nothing
end

function _check_uncoloured_mtbd2_free_masks(; birth, death, sampling, removal_probability, transition, ρ₀)
    size(birth) == (2, 2) || throw(ArgumentError("birth free mask must be a 2 x 2 matrix."))
    size(transition) == (2, 2) || throw(ArgumentError("transition free mask must be a 2 x 2 matrix."))
    length(death) == 2 || throw(ArgumentError("death free mask must have length 2."))
    length(sampling) == 2 || throw(ArgumentError("sampling free mask must have length 2."))
    length(removal_probability) == 2 || throw(ArgumentError("removal_probability free mask must have length 2."))
    length(ρ₀) == 2 || throw(ArgumentError("ρ₀ free mask must have length 2."))
    return (
        birth=Bool.(birth),
        death=Bool.(death),
        sampling=Bool.(sampling),
        removal_probability=Bool.(removal_probability),
        transition=Bool.(transition),
        ρ₀=Bool.(ρ₀),
    )
end

function _check_uncoloured_mtbd2_superspreader_parameters(total_R0, superspreader_fraction,
                                                          relative_transmissibility, death,
                                                          sampling, removal_probability, ρ₀)
    isfinite(total_R0) || throw(ArgumentError("total_R0 must be finite."))
    total_R0 > zero(total_R0) || throw(ArgumentError("total_R0 must be positive."))
    isfinite(superspreader_fraction) || throw(ArgumentError("superspreader_fraction must be finite."))
    zero(superspreader_fraction) < superspreader_fraction < one(superspreader_fraction) ||
        throw(ArgumentError("superspreader_fraction must lie in (0, 1)."))
    isfinite(relative_transmissibility) || throw(ArgumentError("relative_transmissibility must be finite."))
    relative_transmissibility > zero(relative_transmissibility) ||
        throw(ArgumentError("relative_transmissibility must be positive."))
    length(death) == 2 || throw(ArgumentError("death must have length 2."))
    length(sampling) == 2 || throw(ArgumentError("sampling must have length 2."))
    length(removal_probability) == 2 || throw(ArgumentError("removal_probability must have length 2."))
    length(ρ₀) == 2 || throw(ArgumentError("ρ₀ must have length 2."))
    for x in death
        isfinite(x) || throw(ArgumentError("death rates must be finite."))
        x >= zero(x) || throw(ArgumentError("death rates must be non-negative."))
    end
    for x in sampling
        isfinite(x) || throw(ArgumentError("sampling rates must be finite."))
        x >= zero(x) || throw(ArgumentError("sampling rates must be non-negative."))
    end
    for x in removal_probability
        _check_probability("removal_probability", x)
    end
    for x in ρ₀
        _check_probability("ρ₀", x)
    end
    for i in 1:2
        exit_rate = death[i] + sampling[i] * removal_probability[i]
        exit_rate > zero(exit_rate) ||
            throw(ArgumentError("death[i] + sampling[i] * removal_probability[i] must be positive for each type."))
    end
    return nothing
end

"""
    uncoloured_mtbd2_native_parameters(pars::UncolouredMTBD2SuperspreaderParameters)

Map the first narrow superspreader coordinate system into native MTBD-2 rates.

This packet fixes anagenetic transitions to zero. New infections are type 1 with
probability `1 - superspreader_fraction` and type 2 with probability
`superspreader_fraction`, independent of parent type. Type-2 lineages have
`relative_transmissibility` times the transmissibility of type-1 lineages.

Let `δᵢ = death[i] + sampling[i] * removal_probability[i]` be the sampled-removal
exit scale and `τ = (1, relative_transmissibility)`. The parent-type total birth
rates are `λᵢ = c * τᵢ * δᵢ`, where
`c = total_R0 / ((1-q) * τ₁ + q * τ₂)` and `q` is the superspreader fraction.
The native birth matrix is then `birth[i,j] = λᵢ * pⱼ` with `p = (1-q, q)`.
"""
function uncoloured_mtbd2_native_parameters(pars::UncolouredMTBD2SuperspreaderParameters{T}) where {T<:AbstractFloat}
    q = pars.superspreader_fraction
    p = T[one(T) - q, q]
    τ = T[one(T), pars.relative_transmissibility]
    normalizer = p[1] * τ[1] + p[2] * τ[2]
    c = pars.total_R0 / normalizer
    δ = T[
        pars.death[1] + pars.sampling[1] * pars.removal_probability[1],
        pars.death[2] + pars.sampling[2] * pars.removal_probability[2],
    ]
    λ = T[c * τ[1] * δ[1], c * τ[2] * δ[2]]
    birth = T[λ[1] * p[1] λ[1] * p[2];
              λ[2] * p[1] λ[2] * p[2]]
    return UncolouredMTBD2ConstantParameters(birth, pars.death, pars.sampling,
                                             pars.removal_probability, zeros(T, 2, 2), pars.ρ₀)
end

function uncoloured_mtbd2_superspreader_parameter_vector(pars::UncolouredMTBD2SuperspreaderParameters)
    return Float64[
        pars.total_R0,
        pars.superspreader_fraction,
        pars.relative_transmissibility,
        pars.death[1], pars.death[2],
        pars.sampling[1], pars.sampling[2],
        pars.removal_probability[1], pars.removal_probability[2],
        pars.ρ₀[1], pars.ρ₀[2],
    ]
end

function uncoloured_mtbd2_superspreader_parameters_from_vector(θ::AbstractVector{<:Real})
    length(θ) == length(UNCOLOURED_MTBD2_SUPERSPREADER_PARAMETER_ORDER) ||
        throw(ArgumentError("uncoloured MTBD-2 superspreader parameter vector must have length $(length(UNCOLOURED_MTBD2_SUPERSPREADER_PARAMETER_ORDER))."))
    return UncolouredMTBD2SuperspreaderParameters(
        θ[1],
        θ[2],
        θ[3],
        [θ[4], θ[5]],
        [θ[6], θ[7]],
        [θ[8], θ[9]],
        [θ[10], θ[11]],
    )
end

function _uncoloured_mtbd2_superspreader_free_mask_vector(spec::UncolouredMTBD2SuperspreaderSpec)
    return copy(spec.free)
end

function free_parameter_vector(spec::UncolouredMTBD2SuperspreaderSpec)
    θ = uncoloured_mtbd2_superspreader_parameter_vector(spec.fixed)
    return θ[_uncoloured_mtbd2_superspreader_free_mask_vector(spec)]
end

function free_parameter_vector(pars::UncolouredMTBD2SuperspreaderParameters,
                               spec::UncolouredMTBD2SuperspreaderSpec)
    θ = uncoloured_mtbd2_superspreader_parameter_vector(pars)
    return θ[_uncoloured_mtbd2_superspreader_free_mask_vector(spec)]
end

function uncoloured_mtbd2_superspreader_parameters_from_free_vector(θ_free::AbstractVector{<:Real},
                                                                    spec::UncolouredMTBD2SuperspreaderSpec)
    mask = _uncoloured_mtbd2_superspreader_free_mask_vector(spec)
    nfree = count(mask)
    length(θ_free) == nfree ||
        throw(ArgumentError("free superspreader parameter vector has length $(length(θ_free)); expected $nfree."))
    θ = uncoloured_mtbd2_superspreader_parameter_vector(spec.fixed)
    θ[mask] .= Float64.(θ_free)
    return uncoloured_mtbd2_superspreader_parameters_from_vector(θ)
end

const _UNCOLOURED_MTBD2_SUPERSPREADER_POSITIVE_PARAMETER_INDEXES = (1, 3, 4, 5, 6, 7)
const _UNCOLOURED_MTBD2_SUPERSPREADER_PROBABILITY_PARAMETER_INDEXES = (2, 8, 9, 10, 11)

function _uncoloured_mtbd2_superspreader_free_parameter_indexes(spec::UncolouredMTBD2SuperspreaderSpec)
    return findall(_uncoloured_mtbd2_superspreader_free_mask_vector(spec))
end

function _uncoloured_mtbd2_superspreader_unconstrain_value(x::Float64, index::Int)
    if index in _UNCOLOURED_MTBD2_SUPERSPREADER_POSITIVE_PARAMETER_INDEXES
        x > 0 || throw(ArgumentError("free superspreader positive parameters must be positive for log transformation."))
        return log(x)
    elseif index in _UNCOLOURED_MTBD2_SUPERSPREADER_PROBABILITY_PARAMETER_INDEXES
        0 < x < 1 || throw(ArgumentError("free superspreader probability parameters must lie in (0, 1) for logit transformation."))
        return _uncoloured_mtbd2_logit(x)
    end
    throw(ArgumentError("unknown uncoloured MTBD-2 superspreader parameter index $index."))
end

function _uncoloured_mtbd2_superspreader_constrain_value(x::Float64, index::Int)
    if index in _UNCOLOURED_MTBD2_SUPERSPREADER_POSITIVE_PARAMETER_INDEXES
        return exp(x)
    elseif index in _UNCOLOURED_MTBD2_SUPERSPREADER_PROBABILITY_PARAMETER_INDEXES
        return _uncoloured_mtbd2_invlogit(x)
    end
    throw(ArgumentError("unknown uncoloured MTBD-2 superspreader parameter index $index."))
end

function uncoloured_mtbd2_superspreader_unconstrained_from_free(θ_free::AbstractVector{<:Real},
                                                                spec::UncolouredMTBD2SuperspreaderSpec)
    indexes = _uncoloured_mtbd2_superspreader_free_parameter_indexes(spec)
    length(θ_free) == length(indexes) ||
        throw(ArgumentError("free superspreader parameter vector length must match the number of fitted parameters."))
    return [_uncoloured_mtbd2_superspreader_unconstrain_value(Float64(θ_free[i]), indexes[i])
            for i in eachindex(indexes)]
end

function uncoloured_mtbd2_superspreader_free_from_unconstrained(η_free::AbstractVector{<:Real},
                                                                spec::UncolouredMTBD2SuperspreaderSpec)
    indexes = _uncoloured_mtbd2_superspreader_free_parameter_indexes(spec)
    length(η_free) == length(indexes) ||
        throw(ArgumentError("unconstrained superspreader parameter vector length must match the number of fitted parameters."))
    return [_uncoloured_mtbd2_superspreader_constrain_value(Float64(η_free[i]), indexes[i])
            for i in eachindex(indexes)]
end

"""
    uncoloured_mtbd2_parameter_vector(params)

Flatten `UncolouredMTBD2ConstantParameters` into the native 16-entry order:
`birth[1,1]`, `birth[1,2]`, `birth[2,1]`, `birth[2,2]`,
`death[1:2]`, `sampling[1:2]`, `removal_probability[1:2]`,
`transition[1,1]`, `transition[1,2]`, `transition[2,1]`, `transition[2,2]`,
and `ρ₀[1:2]`.
"""
function uncoloured_mtbd2_parameter_vector(pars::UncolouredMTBD2ConstantParameters)
    return Float64[
        pars.birth[1, 1], pars.birth[1, 2], pars.birth[2, 1], pars.birth[2, 2],
        pars.death[1], pars.death[2],
        pars.sampling[1], pars.sampling[2],
        pars.removal_probability[1], pars.removal_probability[2],
        pars.transition[1, 1], pars.transition[1, 2], pars.transition[2, 1], pars.transition[2, 2],
        pars.ρ₀[1], pars.ρ₀[2],
    ]
end

function uncoloured_mtbd2_parameters_from_vector(θ::AbstractVector{<:Real})
    length(θ) == length(UNCOLOURED_MTBD2_PARAMETER_ORDER) ||
        throw(ArgumentError("uncoloured MTBD-2 parameter vector must have length $(length(UNCOLOURED_MTBD2_PARAMETER_ORDER))."))
    return UncolouredMTBD2ConstantParameters(
        [θ[1] θ[2]; θ[3] θ[4]],
        [θ[5], θ[6]],
        [θ[7], θ[8]],
        [θ[9], θ[10]],
        [θ[11] θ[12]; θ[13] θ[14]],
        [θ[15], θ[16]],
    )
end

function _uncoloured_mtbd2_free_mask_vector(spec::UncolouredMTBD2ParameterSpec)
    f = spec.free
    return Bool[
        f.birth[1, 1], f.birth[1, 2], f.birth[2, 1], f.birth[2, 2],
        f.death[1], f.death[2],
        f.sampling[1], f.sampling[2],
        f.removal_probability[1], f.removal_probability[2],
        f.transition[1, 1], f.transition[1, 2], f.transition[2, 1], f.transition[2, 2],
        f.ρ₀[1], f.ρ₀[2],
    ]
end

function free_parameter_vector(spec::UncolouredMTBD2ParameterSpec)
    θ = uncoloured_mtbd2_parameter_vector(spec.fixed)
    return θ[_uncoloured_mtbd2_free_mask_vector(spec)]
end

function free_parameter_vector(pars::UncolouredMTBD2ConstantParameters,
                               spec::UncolouredMTBD2ParameterSpec)
    θ = uncoloured_mtbd2_parameter_vector(pars)
    return θ[_uncoloured_mtbd2_free_mask_vector(spec)]
end

function uncoloured_mtbd2_parameters_from_free_vector(θ_free::AbstractVector{<:Real},
                                                      spec::UncolouredMTBD2ParameterSpec)
    mask = _uncoloured_mtbd2_free_mask_vector(spec)
    nfree = count(mask)
    length(θ_free) == nfree ||
        throw(ArgumentError("free parameter vector has length $(length(θ_free)); expected $nfree."))
    θ = uncoloured_mtbd2_parameter_vector(spec.fixed)
    θ[mask] .= Float64.(θ_free)
    return uncoloured_mtbd2_parameters_from_vector(θ)
end

const _UNCOLOURED_MTBD2_RATE_PARAMETER_INDEXES = (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14)
const _UNCOLOURED_MTBD2_PROBABILITY_PARAMETER_INDEXES = (9, 10, 15, 16)

@inline _uncoloured_mtbd2_logit(p::Float64) = log(p / (1 - p))
@inline _uncoloured_mtbd2_invlogit(x::Float64) = inv(1 + exp(-x))

function _uncoloured_mtbd2_free_parameter_indexes(spec::UncolouredMTBD2ParameterSpec)
    return findall(_uncoloured_mtbd2_free_mask_vector(spec))
end

function _uncoloured_mtbd2_unconstrain_value(x::Float64, native_index::Int)
    if native_index in _UNCOLOURED_MTBD2_RATE_PARAMETER_INDEXES
        x > 0 || throw(ArgumentError("free rate parameters must be positive for log transformation."))
        return log(x)
    elseif native_index in _UNCOLOURED_MTBD2_PROBABILITY_PARAMETER_INDEXES
        0 < x < 1 || throw(ArgumentError("free probability parameters must lie in (0, 1) for logit transformation."))
        return _uncoloured_mtbd2_logit(x)
    end
    throw(ArgumentError("unknown uncoloured MTBD-2 parameter index $native_index."))
end

function _uncoloured_mtbd2_constrain_value(x::Float64, native_index::Int)
    if native_index in _UNCOLOURED_MTBD2_RATE_PARAMETER_INDEXES
        return exp(x)
    elseif native_index in _UNCOLOURED_MTBD2_PROBABILITY_PARAMETER_INDEXES
        return _uncoloured_mtbd2_invlogit(x)
    end
    throw(ArgumentError("unknown uncoloured MTBD-2 parameter index $native_index."))
end

function uncoloured_mtbd2_unconstrained_from_free(θ_free::AbstractVector{<:Real},
                                                  spec::UncolouredMTBD2ParameterSpec)
    indexes = _uncoloured_mtbd2_free_parameter_indexes(spec)
    length(θ_free) == length(indexes) ||
        throw(ArgumentError("free parameter vector length must match the number of fitted parameters."))
    return [_uncoloured_mtbd2_unconstrain_value(Float64(θ_free[i]), indexes[i]) for i in eachindex(indexes)]
end

function uncoloured_mtbd2_free_from_unconstrained(η_free::AbstractVector{<:Real},
                                                  spec::UncolouredMTBD2ParameterSpec)
    indexes = _uncoloured_mtbd2_free_parameter_indexes(spec)
    length(η_free) == length(indexes) ||
        throw(ArgumentError("unconstrained parameter vector length must match the number of fitted parameters."))
    return [_uncoloured_mtbd2_constrain_value(Float64(η_free[i]), indexes[i]) for i in eachindex(indexes)]
end

@inline function _uncoloured_mtbd2_transition(pars::UncolouredMTBD2ConstantParameters{T}) where {T<:AbstractFloat}
    transition = copy(pars.transition)
    transition[1, 1] = zero(T)
    transition[2, 2] = zero(T)
    return transition
end

function _uncoloured_mtbd2_E_derivative!(dE::AbstractVector{T}, E::AbstractVector{T},
                                         pars::UncolouredMTBD2ConstantParameters{T},
                                         transition::AbstractMatrix{T}) where {T<:AbstractFloat}
    @inbounds for a in 1:2
        birth_out = pars.birth[a, 1] + pars.birth[a, 2]
        transition_out = transition[a, 1] + transition[a, 2]
        hidden_birth = pars.birth[a, 1] * E[a] * E[1] + pars.birth[a, 2] * E[a] * E[2]
        hidden_transition = transition[a, 1] * E[1] + transition[a, 2] * E[2]
        total = birth_out + pars.death[a] + pars.sampling[a] + transition_out
        dE[a] = -total * E[a] + hidden_birth + pars.death[a] + hidden_transition
    end
    return dE
end

function _uncoloured_mtbd2_D_derivative!(dD::AbstractVector{T}, D::AbstractVector{T}, E::AbstractVector{T},
                                         pars::UncolouredMTBD2ConstantParameters{T},
                                         transition::AbstractMatrix{T}) where {T<:AbstractFloat}
    @inbounds for a in 1:2
        birth_out = pars.birth[a, 1] + pars.birth[a, 2]
        transition_out = transition[a, 1] + transition[a, 2]
        total = birth_out + pars.death[a] + pars.sampling[a] + transition_out
        hidden_transition = transition[a, 1] * D[1] + transition[a, 2] * D[2]
        hidden_birth = zero(T)
        for b in 1:2
            hidden_birth += pars.birth[a, b] * (D[a] * E[b] + E[a] * D[b])
        end
        dD[a] = -total * D[a] + hidden_birth + hidden_transition
    end
    return dD
end

function _uncoloured_mtbd2_joint_derivative!(dy::AbstractVector{T}, y::AbstractVector{T},
                                             pars::UncolouredMTBD2ConstantParameters{T},
                                             transition::AbstractMatrix{T}) where {T<:AbstractFloat}
    E = view(y, 1:2)
    D = view(y, 3:4)
    dE = view(dy, 1:2)
    dD = view(dy, 3:4)
    _uncoloured_mtbd2_E_derivative!(dE, E, pars, transition)
    _uncoloured_mtbd2_D_derivative!(dD, D, E, pars, transition)
    return dy
end

function _rk4_uncoloured_mtbd2_joint_step!(y::Vector{T}, dt::T,
                                           pars::UncolouredMTBD2ConstantParameters{T},
                                           transition::Matrix{T}) where {T<:AbstractFloat}
    k1 = similar(y)
    k2 = similar(y)
    k3 = similar(y)
    k4 = similar(y)
    tmp = similar(y)

    _uncoloured_mtbd2_joint_derivative!(k1, y, pars, transition)
    @inbounds for i in eachindex(y)
        tmp[i] = y[i] + dt * k1[i] / 2
    end
    _uncoloured_mtbd2_joint_derivative!(k2, tmp, pars, transition)
    @inbounds for i in eachindex(y)
        tmp[i] = y[i] + dt * k2[i] / 2
    end
    _uncoloured_mtbd2_joint_derivative!(k3, tmp, pars, transition)
    @inbounds for i in eachindex(y)
        tmp[i] = y[i] + dt * k3[i]
    end
    _uncoloured_mtbd2_joint_derivative!(k4, tmp, pars, transition)

    @inbounds for i in eachindex(y)
        y[i] += dt * (k1[i] + 2k2[i] + 2k3[i] + k4[i]) / 6
    end
    y[1] = clamp(y[1], zero(T), one(T))
    y[2] = clamp(y[2], zero(T), one(T))
    y[3] = max(y[3], zero(T))
    y[4] = max(y[4], zero(T))
    return y
end

function _uncoloured_mtbd2_propagate(D0::AbstractVector{T}, age0::T, age1::T,
                                     pars::UncolouredMTBD2ConstantParameters{T};
                                     steps_per_unit::Integer,
                                     min_steps::Integer) where {T<:AbstractFloat}
    age1 >= age0 || throw(ArgumentError("branch propagation ages must be non-decreasing."))
    duration = age1 - age0
    E0 = _uncoloured_mtbd2_E(age0, pars; steps_per_unit=steps_per_unit, min_steps=min_steps)
    y = T[E0[1], E0[2], D0[1], D0[2]]
    duration == zero(T) && return T[y[3], y[4]]

    nsteps = max(min_steps, ceil(Int, Float64(duration) * steps_per_unit))
    dt = duration / nsteps
    transition = _uncoloured_mtbd2_transition(pars)
    for _ in 1:nsteps
        _rk4_uncoloured_mtbd2_joint_step!(y, dt, pars, transition)
    end
    return T[y[3], y[4]]
end

function _uncoloured_mtbd2_E(t::Real, pars::UncolouredMTBD2ConstantParameters;
                             steps_per_unit::Integer=256, min_steps::Integer=16)
    tq = _check_multitype_analytical_time(t)
    _check_multitype_analytical_steps(steps_per_unit, min_steps)
    T = eltype(pars.death)
    E = T[one(T) - pars.ρ₀[1], one(T) - pars.ρ₀[2]]
    tq == 0 && return E

    y = T[E[1], E[2], zero(T), zero(T)]
    nsteps = max(min_steps, ceil(Int, tq * steps_per_unit))
    dt = T(tq / nsteps)
    transition = _uncoloured_mtbd2_transition(pars)
    for _ in 1:nsteps
        _rk4_uncoloured_mtbd2_joint_step!(y, dt, pars, transition)
    end
    return T[y[1], y[2]]
end

function _uncoloured_mtbd2_tip_observation(tip_states::AbstractVector, node::Integer)
    length(tip_states) >= node || throw(ArgumentError("tip_states vector must be node-indexed."))
    return tip_states[node]
end

function _uncoloured_mtbd2_tip_observation(tip_states::AbstractDict, node::Integer)
    haskey(tip_states, node) || throw(ArgumentError("missing tip-state observation for sampled node $node."))
    return tip_states[node]
end

function _uncoloured_mtbd2_allowed_mask(obs::Integer)
    1 <= obs <= 2 || throw(ArgumentError("known tip states must be 1 or 2."))
    return (obs == 1, obs == 2)
end

_uncoloured_mtbd2_allowed_mask(::Missing) = (true, true)
_uncoloured_mtbd2_allowed_mask(::Nothing) = (true, true)
_uncoloured_mtbd2_allowed_mask(::Colon) = (true, true)

function _uncoloured_mtbd2_allowed_mask(obs::Tuple)
    return _uncoloured_mtbd2_allowed_mask(collect(obs))
end

function _uncoloured_mtbd2_allowed_mask(obs::AbstractVector{Bool})
    length(obs) == 2 || throw(ArgumentError("Boolean tip-state masks must have length 2."))
    any(obs) || throw(ArgumentError("tip-state masks must allow at least one state."))
    return (Bool(obs[1]), Bool(obs[2]))
end

function _uncoloured_mtbd2_allowed_mask(obs::AbstractVector{<:Integer})
    isempty(obs) && throw(ArgumentError("ambiguous tip-state sets must allow at least one state."))
    allow1 = false
    allow2 = false
    for state in obs
        1 <= state <= 2 || throw(ArgumentError("ambiguous tip-state sets may contain only states 1 and 2."))
        allow1 |= state == 1
        allow2 |= state == 2
    end
    return (allow1, allow2)
end

function _uncoloured_mtbd2_allowed_mask(obs)
    throw(ArgumentError("unsupported tip-state observation $obs; use 1, 2, missing, nothing, :, a length-2 Bool mask, or a state set such as (1, 2)."))
end

function _uncoloured_mtbd2_tip_boundary(obs, age::T,
                                        pars::UncolouredMTBD2ConstantParameters{T};
                                        steps_per_unit::Integer,
                                        min_steps::Integer) where {T<:AbstractFloat}
    allowed = _uncoloured_mtbd2_allowed_mask(obs)
    Eage = _uncoloured_mtbd2_E(age, pars; steps_per_unit=steps_per_unit, min_steps=min_steps)
    out = zeros(T, 2)
    @inbounds for state in 1:2
        allowed[state] || continue
        factor = iszero(age) ? pars.ρ₀[state] :
                 pars.sampling[state] * (pars.removal_probability[state] +
                 (one(T) - pars.removal_probability[state]) * Eage[state])
        factor > zero(T) || throw(ArgumentError("allowed tip state $state has zero sampling probability."))
        out[state] = factor
    end
    return out
end

function _uncoloured_mtbd2_sampled_unary_boundary(obs, childD::AbstractVector{T},
                                                  pars::UncolouredMTBD2ConstantParameters{T}) where {T<:AbstractFloat}
    allowed = _uncoloured_mtbd2_allowed_mask(obs)
    out = zeros(T, 2)
    @inbounds for state in 1:2
        allowed[state] || continue
        factor = pars.sampling[state] * (one(T) - pars.removal_probability[state])
        factor > zero(T) || throw(ArgumentError("allowed sampled-unary state $state has zero non-removing sampling probability."))
        out[state] = factor * childD[state]
    end
    return out
end

function _uncoloured_mtbd2_only_child(tree::TreeSim.Tree, node::Integer)
    left = tree.left[node]
    right = tree.right[node]
    if left != 0 && right == 0
        return left
    elseif left == 0 && right != 0
        return right
    end
    throw(ArgumentError("sampled-unary node $node must have exactly one child."))
end

function _check_uncoloured_mtbd2_tree(tree::TreeSim.Tree)
    isempty(tree.time) && throw(ArgumentError("tree must be non-empty."))
    try
        TreeSim.validate_tree(tree; require_single_root=true, require_reachable=true)
    catch err
        throw(ArgumentError("tree is not structurally valid for uncoloured MTBD-2 likelihood evaluation: $(sprint(showerror, err))"))
    end
    all(isfinite, tree.time) || throw(ArgumentError("tree times must be finite."))
    supported = (TreeSim.Root, TreeSim.Binary, TreeSim.SampledLeaf, TreeSim.SampledUnary)
    for (i, kind) in pairs(tree.kind)
        kind in supported || throw(ArgumentError("node $i has kind $kind, which is not supported by the uncoloured MTBD-2 likelihood."))
    end
    any(k -> k == TreeSim.SampledLeaf || k == TreeSim.SampledUnary, tree.kind) ||
        throw(ArgumentError("tree must contain at least one sampled node."))
    return nothing
end

function _check_uncoloured_mtbd2_root_prior(root_prior, ::Type{T}) where {T<:AbstractFloat}
    length(root_prior) == 2 || throw(ArgumentError("root_prior must have length 2."))
    prior = T.(root_prior)
    for x in prior
        isfinite(x) || throw(ArgumentError("root_prior entries must be finite."))
        x >= zero(T) || throw(ArgumentError("root_prior entries must be non-negative."))
    end
    sum(prior) > zero(T) || throw(ArgumentError("root_prior must have positive mass."))
    return prior ./ sum(prior)
end

"""
    loglikelihood_uncoloured_mtbd2(tree, params, tip_states; root_prior=[0.5, 0.5])

Evaluate the constant-rate two-type birth-death likelihood for an uncoloured
sampled `TreeSim.Tree` with latent internal states and observed sampled-node
state constraints.

`tip_states` must provide an observation for every `SampledLeaf` and
`SampledUnary`, either as a node-indexed vector or as a dictionary keyed by
sampled node id. Supported
observations are:

- `1` or `2` for a known state
- `missing`, `nothing`, or `:` for a fully unknown state
- a length-2 Boolean mask
- a tuple or vector of allowed states, such as `(1,)`, `(2,)`, or `(1, 2)`

Supported tree node kinds are `Root`, `Binary`, `SampledLeaf`, and
`SampledUnary`. A `SampledUnary` node is handled as an observed non-removing
sampling event on the continuing lineage.
"""
function loglikelihood_uncoloured_mtbd2(tree::TreeSim.Tree,
                                        pars::UncolouredMTBD2ConstantParameters{T},
                                        tip_states;
                                        root_prior=[0.5, 0.5],
                                        steps_per_unit::Integer=256,
                                        min_steps::Integer=16) where {T<:AbstractFloat}
    _check_uncoloured_mtbd2_tree(tree)
    _check_multitype_analytical_steps(steps_per_unit, min_steps)
    prior = _check_uncoloured_mtbd2_root_prior(root_prior, T)

    horizon = T(maximum(tree.time))
    root = TreeSim.root(tree)
    node_vectors = [zeros(T, 2) for _ in eachindex(tree)]

    for node in TreeSim.postorder(tree, root)
        age = horizon - T(tree.time[node])
        if tree.kind[node] == TreeSim.SampledLeaf
            obs = _uncoloured_mtbd2_tip_observation(tip_states, node)
            node_vectors[node] .= _uncoloured_mtbd2_tip_boundary(obs, age, pars;
                                                                 steps_per_unit=steps_per_unit,
                                                                 min_steps=min_steps)
        elseif tree.kind[node] == TreeSim.SampledUnary
            child = _uncoloured_mtbd2_only_child(tree, node)
            childD = _uncoloured_mtbd2_propagate(node_vectors[child], horizon - T(tree.time[child]), age, pars;
                                                 steps_per_unit=steps_per_unit, min_steps=min_steps)
            obs = _uncoloured_mtbd2_tip_observation(tip_states, node)
            node_vectors[node] .= _uncoloured_mtbd2_sampled_unary_boundary(obs, childD, pars)
        elseif tree.kind[node] == TreeSim.Binary
            left = tree.left[node]
            right = tree.right[node]
            leftD = _uncoloured_mtbd2_propagate(node_vectors[left], horizon - T(tree.time[left]), age, pars;
                                                steps_per_unit=steps_per_unit, min_steps=min_steps)
            rightD = _uncoloured_mtbd2_propagate(node_vectors[right], horizon - T(tree.time[right]), age, pars;
                                                 steps_per_unit=steps_per_unit, min_steps=min_steps)
            out = node_vectors[node]
            @inbounds for a in 1:2
                out[a] = zero(T)
                for b in 1:2
                    out[a] += pars.birth[a, b] * (leftD[a] * rightD[b] + leftD[b] * rightD[a])
                end
            end
        elseif tree.kind[node] == TreeSim.Root
            children = TreeSim.children(tree, node)
            length(children) == 1 || length(children) == 2 ||
                throw(ArgumentError("root must have one or two children for uncoloured MTBD-2 likelihood evaluation."))
            if length(children) == 1
                child = only(children)
                node_vectors[node] .= _uncoloured_mtbd2_propagate(node_vectors[child], horizon - T(tree.time[child]), age, pars;
                                                                  steps_per_unit=steps_per_unit, min_steps=min_steps)
            else
                left, right = children
                leftD = _uncoloured_mtbd2_propagate(node_vectors[left], horizon - T(tree.time[left]), age, pars;
                                                    steps_per_unit=steps_per_unit, min_steps=min_steps)
                rightD = _uncoloured_mtbd2_propagate(node_vectors[right], horizon - T(tree.time[right]), age, pars;
                                                     steps_per_unit=steps_per_unit, min_steps=min_steps)
                out = node_vectors[node]
                @inbounds for a in 1:2
                    out[a] = zero(T)
                    for b in 1:2
                        out[a] += pars.birth[a, b] * (leftD[a] * rightD[b] + leftD[b] * rightD[a])
                    end
                end
            end
        end
    end

    likelihood = dot(prior, node_vectors[root])
    likelihood > zero(T) || return -Inf
    return log(likelihood)
end

function likelihood_uncoloured_mtbd2(tree::TreeSim.Tree,
                                     pars::UncolouredMTBD2ConstantParameters,
                                     tip_states; kwargs...)
    return exp(loglikelihood_uncoloured_mtbd2(tree, pars, tip_states; kwargs...))
end

function loglikelihood_uncoloured_mtbd2_superspreader(tree::TreeSim.Tree,
                                                      pars::UncolouredMTBD2SuperspreaderParameters,
                                                      tip_states; kwargs...)
    return loglikelihood_uncoloured_mtbd2(tree, uncoloured_mtbd2_native_parameters(pars), tip_states; kwargs...)
end

function likelihood_uncoloured_mtbd2_superspreader(tree::TreeSim.Tree,
                                                   pars::UncolouredMTBD2SuperspreaderParameters,
                                                   tip_states; kwargs...)
    return exp(loglikelihood_uncoloured_mtbd2_superspreader(tree, pars, tip_states; kwargs...))
end

function _check_uncoloured_mtbd2_batch_lengths(trees, tip_states_list)
    length(trees) == length(tip_states_list) ||
        throw(ArgumentError("trees and tip_states_list must have the same length."))
    return nothing
end

"""
    loglikelihoods_uncoloured_mtbd2(trees, params, tip_states_list; kwargs...)

Return one log likelihood per tree by repeatedly calling
[`loglikelihood_uncoloured_mtbd2`](@ref). `tip_states_list[i]` is the sampled-node
observation object for `trees[i]`.
"""
function loglikelihoods_uncoloured_mtbd2(trees::AbstractVector{<:TreeSim.Tree},
                                         pars::UncolouredMTBD2ConstantParameters,
                                         tip_states_list::AbstractVector; kwargs...)
    _check_uncoloured_mtbd2_batch_lengths(trees, tip_states_list)
    return [loglikelihood_uncoloured_mtbd2(tree, pars, tip_states; kwargs...)
            for (tree, tip_states) in zip(trees, tip_states_list)]
end

function likelihoods_uncoloured_mtbd2(trees::AbstractVector{<:TreeSim.Tree},
                                      pars::UncolouredMTBD2ConstantParameters,
                                      tip_states_list::AbstractVector; kwargs...)
    return exp.(loglikelihoods_uncoloured_mtbd2(trees, pars, tip_states_list; kwargs...))
end

function total_loglikelihood_uncoloured_mtbd2(trees::AbstractVector{<:TreeSim.Tree},
                                              pars::UncolouredMTBD2ConstantParameters,
                                              tip_states_list::AbstractVector; kwargs...)
    return sum(loglikelihoods_uncoloured_mtbd2(trees, pars, tip_states_list; kwargs...))
end

function total_loglikelihood_uncoloured_mtbd2_superspreader(trees::AbstractVector{<:TreeSim.Tree},
                                                            pars::UncolouredMTBD2SuperspreaderParameters,
                                                            tip_states_list::AbstractVector; kwargs...)
    return total_loglikelihood_uncoloured_mtbd2(trees, uncoloured_mtbd2_native_parameters(pars), tip_states_list; kwargs...)
end

function loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_free::AbstractVector{<:Real},
                                                                tree::TreeSim.Tree,
                                                                spec::UncolouredMTBD2SuperspreaderSpec,
                                                                tip_states; kwargs...)
    pars = uncoloured_mtbd2_superspreader_parameters_from_free_vector(θ_free, spec)
    return loglikelihood_uncoloured_mtbd2_superspreader(tree, pars, tip_states; kwargs...)
end

function total_loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_free::AbstractVector{<:Real},
                                                                      trees::AbstractVector{<:TreeSim.Tree},
                                                                      spec::UncolouredMTBD2SuperspreaderSpec,
                                                                      tip_states_list::AbstractVector; kwargs...)
    pars = uncoloured_mtbd2_superspreader_parameters_from_free_vector(θ_free, spec)
    return total_loglikelihood_uncoloured_mtbd2_superspreader(trees, pars, tip_states_list; kwargs...)
end

function loglikelihood_uncoloured_mtbd2_superspreader_from_unconstrained(η_free::AbstractVector{<:Real},
                                                                         tree::TreeSim.Tree,
                                                                         spec::UncolouredMTBD2SuperspreaderSpec,
                                                                         tip_states; kwargs...)
    θ_free = uncoloured_mtbd2_superspreader_free_from_unconstrained(η_free, spec)
    return loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_free, tree, spec, tip_states; kwargs...)
end

function total_loglikelihood_uncoloured_mtbd2_superspreader_from_unconstrained(η_free::AbstractVector{<:Real},
                                                                               trees::AbstractVector{<:TreeSim.Tree},
                                                                               spec::UncolouredMTBD2SuperspreaderSpec,
                                                                               tip_states_list::AbstractVector; kwargs...)
    θ_free = uncoloured_mtbd2_superspreader_free_from_unconstrained(η_free, spec)
    return total_loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_free, trees, spec, tip_states_list; kwargs...)
end

function score_uncoloured_mtbd2(trees::AbstractVector{<:TreeSim.Tree},
                                pars::UncolouredMTBD2ConstantParameters,
                                tip_states_list::AbstractVector; kwargs...)
    per_tree = loglikelihoods_uncoloured_mtbd2(trees, pars, tip_states_list; kwargs...)
    total = sum(per_tree)
    n = length(per_tree)
    return (
        per_tree_loglikelihood=per_tree,
        total_loglikelihood=total,
        mean_loglikelihood=n == 0 ? NaN : total / n,
        n_scored=n,
    )
end

function loglikelihood_uncoloured_mtbd2_from_free(θ_free::AbstractVector{<:Real},
                                                  tree::TreeSim.Tree,
                                                  spec::UncolouredMTBD2ParameterSpec,
                                                  tip_states; kwargs...)
    pars = uncoloured_mtbd2_parameters_from_free_vector(θ_free, spec)
    return loglikelihood_uncoloured_mtbd2(tree, pars, tip_states; kwargs...)
end

function total_loglikelihood_uncoloured_mtbd2_from_free(θ_free::AbstractVector{<:Real},
                                                        trees::AbstractVector{<:TreeSim.Tree},
                                                        spec::UncolouredMTBD2ParameterSpec,
                                                        tip_states_list::AbstractVector; kwargs...)
    pars = uncoloured_mtbd2_parameters_from_free_vector(θ_free, spec)
    return total_loglikelihood_uncoloured_mtbd2(trees, pars, tip_states_list; kwargs...)
end

function loglikelihood_uncoloured_mtbd2_from_unconstrained(η_free::AbstractVector{<:Real},
                                                           tree::TreeSim.Tree,
                                                           spec::UncolouredMTBD2ParameterSpec,
                                                           tip_states; kwargs...)
    θ_free = uncoloured_mtbd2_free_from_unconstrained(η_free, spec)
    return loglikelihood_uncoloured_mtbd2_from_free(θ_free, tree, spec, tip_states; kwargs...)
end

function total_loglikelihood_uncoloured_mtbd2_from_unconstrained(η_free::AbstractVector{<:Real},
                                                                 trees::AbstractVector{<:TreeSim.Tree},
                                                                 spec::UncolouredMTBD2ParameterSpec,
                                                                 tip_states_list::AbstractVector; kwargs...)
    θ_free = uncoloured_mtbd2_free_from_unconstrained(η_free, spec)
    return total_loglikelihood_uncoloured_mtbd2_from_free(θ_free, trees, spec, tip_states_list; kwargs...)
end

function _uncoloured_mtbd2_free_length(spec::UncolouredMTBD2ParameterSpec)
    return count(_uncoloured_mtbd2_free_mask_vector(spec))
end

function _uncoloured_mtbd2_initial_free_vector(init, spec::UncolouredMTBD2ParameterSpec)
    nfree = _uncoloured_mtbd2_free_length(spec)
    nfree > 0 || throw(ArgumentError("at least one uncoloured MTBD-2 parameter must be selected for fitting."))
    θ0 = if init isa UncolouredMTBD2ConstantParameters
        free_parameter_vector(init, spec)
    else
        collect(Float64, init)
    end
    length(θ0) == nfree || throw(ArgumentError("initial free vector length must match the number of fitted parameters."))
    return θ0
end

function _uncoloured_mtbd2_negloglikelihood(θ_free::AbstractVector{<:Real},
                                            trees::AbstractVector{<:TreeSim.Tree},
                                            spec::UncolouredMTBD2ParameterSpec,
                                            tip_states_list::AbstractVector; kwargs...)
    try
        ll = total_loglikelihood_uncoloured_mtbd2_from_free(θ_free, trees, spec, tip_states_list; kwargs...)
        return isfinite(ll) ? -ll : 1e12
    catch err
        err isa ArgumentError || rethrow()
        return 1e12
    end
end

function fit_uncoloured_mtbd2_mle(trees::AbstractVector{<:TreeSim.Tree},
                                  init,
                                  spec::UncolouredMTBD2ParameterSpec,
                                  tip_states_list::AbstractVector;
                                  lower::Union{Nothing,AbstractVector{<:Real}}=nothing,
                                  upper::Union{Nothing,AbstractVector{<:Real}}=nothing,
                                  initial_step::Real=0.1,
                                  tolerance::Real=1e-5,
                                  maxiter::Integer=2_000,
                                  kwargs...)
    isempty(trees) && throw(ArgumentError("trees must be non-empty."))
    _check_uncoloured_mtbd2_batch_lengths(trees, tip_states_list)
    θ0 = _uncoloured_mtbd2_initial_free_vector(init, spec)
    n = length(θ0)
    lo = lower === nothing ? fill(-Inf, n) : collect(Float64, lower)
    hi = upper === nothing ? fill(Inf, n) : collect(Float64, upper)
    length(lo) == n || throw(ArgumentError("lower length must match the number of fitted parameters."))
    length(hi) == n || throw(ArgumentError("upper length must match the number of fitted parameters."))
    all(lo .< hi) || throw(ArgumentError("lower bounds must be smaller than upper bounds."))

    obj = θ -> _uncoloured_mtbd2_negloglikelihood(θ, trees, spec, tip_states_list; kwargs...)
    initial_negloglikelihood = obj(θ0)
    result = _coordinate_minimize(obj, θ0, lo, hi;
                                  initial_step=Float64(initial_step),
                                  tolerance=Float64(tolerance),
                                  maxiter=Int(maxiter))
    θhat = result.minimizer
    pars_hat = uncoloured_mtbd2_parameters_from_free_vector(θhat, spec)
    return (
        θ_free_hat=θhat,
        params_hat=pars_hat,
        maximum_loglikelihood=-result.minimum,
        minimum_negloglikelihood=result.minimum,
        initial_negloglikelihood=initial_negloglikelihood,
        converged=result.converged,
        optimizer_summary=result,
        spec=spec,
    )
end

function fit_uncoloured_mtbd2_mle(tree::TreeSim.Tree,
                                  init,
                                  spec::UncolouredMTBD2ParameterSpec,
                                  tip_states; kwargs...)
    return fit_uncoloured_mtbd2_mle([tree], init, spec, [tip_states]; kwargs...)
end

function _uncoloured_mtbd2_initial_unconstrained_vector(init, spec::UncolouredMTBD2ParameterSpec; init_is_unconstrained::Bool=false)
    nfree = _uncoloured_mtbd2_free_length(spec)
    nfree > 0 || throw(ArgumentError("at least one uncoloured MTBD-2 parameter must be selected for fitting."))
    if init_is_unconstrained
        init isa UncolouredMTBD2ConstantParameters &&
            throw(ArgumentError("init_is_unconstrained=true requires a vector init, not a parameter object."))
        η0 = collect(Float64, init)
        length(η0) == nfree || throw(ArgumentError("initial unconstrained vector length must match the number of fitted parameters."))
        return η0
    end
    θ0 = _uncoloured_mtbd2_initial_free_vector(init, spec)
    return uncoloured_mtbd2_unconstrained_from_free(θ0, spec)
end

function _uncoloured_mtbd2_transformed_negloglikelihood(η_free::AbstractVector{<:Real},
                                                        trees::AbstractVector{<:TreeSim.Tree},
                                                        spec::UncolouredMTBD2ParameterSpec,
                                                        tip_states_list::AbstractVector; kwargs...)
    try
        ll = total_loglikelihood_uncoloured_mtbd2_from_unconstrained(η_free, trees, spec, tip_states_list; kwargs...)
        return isfinite(ll) ? -ll : 1e12
    catch err
        err isa ArgumentError || rethrow()
        return 1e12
    end
end

function fit_uncoloured_mtbd2_mle_transformed(trees::AbstractVector{<:TreeSim.Tree},
                                              init,
                                              spec::UncolouredMTBD2ParameterSpec,
                                              tip_states_list::AbstractVector;
                                              init_is_unconstrained::Bool=false,
                                              initial_step::Real=0.1,
                                              tolerance::Real=1e-5,
                                              maxiter::Integer=2_000,
                                              kwargs...)
    isempty(trees) && throw(ArgumentError("trees must be non-empty."))
    _check_uncoloured_mtbd2_batch_lengths(trees, tip_states_list)
    η0 = _uncoloured_mtbd2_initial_unconstrained_vector(init, spec; init_is_unconstrained)
    obj = η -> _uncoloured_mtbd2_transformed_negloglikelihood(η, trees, spec, tip_states_list; kwargs...)
    initial_negloglikelihood = obj(η0)
    result = _coordinate_minimize(obj, η0, fill(-Inf, length(η0)), fill(Inf, length(η0));
                                  initial_step=Float64(initial_step),
                                  tolerance=Float64(tolerance),
                                  maxiter=Int(maxiter))
    ηhat = result.minimizer
    θhat = uncoloured_mtbd2_free_from_unconstrained(ηhat, spec)
    pars_hat = uncoloured_mtbd2_parameters_from_free_vector(θhat, spec)
    return (
        η_free_hat=ηhat,
        θ_free_hat=θhat,
        params_hat=pars_hat,
        maximum_loglikelihood=-result.minimum,
        minimum_negloglikelihood=result.minimum,
        initial_negloglikelihood=initial_negloglikelihood,
        converged=result.converged,
        optimizer_summary=result,
        spec=spec,
    )
end

function fit_uncoloured_mtbd2_mle_transformed(tree::TreeSim.Tree,
                                              init,
                                              spec::UncolouredMTBD2ParameterSpec,
                                              tip_states; kwargs...)
    return fit_uncoloured_mtbd2_mle_transformed([tree], init, spec, [tip_states]; kwargs...)
end

function _uncoloured_mtbd2_superspreader_free_length(spec::UncolouredMTBD2SuperspreaderSpec)
    return count(_uncoloured_mtbd2_superspreader_free_mask_vector(spec))
end

function _uncoloured_mtbd2_superspreader_initial_free_vector(init, spec::UncolouredMTBD2SuperspreaderSpec)
    nfree = _uncoloured_mtbd2_superspreader_free_length(spec)
    nfree > 0 || throw(ArgumentError("at least one uncoloured MTBD-2 superspreader parameter must be selected for fitting."))
    θ0 = if init isa UncolouredMTBD2SuperspreaderParameters
        free_parameter_vector(init, spec)
    else
        collect(Float64, init)
    end
    length(θ0) == nfree || throw(ArgumentError("initial superspreader free vector length must match the number of fitted parameters."))
    return θ0
end

function _uncoloured_mtbd2_superspreader_initial_unconstrained_vector(init,
                                                                      spec::UncolouredMTBD2SuperspreaderSpec;
                                                                      init_is_unconstrained::Bool=false)
    nfree = _uncoloured_mtbd2_superspreader_free_length(spec)
    nfree > 0 || throw(ArgumentError("at least one uncoloured MTBD-2 superspreader parameter must be selected for fitting."))
    if init_is_unconstrained
        init isa UncolouredMTBD2SuperspreaderParameters &&
            throw(ArgumentError("init_is_unconstrained=true requires a vector init, not a superspreader parameter object."))
        η0 = collect(Float64, init)
        length(η0) == nfree || throw(ArgumentError("initial superspreader unconstrained vector length must match the number of fitted parameters."))
        return η0
    end
    θ0 = _uncoloured_mtbd2_superspreader_initial_free_vector(init, spec)
    return uncoloured_mtbd2_superspreader_unconstrained_from_free(θ0, spec)
end

function _uncoloured_mtbd2_superspreader_transformed_negloglikelihood(η_free::AbstractVector{<:Real},
                                                                      trees::AbstractVector{<:TreeSim.Tree},
                                                                      spec::UncolouredMTBD2SuperspreaderSpec,
                                                                      tip_states_list::AbstractVector; kwargs...)
    try
        ll = total_loglikelihood_uncoloured_mtbd2_superspreader_from_unconstrained(η_free, trees, spec, tip_states_list; kwargs...)
        return isfinite(ll) ? -ll : 1e12
    catch err
        err isa ArgumentError || rethrow()
        return 1e12
    end
end

function fit_uncoloured_mtbd2_superspreader_mle(trees::AbstractVector{<:TreeSim.Tree},
                                                init,
                                                spec::UncolouredMTBD2SuperspreaderSpec,
                                                tip_states_list::AbstractVector;
                                                init_is_unconstrained::Bool=false,
                                                initial_step::Real=0.1,
                                                tolerance::Real=1e-5,
                                                maxiter::Integer=2_000,
                                                kwargs...)
    isempty(trees) && throw(ArgumentError("trees must be non-empty."))
    _check_uncoloured_mtbd2_batch_lengths(trees, tip_states_list)
    η0 = _uncoloured_mtbd2_superspreader_initial_unconstrained_vector(init, spec; init_is_unconstrained)
    obj = η -> _uncoloured_mtbd2_superspreader_transformed_negloglikelihood(η, trees, spec, tip_states_list; kwargs...)
    initial_negloglikelihood = obj(η0)
    result = _coordinate_minimize(obj, η0, fill(-Inf, length(η0)), fill(Inf, length(η0));
                                  initial_step=Float64(initial_step),
                                  tolerance=Float64(tolerance),
                                  maxiter=Int(maxiter))
    ηhat = result.minimizer
    θhat = uncoloured_mtbd2_superspreader_free_from_unconstrained(ηhat, spec)
    superspreader_hat = uncoloured_mtbd2_superspreader_parameters_from_free_vector(θhat, spec)
    native_hat = uncoloured_mtbd2_native_parameters(superspreader_hat)
    return (
        η_free_hat=ηhat,
        θ_free_hat=θhat,
        superspreader_params_hat=superspreader_hat,
        params_hat=native_hat,
        maximum_loglikelihood=-result.minimum,
        minimum_negloglikelihood=result.minimum,
        initial_negloglikelihood=initial_negloglikelihood,
        converged=result.converged,
        optimizer_summary=result,
        spec=spec,
    )
end

function fit_uncoloured_mtbd2_superspreader_mle(tree::TreeSim.Tree,
                                                init,
                                                spec::UncolouredMTBD2SuperspreaderSpec,
                                                tip_states; kwargs...)
    return fit_uncoloured_mtbd2_superspreader_mle([tree], init, spec, [tip_states]; kwargs...)
end

function loglikelihood_uncoloured_mtbd2_known_tips(tree::TreeSim.Tree,
                                                  pars::UncolouredMTBD2ConstantParameters,
                                                  tip_states; kwargs...)
    return loglikelihood_uncoloured_mtbd2(tree, pars, tip_states; kwargs...)
end

function likelihood_uncoloured_mtbd2_known_tips(tree::TreeSim.Tree,
                                                pars::UncolouredMTBD2ConstantParameters,
                                                tip_states; kwargs...)
    return likelihood_uncoloured_mtbd2(tree, pars, tip_states; kwargs...)
end
