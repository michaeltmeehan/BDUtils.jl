struct MultitypeMLESpec
    initial::MultitypeBDParameters{Float64}
    fit_birth::Matrix{Bool}
    fit_death::Vector{Bool}
    fit_sampling::Vector{Bool}
    fit_transition::Matrix{Bool}
end

function MultitypeMLESpec(initial::MultitypeBDParameters;
                          fit_birth::AbstractMatrix{Bool}=initial.birth .> 0,
                          fit_death::AbstractVector{Bool}=initial.death .> 0,
                          fit_sampling::AbstractVector{Bool}=initial.sampling .> 0,
                          fit_transition::AbstractMatrix{Bool}=initial.transition .> 0)
    K = length(initial)
    size(fit_birth) == (K, K) || throw(ArgumentError("fit_birth must be a K x K Bool matrix."))
    length(fit_death) == K || throw(ArgumentError("fit_death must have length K."))
    length(fit_sampling) == K || throw(ArgumentError("fit_sampling must have length K."))
    size(fit_transition) == (K, K) || throw(ArgumentError("fit_transition must be a K x K Bool matrix."))

    birth_mask = Matrix(fit_birth)
    death_mask = Vector(fit_death)
    sampling_mask = Vector(fit_sampling)
    transition_mask = Matrix(fit_transition)
    for k in 1:K
        transition_mask[k, k] = false
    end

    _check_multitype_fit_mask_positive("birth", initial.birth, birth_mask)
    _check_multitype_fit_mask_positive("death", initial.death, death_mask)
    _check_multitype_fit_mask_positive("sampling", initial.sampling, sampling_mask)
    _check_multitype_fit_mask_positive("transition", initial.transition, transition_mask)

    fixed = MultitypeBDParameters(initial.birth, initial.death, initial.sampling,
                                  initial.removal_probability, initial.transition, initial.ρ₀)
    return MultitypeMLESpec(fixed, birth_mask, death_mask, sampling_mask, transition_mask)
end

function _check_multitype_fit_mask_positive(name::AbstractString, values, mask)
    for i in eachindex(mask)
        mask[i] || continue
        values[i] > 0 || throw(ArgumentError("fitted $name entries must be positive in the initial parameter template."))
    end
    return nothing
end

function _multitype_fit_length(spec::MultitypeMLESpec)
    return count(spec.fit_birth) + count(spec.fit_death) +
           count(spec.fit_sampling) + count(spec.fit_transition)
end

function multitype_pack_parameters(pars::MultitypeBDParameters, spec::MultitypeMLESpec)
    length(pars) == length(spec.initial) || throw(ArgumentError("parameter dimension must match spec."))
    θ = Float64[]
    append!(θ, log.(pars.birth[spec.fit_birth]))
    append!(θ, log.(pars.death[spec.fit_death]))
    append!(θ, log.(pars.sampling[spec.fit_sampling]))
    append!(θ, log.(pars.transition[spec.fit_transition]))
    all(isfinite, θ) || throw(ArgumentError("all packed fitted rates must be positive and finite."))
    return θ
end

function multitype_unpack_parameters(θ::AbstractVector{<:Real}, spec::MultitypeMLESpec)
    n = _multitype_fit_length(spec)
    length(θ) == n || throw(ArgumentError("θ length must match the number of fitted multitype rates."))
    birth = copy(spec.initial.birth)
    death = copy(spec.initial.death)
    sampling = copy(spec.initial.sampling)
    transition = copy(spec.initial.transition)

    idx = 1
    for i in eachindex(birth)
        spec.fit_birth[i] || continue
        birth[i] = exp(Float64(θ[idx]))
        idx += 1
    end
    for i in eachindex(death)
        spec.fit_death[i] || continue
        death[i] = exp(Float64(θ[idx]))
        idx += 1
    end
    for i in eachindex(sampling)
        spec.fit_sampling[i] || continue
        sampling[i] = exp(Float64(θ[idx]))
        idx += 1
    end
    for i in eachindex(transition)
        spec.fit_transition[i] || continue
        transition[i] = exp(Float64(θ[idx]))
        idx += 1
    end

    return MultitypeBDParameters(birth, death, sampling,
                                 spec.initial.removal_probability,
                                 transition, spec.initial.ρ₀)
end

multitype_loglikelihood(tree::MultitypeColoredTree, pars::MultitypeBDParameters; kwargs...) =
    multitype_colored_loglikelihood(tree, pars; kwargs...)

function multitype_loglikelihood(trees::AbstractVector{<:MultitypeColoredTree},
                                 pars::MultitypeBDParameters; kwargs...)
    isempty(trees) && throw(ArgumentError("trees must be non-empty."))
    return sum(tree -> multitype_colored_loglikelihood(tree, pars; kwargs...), trees)
end

function multitype_negloglikelihood(θ::AbstractVector{<:Real},
                                    tree::MultitypeColoredTree,
                                    spec::MultitypeMLESpec; kwargs...)
    return multitype_negloglikelihood(θ, [tree], spec; kwargs...)
end

function multitype_negloglikelihood(θ::AbstractVector{<:Real},
                                    trees::AbstractVector{<:MultitypeColoredTree},
                                    spec::MultitypeMLESpec; kwargs...)
    try
        pars = multitype_unpack_parameters(θ, spec)
        ll = multitype_loglikelihood(trees, pars; kwargs...)
        return isfinite(ll) ? -ll : 1e12
    catch err
        err isa ArgumentError || rethrow()
        return 1e12
    end
end

function fit_multitype_mle(trees::AbstractVector{<:MultitypeColoredTree};
                           spec::MultitypeMLESpec,
                           θ_init::Union{Nothing,AbstractVector{<:Real}}=nothing,
                           lower::Union{Nothing,AbstractVector{<:Real}}=nothing,
                           upper::Union{Nothing,AbstractVector{<:Real}}=nothing,
                           initial_step::Real=1.0,
                           tolerance::Real=1e-5,
                           maxiter::Integer=2_000,
                           kwargs...)
    isempty(trees) && throw(ArgumentError("trees must be non-empty."))
    n = _multitype_fit_length(spec)
    n > 0 || throw(ArgumentError("at least one multitype rate must be selected for fitting."))
    θ0 = θ_init === nothing ? multitype_pack_parameters(spec.initial, spec) : collect(Float64, θ_init)
    length(θ0) == n || throw(ArgumentError("θ_init length must match the number of fitted multitype rates."))
    lo = lower === nothing ? fill(log(1e-8), n) : collect(Float64, lower)
    hi = upper === nothing ? fill(log(1e3), n) : collect(Float64, upper)
    length(lo) == n || throw(ArgumentError("lower length must match the number of fitted multitype rates."))
    length(hi) == n || throw(ArgumentError("upper length must match the number of fitted multitype rates."))
    all(lo .< hi) || throw(ArgumentError("lower bounds must be smaller than upper bounds."))

    obj = θ -> multitype_negloglikelihood(θ, trees, spec; kwargs...)
    result = _coordinate_minimize(obj, θ0, lo, hi;
                                  initial_step=Float64(initial_step),
                                  tolerance=Float64(tolerance),
                                  maxiter=Int(maxiter))
    θ̂ = result.minimizer
    pars = multitype_unpack_parameters(θ̂, spec)
    return (
        result=result,
        θ̂=θ̂,
        parameters=pars,
        loglikelihood=-result.minimum,
        negloglikelihood=result.minimum,
        spec=spec,
    )
end

fit_multitype_mle(tree::MultitypeColoredTree; kwargs...) =
    fit_multitype_mle([tree]; kwargs...)
