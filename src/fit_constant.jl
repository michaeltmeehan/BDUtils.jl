struct BDFixedSpec{T<:Real}
    fixed_symbol::Symbol
    fixed_value::T
end

abstract type AbstractBDParameterization end

struct RateParameterization{T<:Real} <: AbstractBDParameterization
    spec::BDFixedSpec{T}
end

struct R0DeltaParameterization{T<:Real} <: AbstractBDParameterization
    spec::BDFixedSpec{T}
end

backtransform(::RateParameterization, λ, μ, ψ) = (λ=λ, μ=μ, ψ=ψ)

function backtransform(::R0DeltaParameterization, λ, μ, ψ)
    δ = μ + ψ
    p = ψ / δ
    R₀ = λ / δ
    return (R₀=R₀, δ=δ, p=p)
end

function expand_rates(param::RateParameterization, θ::AbstractVector{T}) where {T<:Real}
    spec = param.spec
    if spec.fixed_symbol === :λ
        return T(spec.fixed_value), exp(θ[1]), exp(θ[2])
    elseif spec.fixed_symbol === :μ
        return exp(θ[1]), T(spec.fixed_value), exp(θ[2])
    elseif spec.fixed_symbol === :ψ
        return exp(θ[1]), exp(θ[2]), T(spec.fixed_value)
    end
    throw(ArgumentError("fixed rate parameter must be :λ, :μ, or :ψ."))
end

function expand_rates(param::R0DeltaParameterization, θ::AbstractVector{T}) where {T<:Real}
    spec = param.spec
    if spec.fixed_symbol === :R0
        R₀ = T(spec.fixed_value)
        δ = exp(θ[1])
        p = inv(one(T) + exp(-θ[2]))
    elseif spec.fixed_symbol === :δ
        R₀ = exp(θ[1])
        δ = T(spec.fixed_value)
        p = inv(one(T) + exp(-θ[2]))
    elseif spec.fixed_symbol === :p
        R₀ = exp(θ[1])
        δ = exp(θ[2])
        p = T(spec.fixed_value)
        zero(T) < p < one(T) || throw(ArgumentError("fixed p must be in (0, 1)."))
    else
        throw(ArgumentError("fixed R0/δ/p parameter must be :R0, :δ, or :p."))
    end
    λ = R₀ * δ
    ψ = p * δ
    μ = (one(T) - p) * δ
    return λ, μ, ψ
end

_tuple_to_vector(nt::NamedTuple) = collect(values(nt))
_parameter_transform(param::AbstractBDParameterization, θ) = _tuple_to_vector(backtransform(param, expand_rates(param, θ)...))

function _bd_negloglikelihood(θ::AbstractVector{T}, tree::TreeSim.Tree, param::AbstractBDParameterization; r, ρ₀=zero(T)) where {T<:Real}
    λ, μ, ψ = expand_rates(param, θ)
    ll = bd_loglikelihood_constant(tree, ConstantRateBDParameters(λ, μ, ψ, T(r), T(ρ₀)))
    return isfinite(ll) ? -ll : T(1e12)
end

struct BDConstantObjective{P,T}
    tree::TreeSim.Tree
    param::P
    r::T
    ρ₀::T
end

function (obj::BDConstantObjective)(θ::AbstractVector{T}) where {T<:Real}
    try
        return _bd_negloglikelihood(θ, obj.tree, obj.param; r=T(obj.r), ρ₀=T(obj.ρ₀))
    catch err
        err isa ArgumentError || rethrow()
        return T(1e12)
    end
end

"""
    fit_bd_full(tree; param, r, ρ₀=0.0, θ_init=zeros(2))

Fit the supported constant-rate birth-death-sampling likelihood.

The result preserves the historical `rates = (λ, μ, ψ)` and `parameters`
fields, and also includes `constant_rates::ConstantRateBDParameters`, the
canonical fitted constant-rate parameter object including `r` and `ρ₀`.
"""
function fit_bd_full(
    tree::TreeSim.Tree;
    param::AbstractBDParameterization,
    r::Real,
    ρ₀::Real=0.0,
    θ_init::AbstractVector{<:Real}=zeros(2),
)
    length(θ_init) == 2 || throw(ArgumentError("θ_init must have length 2."))
    obj = BDConstantObjective(tree, param, Float64(r), Float64(ρ₀))
    lower = fill(log(1e-6), 2)
    upper = fill(log(1e3), 2)
    result = _coordinate_minimize(obj, collect(Float64, θ_init), lower, upper)
    θ̂ = result.minimizer
    grad = _finite_gradient(obj, θ̂)
    H = _finite_hessian(obj, θ̂)
    vcov_θ = try
        inv(H)
    catch
        fill(NaN, size(H))
    end
    se_θ = sqrt.(max.(diag(vcov_θ), 0.0))
    λ̂, μ̂, ψ̂ = expand_rates(param, θ̂)
    params = backtransform(param, λ̂, μ̂, ψ̂)
    J = _finite_jacobian(θ -> _parameter_transform(param, θ), θ̂)
    vcov_param = J * vcov_θ * J'
    se_param = sqrt.(max.(diag(vcov_param), 0.0))
    constant_rates = ConstantRateBDParameters(λ̂, μ̂, ψ̂, Float64(r), Float64(ρ₀))
    return (
        result=result,
        θ̂=θ̂,
        gradient=grad,
        hessian=H,
        vcov_θ=vcov_θ,
        se_θ=se_θ,
        rates=(λ=λ̂, μ=μ̂, ψ=ψ̂),
        constant_rates=constant_rates,
        parameters=params,
        vcov_parameters=vcov_param,
        se_parameters=se_param,
    )
end

function _finite_jacobian(f, θ::Vector{Float64})
    y0 = f(θ)
    J = zeros(Float64, length(y0), length(θ))
    for i in eachindex(θ)
        h = sqrt(eps(Float64)) * max(abs(θ[i]), 1.0)
        hi = copy(θ)
        lo = copy(θ)
        hi[i] += h
        lo[i] -= h
        J[:, i] .= (f(hi) .- f(lo)) ./ (2h)
    end
    return J
end

function _coordinate_minimize(obj, θ0::Vector{Float64}, lower::Vector{Float64}, upper::Vector{Float64};
                              initial_step::Float64=1.0, tolerance::Float64=1e-5, maxiter::Int=2_000)
    θ = clamp.(copy(θ0), lower, upper)
    value = obj(θ)
    step = initial_step
    iterations = 0

    while step > tolerance && iterations < maxiter
        improved = false
        iterations += 1

        for j in eachindex(θ)
            for direction in (-1.0, 1.0)
                candidate = copy(θ)
                candidate[j] = clamp(candidate[j] + direction * step, lower[j], upper[j])
                candidate_value = obj(candidate)
                if candidate_value < value
                    θ = candidate
                    value = candidate_value
                    improved = true
                end
            end
        end

        improved || (step /= 2)
    end

    return (minimizer=θ, minimum=value, iterations=iterations, converged=step <= tolerance)
end

function _finite_gradient(f, θ::Vector{Float64})
    grad = similar(θ)
    for i in eachindex(θ)
        h = sqrt(eps(Float64)) * max(abs(θ[i]), 1.0)
        hi = copy(θ)
        lo = copy(θ)
        hi[i] += h
        lo[i] -= h
        grad[i] = (f(hi) - f(lo)) / (2h)
    end
    return grad
end

function _finite_hessian(f, θ::Vector{Float64})
    n = length(θ)
    H = zeros(Float64, n, n)
    f0 = f(θ)
    for i in 1:n
        hi = copy(θ)
        lo = copy(θ)
        h = cbrt(eps(Float64)) * max(abs(θ[i]), 1.0)
        hi[i] += h
        lo[i] -= h
        H[i, i] = (f(hi) - 2f0 + f(lo)) / (h * h)
        for j in (i + 1):n
            k = cbrt(eps(Float64)) * max(max(abs(θ[i]), abs(θ[j])), 1.0)
            pp = copy(θ); pp[i] += k; pp[j] += k
            pm = copy(θ); pm[i] += k; pm[j] -= k
            mp = copy(θ); mp[i] -= k; mp[j] += k
            mm = copy(θ); mm[i] -= k; mm[j] -= k
            H[i, j] = H[j, i] = (f(pp) - f(pm) - f(mp) + f(mm)) / (4k * k)
        end
    end
    return H
end

function fit_bd_pars(tree::TreeSim.Tree; param::AbstractBDParameterization, r::Real, ρ₀::Real=0.0, θ_init::AbstractVector{<:Real}=zeros(2))
    return fit_bd_full(tree; param, r, ρ₀, θ_init).parameters
end

function fit_bd_ensemble_mle(trees::AbstractVector{<:TreeSim.Tree}; fixed::Tuple{Symbol,<:Real}, r::Real, ρ₀::Real=0.0)
    param = RateParameterization(BDFixedSpec(fixed[1], fixed[2]))
    λ = Float64[]
    μ = Float64[]
    ψ = Float64[]
    for tree in trees
        pars = fit_bd_pars(tree; param, r, ρ₀)
        push!(λ, pars.λ)
        push!(μ, pars.μ)
        push!(ψ, pars.ψ)
    end
    return (λ=λ, μ=μ, ψ=ψ)
end
