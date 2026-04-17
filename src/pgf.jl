@inline function bd_coefficients(w::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    isfinite(w) || throw(ArgumentError("w must be finite."))

    a = muladd(pars.r * pars.ψ, w, pars.μ)
    b = muladd((one(T) - pars.r) * pars.ψ, w, -(pars.λ + pars.μ + pars.ψ))
    disc = muladd(-4a, pars.λ, b * b)
    disc = ifelse(disc < zero(T), zero(T), disc)
    Δ = sqrt(disc)
    return a, b, Δ
end

@inline function bd_coefficients(w::T, λ::T, μ::T, ψ::T, r::T) where {T<:AbstractFloat}
    return bd_coefficients(w, ConstantRateBDParameters{T}(λ, μ, ψ, r))
end

@inline function bd_coefficients(w::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(w), typeof(pars.λ), Float64)
    return bd_coefficients(T(w), ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀)))
end

@inline function bd_coefficients(w::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return bd_coefficients(w, ConstantRateBDParameters(λ, μ, ψ, r))
end

const _bd_coefficients = bd_coefficients

@inline function _stabilize_denominator(den::T) where {T<:AbstractFloat}
    scale = max(abs(den), one(T))
    cutoff = eps(T) * scale
    return abs(den) < cutoff ? copysign(cutoff, den) : den
end

@inline function gamma_bd(w::T, tᵢ::T, tⱼ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    τ = tⱼ - tᵢ
    τ <= zero(T) && return zero(T)

    _, b, Δ = bd_coefficients(w, pars)
    x = Δ * τ

    if abs(x) <= sqrt(eps(T))
        num = 2pars.λ * (x - x * x / 2)
        den = (Δ - b) + (Δ + b) * (one(T) - x + x * x / 2)
        return num / _stabilize_denominator(den)
    end

    em = exp(-x)
    num = 2pars.λ * (-expm1(-x))
    den = (Δ - b) + (Δ + b) * em
    return num / _stabilize_denominator(den)
end

@inline function gamma_bd(w::T, tᵢ::T, tⱼ::T, λ::T, μ::T, ψ::T, r::T) where {T<:AbstractFloat}
    return gamma_bd(w, tᵢ, tⱼ, ConstantRateBDParameters{T}(λ, μ, ψ, r))
end

@inline function gamma_bd(w::Real, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(w), typeof(tᵢ), typeof(tⱼ), typeof(pars.λ), Float64)
    return gamma_bd(T(w), T(tᵢ), T(tⱼ), ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀)))
end

@inline function gamma_bd(w::Real, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return gamma_bd(w, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

const γ = gamma_bd

@inline function alpha_bd(w::T, tᵢ::T, tⱼ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    a = muladd(pars.r * pars.ψ, w, pars.μ)
    return a / pars.λ * gamma_bd(w, tᵢ, tⱼ, pars)
end

@inline function alpha_bd(w::T, tᵢ::T, tⱼ::T, λ::T, μ::T, ψ::T, r::T) where {T<:AbstractFloat}
    return alpha_bd(w, tᵢ, tⱼ, ConstantRateBDParameters{T}(λ, μ, ψ, r))
end

@inline function alpha_bd(w::Real, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(w), typeof(tᵢ), typeof(tⱼ), typeof(pars.λ), Float64)
    return alpha_bd(T(w), T(tᵢ), T(tⱼ), ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀)))
end

@inline function alpha_bd(w::Real, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return alpha_bd(w, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

const α = alpha_bd

@inline function beta_bd(w::T, tᵢ::T, tⱼ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    a, b, _ = bd_coefficients(w, pars)
    γij = gamma_bd(w, tᵢ, tⱼ, pars)
    return one(T) + b / pars.λ * γij + a / pars.λ * γij * γij
end

@inline function beta_bd(w::T, tᵢ::T, tⱼ::T, λ::T, μ::T, ψ::T, r::T) where {T<:AbstractFloat}
    return beta_bd(w, tᵢ, tⱼ, ConstantRateBDParameters{T}(λ, μ, ψ, r))
end

@inline function beta_bd(w::Real, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(w), typeof(tᵢ), typeof(tⱼ), typeof(pars.λ), Float64)
    return beta_bd(T(w), T(tᵢ), T(tⱼ), ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀)))
end

@inline function beta_bd(w::Real, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return beta_bd(w, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

const β = beta_bd

@inline function pn_birthdeath(n::S, tᵢ::T, tⱼ::T, pars::ConstantRateBDParameters{T}) where {S<:Integer,T<:AbstractFloat}
    n < 0 && throw(ArgumentError("n must be non-negative."))
    n == 0 && return alpha_bd(one(T), tᵢ, tⱼ, pars)
    return beta_bd(one(T), tᵢ, tⱼ, pars) * gamma_bd(one(T), tᵢ, tⱼ, pars)^(n - 1)
end

@inline function pn_birthdeath(n::S, tᵢ::T, tⱼ::T, λ::T, μ::T, ψ::T, r::T) where {S<:Integer,T<:AbstractFloat}
    return pn_birthdeath(n, tᵢ, tⱼ, ConstantRateBDParameters{T}(λ, μ, ψ, r))
end

@inline function pn_birthdeath(n::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(tᵢ), typeof(tⱼ), typeof(pars.λ), Float64)
    return pn_birthdeath(n, T(tᵢ), T(tⱼ), ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀)))
end

@inline function pn_birthdeath(n::Integer, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return pn_birthdeath(n, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

@inline function pn_birthdeath(n::AbstractVector{S}, tᵢ::T, tⱼ::T, pars::ConstantRateBDParameters{T}) where {S<:Integer,T<:AbstractFloat}
    return [pn_birthdeath(nᵢ, tᵢ, tⱼ, pars) for nᵢ in n]
end

@inline function pn_birthdeath(n::AbstractVector{S}, tᵢ::T, tⱼ::T, λ::T, μ::T, ψ::T, r::T) where {S<:Integer,T<:AbstractFloat}
    return pn_birthdeath(n, tᵢ, tⱼ, ConstantRateBDParameters{T}(λ, μ, ψ, r))
end

@inline function pn_birthdeath(n::AbstractVector{S}, tᵢ::T, tⱼ::AbstractVector{T}, pars::ConstantRateBDParameters{T}) where {S<:Integer,T<:AbstractFloat}
    return transpose(reduce(hcat, [pn_birthdeath(n, tᵢ, tⱼᵢ, pars) for tⱼᵢ in tⱼ]))
end

@inline function pn_birthdeath(n::AbstractVector{S}, tᵢ::T, tⱼ::AbstractVector{T}, λ::T, μ::T, ψ::T, r::T) where {S<:Integer,T<:AbstractFloat}
    return pn_birthdeath(n, tᵢ, tⱼ, ConstantRateBDParameters{T}(λ, μ, ψ, r))
end

const pₙ = pn_birthdeath
