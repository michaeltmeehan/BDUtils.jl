@inline function bd_coefficients(w::T, λ::T, μ::T, ψ::T, r::T) where {T<:AbstractFloat}
    a = muladd(r * ψ, w, μ)
    b = muladd((one(T) - r) * ψ, w, -(λ + μ + ψ))
    disc = muladd(-4a, λ, b * b)
    disc = ifelse(disc < zero(T), zero(T), disc)
    Δ = sqrt(disc)
    return a, b, Δ
end

const _bd_coefficients = bd_coefficients

@inline function _stabilize_denominator(den::T) where {T<:AbstractFloat}
    scale = max(abs(den), one(T))
    cutoff = eps(T) * scale
    return abs(den) < cutoff ? copysign(cutoff, den) : den
end

@inline function gamma_bd(w::T, tᵢ::T, tⱼ::T, λ::T, μ::T, ψ::T, r::T) where {T<:AbstractFloat}
    τ = tⱼ - tᵢ
    τ <= zero(T) && return zero(T)

    _, b, Δ = bd_coefficients(w, λ, μ, ψ, r)
    x = Δ * τ

    if abs(x) <= sqrt(eps(T))
        num = 2λ * (x - x * x / 2)
        den = (Δ - b) + (Δ + b) * (one(T) - x + x * x / 2)
        return num / _stabilize_denominator(den)
    end

    em = exp(-x)
    num = 2λ * (-expm1(-x))
    den = (Δ - b) + (Δ + b) * em
    return num / _stabilize_denominator(den)
end

@inline function gamma_bd(w::Real, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    T = promote_type(typeof(w), typeof(tᵢ), typeof(tⱼ), typeof(λ), typeof(μ), typeof(ψ), typeof(r), Float64)
    return gamma_bd(T(w), T(tᵢ), T(tⱼ), T(λ), T(μ), T(ψ), T(r))
end

const γ = gamma_bd

@inline function alpha_bd(w::T, tᵢ::T, tⱼ::T, λ::T, μ::T, ψ::T, r::T) where {T<:AbstractFloat}
    a = muladd(r * ψ, w, μ)
    return a / λ * gamma_bd(w, tᵢ, tⱼ, λ, μ, ψ, r)
end

@inline function alpha_bd(w::Real, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    T = promote_type(typeof(w), typeof(tᵢ), typeof(tⱼ), typeof(λ), typeof(μ), typeof(ψ), typeof(r), Float64)
    return alpha_bd(T(w), T(tᵢ), T(tⱼ), T(λ), T(μ), T(ψ), T(r))
end

const α = alpha_bd

@inline function beta_bd(w::T, tᵢ::T, tⱼ::T, λ::T, μ::T, ψ::T, r::T) where {T<:AbstractFloat}
    a, b, _ = bd_coefficients(w, λ, μ, ψ, r)
    γij = gamma_bd(w, tᵢ, tⱼ, λ, μ, ψ, r)
    return one(T) + b / λ * γij + a / λ * γij * γij
end

@inline function beta_bd(w::Real, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    T = promote_type(typeof(w), typeof(tᵢ), typeof(tⱼ), typeof(λ), typeof(μ), typeof(ψ), typeof(r), Float64)
    return beta_bd(T(w), T(tᵢ), T(tⱼ), T(λ), T(μ), T(ψ), T(r))
end

const β = beta_bd

@inline function pn_birthdeath(n::S, tᵢ::T, tⱼ::T, λ::T, μ::T, ψ::T, r::T) where {S<:Integer,T<:AbstractFloat}
    n < 0 && throw(ArgumentError("n must be non-negative."))
    n == 0 && return alpha_bd(one(T), tᵢ, tⱼ, λ, μ, ψ, r)
    return beta_bd(one(T), tᵢ, tⱼ, λ, μ, ψ, r) * gamma_bd(one(T), tᵢ, tⱼ, λ, μ, ψ, r)^(n - 1)
end

@inline function pn_birthdeath(n::Integer, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    T = promote_type(typeof(tᵢ), typeof(tⱼ), typeof(λ), typeof(μ), typeof(ψ), typeof(r), Float64)
    return pn_birthdeath(n, T(tᵢ), T(tⱼ), T(λ), T(μ), T(ψ), T(r))
end

@inline function pn_birthdeath(n::AbstractVector{S}, tᵢ::T, tⱼ::T, λ::T, μ::T, ψ::T, r::T) where {S<:Integer,T<:AbstractFloat}
    return [pn_birthdeath(nᵢ, tᵢ, tⱼ, λ, μ, ψ, r) for nᵢ in n]
end

@inline function pn_birthdeath(n::AbstractVector{S}, tᵢ::T, tⱼ::AbstractVector{T}, λ::T, μ::T, ψ::T, r::T) where {S<:Integer,T<:AbstractFloat}
    return transpose(reduce(hcat, [pn_birthdeath(n, tᵢ, tⱼᵢ, λ, μ, ψ, r) for tⱼᵢ in tⱼ]))
end

const pₙ = pn_birthdeath
