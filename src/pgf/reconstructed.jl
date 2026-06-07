function _check_time_order(tᵢ::T, tⱼ::T, tₖ::T) where {T<:AbstractFloat}
    tᵢ <= tⱼ <= tₖ || throw(ArgumentError("times must satisfy tᵢ <= tⱼ <= tₖ."))
    return nothing
end

@inline function _promote_reconstructed_inputs(w::Real, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(w), typeof(tᵢ), typeof(tⱼ), typeof(tₖ), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    return T(w), T(tᵢ), T(tⱼ), T(tₖ), pT
end

"""
    unsampled_probability(tⱼ, tₖ, pars)

Probability that one lineage extant at `tⱼ` is not sampled in `(tⱼ, tₖ]`
under the constant-rate generalized birth-death-sampling process, including
terminal sampling at `tₖ` with probability `ρ₀`.
"""
function unsampled_probability(tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    tⱼ <= tₖ || throw(ArgumentError("times must satisfy tⱼ <= tₖ."))
    z = one(T) - pars.ρ₀
    α0 = alpha_bd(zero(T), tⱼ, tₖ, pars)
    β0 = beta_bd(zero(T), tⱼ, tₖ, pars)
    γ0 = gamma_bd(zero(T), tⱼ, tₖ, pars)
    return α0 + β0 * z / (one(T) - γ0 * z)
end

function unsampled_probability(tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(tⱼ), typeof(tₖ), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    return unsampled_probability(T(tⱼ), T(tₖ), pT)
end

function unsampled_probability(tⱼ::Real, tₖ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return unsampled_probability(tⱼ, tₖ, ConstantRateBDParameters(λ, μ, ψ, r))
end

function _one_minus_unsampled_probability(tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    q = one(T) - unsampled_probability(tⱼ, tₖ, pars)
    cutoff = sqrt(eps(T))
    q > cutoff || throw(ArgumentError("1 - unsampled_probability(tⱼ,tₖ) is too small for transformed rates."))
    return q
end

"""
    transformed_birth_rate(tⱼ, tₖ, pars)

Generalized birth rate `λ * (1 - p(tⱼ,tₖ))` for the reconstructed process.
"""
function transformed_birth_rate(tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    return pars.λ * _one_minus_unsampled_probability(tⱼ, tₖ, pars)
end

function transformed_birth_rate(tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(tⱼ), typeof(tₖ), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    return transformed_birth_rate(T(tⱼ), T(tₖ), pT)
end

function transformed_birth_rate(tⱼ::Real, tₖ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return transformed_birth_rate(tⱼ, tₖ, ConstantRateBDParameters(λ, μ, ψ, r))
end

"""
    transformed_death_rate(tⱼ, tₖ, pars)

Generalized death rate `ψ * (r + (1-r) * p(tⱼ,tₖ)) / (1 - p(tⱼ,tₖ))`
for the reconstructed process.
"""
function transformed_death_rate(tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    p = unsampled_probability(tⱼ, tₖ, pars)
    q = _one_minus_unsampled_probability(tⱼ, tₖ, pars)
    return pars.ψ * (pars.r + (one(T) - pars.r) * p) / q
end

function transformed_death_rate(tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(tⱼ), typeof(tₖ), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    return transformed_death_rate(T(tⱼ), T(tₖ), pT)
end

function transformed_death_rate(tⱼ::Real, tₖ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return transformed_death_rate(tⱼ, tₖ, ConstantRateBDParameters(λ, μ, ψ, r))
end

"""
    transformed_sampling_rate(tⱼ, tₖ, pars)

Generalized sampling rate `ψ / (1 - p(tⱼ,tₖ))` for the reconstructed process.
"""
function transformed_sampling_rate(tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    return pars.ψ / _one_minus_unsampled_probability(tⱼ, tₖ, pars)
end

function transformed_sampling_rate(tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(tⱼ), typeof(tₖ), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    return transformed_sampling_rate(T(tⱼ), T(tₖ), pT)
end

function transformed_sampling_rate(tⱼ::Real, tₖ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return transformed_sampling_rate(tⱼ, tₖ, ConstantRateBDParameters(λ, μ, ψ, r))
end

function _reconstructed_alpha_beta_gamma(w::T, tᵢ::T, tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    _check_time_order(tᵢ, tⱼ, tₖ)
    p = unsampled_probability(tⱼ, tₖ, pars)
    αij = alpha_bd(w, tᵢ, tⱼ, pars)
    βij = beta_bd(w, tᵢ, tⱼ, pars)
    γij = gamma_bd(w, tᵢ, tⱼ, pars)
    den = one(T) - γij * p
    den = _stabilize_denominator(den)
    return (
        α=αij + βij * p / den,
        β=βij * (one(T) - p) / (den * den),
        γ=one(T) - (one(T) - γij) / den,
    )
end

function _conditioned_reconstructed_alpha_beta_gamma(w::T, tᵢ::T, tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    raw = _reconstructed_alpha_beta_gamma(w, tᵢ, tⱼ, tₖ, pars)
    pᵢ = unsampled_probability(tᵢ, tₖ, pars)
    qᵢ = _one_minus_unsampled_probability(tᵢ, tₖ, pars)
    return (
        α=(raw.α - pᵢ) / qᵢ,
        β=raw.β / qᵢ,
        γ=raw.γ,
    )
end

"""
    reconstructed_alpha_bd(w, tᵢ, tⱼ, tₖ, pars)

Zero-count parameter `α(w,tᵢ,tⱼ,tₖ)` for the constant-rate reconstructed
raw thinned hidden-lineage PGF
`Gᵢⱼ(pⱼᵏ + qⱼᵏ z, w)`. This depends on the sampling horizon `tₖ` but is not
conditioned on the initial lineage being reconstructed (`Aᵢᵏ = 1`).
"""
function reconstructed_alpha_bd(w::T, tᵢ::T, tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    return _reconstructed_alpha_beta_gamma(w, tᵢ, tⱼ, tₖ, pars).α
end

function reconstructed_alpha_bd(w::Real, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    wT, tiT, tjT, tkT, pT = _promote_reconstructed_inputs(w, tᵢ, tⱼ, tₖ, pars)
    return reconstructed_alpha_bd(wT, tiT, tjT, tkT, pT)
end

"""
    reconstructed_beta_bd(w, tᵢ, tⱼ, tₖ, pars)

Geometric mass parameter `β(w,tᵢ,tⱼ,tₖ)` for the constant-rate reconstructed
raw thinned hidden-lineage PGF
`Gᵢⱼ(pⱼᵏ + qⱼᵏ z, w)`. This depends on the sampling horizon `tₖ` but is not
conditioned on the initial lineage being reconstructed (`Aᵢᵏ = 1`).
"""
function reconstructed_beta_bd(w::T, tᵢ::T, tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    return _reconstructed_alpha_beta_gamma(w, tᵢ, tⱼ, tₖ, pars).β
end

function reconstructed_beta_bd(w::Real, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    wT, tiT, tjT, tkT, pT = _promote_reconstructed_inputs(w, tᵢ, tⱼ, tₖ, pars)
    return reconstructed_beta_bd(wT, tiT, tjT, tkT, pT)
end

"""
    reconstructed_gamma_bd(w, tᵢ, tⱼ, tₖ, pars)

Geometric ratio parameter `γ(w,tᵢ,tⱼ,tₖ)` for the constant-rate reconstructed
raw thinned hidden-lineage PGF
`Gᵢⱼ(pⱼᵏ + qⱼᵏ z, w)`. This depends on the sampling horizon `tₖ` but is not
conditioned on the initial lineage being reconstructed (`Aᵢᵏ = 1`).
"""
function reconstructed_gamma_bd(w::T, tᵢ::T, tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    return _reconstructed_alpha_beta_gamma(w, tᵢ, tⱼ, tₖ, pars).γ
end

function reconstructed_gamma_bd(w::Real, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    wT, tiT, tjT, tkT, pT = _promote_reconstructed_inputs(w, tᵢ, tⱼ, tₖ, pars)
    return reconstructed_gamma_bd(wT, tiT, tjT, tkT, pT)
end

"""
    reconstructed_pgf(z, w, tᵢ, tⱼ, tₖ, pars)

Single-lineage PGF `α + β*z/(1 - γ*z)` for the constant-rate reconstructed
raw thinned hidden-lineage quantity `Gᵢⱼ(pⱼᵏ + qⱼᵏ z, w)`. This is the
backward-compatible `reconstructed_*` convention and is not normalized by
`qᵢᵏ` or conditioned on `Aᵢᵏ = 1`.
"""
function reconstructed_pgf(z::Real, w::Real, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    wT, tiT, tjT, tkT, pT = _promote_reconstructed_inputs(w, tᵢ, tⱼ, tₖ, pars)
    zT = typeof(wT)(z)
    pars3 = _reconstructed_alpha_beta_gamma(wT, tiT, tjT, tkT, pT)
    return pars3.α + pars3.β * zT / _stabilize_denominator(one(typeof(wT)) - pars3.γ * zT)
end

"""
    reconstructed_xi(tᵢ, tⱼ, tₖ, pars)

Zero-count probability `ξ = α(1,tᵢ,tⱼ,tₖ)` for the raw thinned reconstructed
count. This is not conditioned on `Aᵢᵏ = 1`.
"""
reconstructed_xi(tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters) =
    reconstructed_alpha_bd(1.0, tᵢ, tⱼ, tₖ, pars)

"""
    reconstructed_eta(tᵢ, tⱼ, tₖ, pars)

Geometric ratio `η = γ(1,tᵢ,tⱼ,tₖ)` for the raw thinned reconstructed count.
This is not conditioned on `Aᵢᵏ = 1`.
"""
reconstructed_eta(tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters) =
    reconstructed_gamma_bd(1.0, tᵢ, tⱼ, tₖ, pars)

"""
    reconstructed_count_pmf(a, tᵢ, tⱼ, tₖ, pars)

Raw thinned PMF for the reconstructed lineage count `Aⱼᵏ` from one lineage at
`tᵢ`, given the observation horizon `tₖ`. For probabilities conditioned on the
initial lineage being reconstructed, `P(Aⱼᵏ = a | Aᵢᵏ = 1)`, prefer
[`conditioned_reconstructed_count_pmf`](@ref).
"""
function reconstructed_count_pmf(a::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("a", a)
    ξ = reconstructed_xi(tᵢ, tⱼ, tₖ, pars)
    a == 0 && return ξ
    η = reconstructed_eta(tᵢ, tⱼ, tₖ, pars)
    return (1 - ξ) * (1 - η) * η^(a - 1)
end
