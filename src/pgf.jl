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

function _check_series_order(smax::Integer)
    smax >= 0 || throw(ArgumentError("smax must be non-negative."))
    return nothing
end

function _check_count(name::AbstractString, x::Integer)
    x >= 0 || throw(ArgumentError("$name must be non-negative."))
    return nothing
end

function _series_constant(x::T, smax::Integer) where {T<:AbstractFloat}
    out = zeros(T, smax + 1)
    out[1] = x
    return out
end

function _series_linear(c0::T, c1::T, smax::Integer) where {T<:AbstractFloat}
    out = _series_constant(c0, smax)
    smax >= 1 && (out[2] = c1)
    return out
end

function _series_add(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    return a .+ b
end

function _series_sub(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    return a .- b
end

function _series_scale(a::AbstractVector{T}, c::T) where {T<:AbstractFloat}
    return c .* a
end

function _series_mul(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(a)
    out = zeros(T, n)
    @inbounds for i in 1:n
        acc = zero(T)
        for k in 1:i
            acc += a[k] * b[i - k + 1]
        end
        out[i] = acc
    end
    return out
end

function _series_inv(a::AbstractVector{T}) where {T<:AbstractFloat}
    iszero(a[1]) && throw(ArgumentError("cannot invert a power series with zero constant term."))
    n = length(a)
    out = zeros(T, n)
    out[1] = inv(a[1])
    @inbounds for i in 2:n
        acc = zero(T)
        for k in 2:i
            acc += a[k] * out[i - k + 1]
        end
        out[i] = -acc / a[1]
    end
    return out
end

function _series_div(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    return _series_mul(a, _series_inv(b))
end

function _series_sqrt(a::AbstractVector{T}) where {T<:AbstractFloat}
    a[1] >= zero(T) || throw(ArgumentError("cannot take real square root of a power series with negative constant term."))
    n = length(a)
    out = zeros(T, n)
    out[1] = sqrt(a[1])
    iszero(out[1]) && n > 1 && throw(ArgumentError("series square root with zero constant term is not supported."))
    @inbounds for i in 2:n
        acc = zero(T)
        for k in 2:(i - 1)
            acc += out[k] * out[i - k + 1]
        end
        out[i] = (a[i] - acc) / (2out[1])
    end
    return out
end

function _series_exp(a::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(a)
    out = zeros(T, n)
    out[1] = exp(a[1])
    @inbounds for i in 2:n
        m = i - 1
        acc = zero(T)
        for k in 1:m
            acc += k * a[k + 1] * out[m - k + 1]
        end
        out[i] = acc / m
    end
    return out
end

function _constant_rate_series(smax::Integer, tᵢ::T, tⱼ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    _check_series_order(smax)
    τ = tⱼ - tᵢ
    τ < zero(T) && throw(ArgumentError("tⱼ must be greater than or equal to tᵢ."))

    one_series = _series_constant(one(T), smax)
    zero_series = zeros(T, smax + 1)
    a = _series_linear(pars.μ, pars.r * pars.ψ, smax)
    b = _series_linear(-(pars.λ + pars.μ + pars.ψ), (one(T) - pars.r) * pars.ψ, smax)
    disc = _series_sub(_series_mul(b, b), _series_scale(a, 4pars.λ))
    Δ = _series_sqrt(disc)

    exp_term = _series_exp(_series_scale(Δ, -τ))
    numerator = _series_scale(_series_sub(one_series, exp_term), 2pars.λ)
    denominator = _series_add(_series_sub(Δ, b), _series_mul(_series_add(Δ, b), exp_term))
    γs = τ == zero(T) ? zero_series : _series_div(numerator, denominator)
    αs = _series_scale(_series_mul(a, γs), inv(pars.λ))
    βs = _series_add(one_series,
        _series_add(
            _series_scale(_series_mul(b, γs), inv(pars.λ)),
            _series_scale(_series_mul(a, _series_mul(γs, γs)), inv(pars.λ)),
        ),
    )
    return αs, βs, γs
end

function constant_rate_pgf_series(smax::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(tᵢ), typeof(tⱼ), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    return _constant_rate_series(smax, T(tᵢ), T(tⱼ), pT)
end

function constant_rate_pgf_series(smax::Integer, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return constant_rate_pgf_series(smax, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

function _s_marginal_series(smax::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    _check_series_order(smax)
    αs, βs, γs = constant_rate_pgf_series(smax, tᵢ, tⱼ, pars)
    return _pgf_sampling_marginal_series(αs, βs, γs)
end

function _pgf_sampling_marginal_series(αs::AbstractVector{T}, βs::AbstractVector{T}, γs::AbstractVector{T}) where {T<:AbstractFloat}
    smax = length(αs) - 1
    return _series_add(αs, _series_div(βs, _series_sub(_series_constant(one(T), smax), γs)))
end

function joint_pmf_NS(n::Integer, s::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    _check_count("n", n)
    _check_count("s", s)
    αs, βs, γs = constant_rate_pgf_series(s, tᵢ, tⱼ, pars)
    n == 0 && return αs[s + 1]
    f = copy(βs)
    for _ in 2:n
        f = _series_mul(γs, f)
    end
    return f[s + 1]
end

function joint_pmf_NS(n::Integer, s::Integer, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return joint_pmf_NS(n, s, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

function _joint_pmf_table_from_series(nmax::Integer, αs::AbstractVector{T}, βs::AbstractVector{T}, γs::AbstractVector{T}) where {T<:AbstractFloat}
    smax = length(αs) - 1
    out = zeros(eltype(αs), nmax + 1, smax + 1)
    out[1, :] .= αs
    nmax == 0 && return out
    f = copy(βs)
    out[2, :] .= f
    for n in 2:nmax
        f = _series_mul(γs, f)
        out[n + 1, :] .= f
    end
    return out
end

function _joint_pmf_NS_table(nmax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    _check_count("nmax", nmax)
    _check_series_order(smax)
    αs, βs, γs = constant_rate_pgf_series(smax, tᵢ, tⱼ, pars)
    return _joint_pmf_table_from_series(nmax, αs, βs, γs)
end

function _tail_overlap(n_tail::T, s_tail::T, retained_mass::T) where {T<:AbstractFloat}
    missing = max(zero(T), one(T) - retained_mass)
    overlap = n_tail + s_tail - missing
    return min(min(n_tail, s_tail), max(zero(T), overlap))
end

function _joint_pmf_NS_table_diagnostics(table::AbstractMatrix{T}, nmax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters) where {T<:AbstractFloat}
    retained_mass = sum(table)
    n_tail = n_marginal_tail(nmax, tᵢ, tⱼ, pars)
    s_tail = s_marginal_tail(smax, tᵢ, tⱼ, pars)
    overlap = _tail_overlap(T(n_tail), T(s_tail), retained_mass)
    return (
        table=table,
        nmax=nmax,
        smax=smax,
        retained_mass=retained_mass,
        missing_mass=max(zero(T), one(T) - retained_mass),
        n_tail_mass=T(n_tail),
        s_tail_mass=T(s_tail),
        n_only_tail_mass=max(zero(T), T(n_tail) - overlap),
        s_only_tail_mass=max(zero(T), T(s_tail) - overlap),
        joint_tail_overlap_mass=overlap,
    )
end

function joint_pmf_NS_table(nmax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters; diagnostics::Bool=false)
    table = _joint_pmf_NS_table(nmax, smax, tᵢ, tⱼ, pars)
    diagnostics || return table
    return _joint_pmf_NS_table_diagnostics(table, nmax, smax, tᵢ, tⱼ, pars)
end

function joint_pmf_NS_table(nmax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real; diagnostics::Bool=false)
    return joint_pmf_NS_table(nmax, smax, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r); diagnostics=diagnostics)
end

function n_marginal_pmf(n::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    return pn_birthdeath(n, tᵢ, tⱼ, pars)
end

function n_marginal_pmf(n::Integer, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return n_marginal_pmf(n, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

function s_marginal_pmf(s::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    _check_count("s", s)
    return _s_marginal_series(s, tᵢ, tⱼ, pars)[s + 1]
end

function s_marginal_pmf(s::Integer, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return s_marginal_pmf(s, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

"""
    s_marginal_tail(smax, tᵢ, tⱼ, pars)

Return the omitted S-marginal probability `P(S(tⱼ) > smax | N(tᵢ)=1, S(tᵢ)=0)`
for the constant-rate generalized birth-death-sampling process.

The tail is computed as `1 - sum(s_marginal_pmf(s), s=0:smax)` using the same
truncated coefficient construction as [`s_marginal_pmf`](@ref).
"""
function s_marginal_tail(smax::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    _check_series_order(smax)
    marginal = _s_marginal_series(smax, tᵢ, tⱼ, pars)
    tail = one(eltype(marginal)) - sum(marginal)
    roundoff = eps(eltype(marginal)) * max(one(eltype(marginal)), eltype(marginal)(length(marginal)))
    tail <= roundoff && return zero(eltype(marginal))
    return tail
end

function s_marginal_tail(smax::Integer, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return s_marginal_tail(smax, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

function n_marginal_tail(nmax::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    _check_count("nmax", nmax)
    T = promote_type(typeof(tᵢ), typeof(tⱼ), typeof(pars.λ), Float64)
    γ1 = gamma_bd(one(T), T(tᵢ), T(tⱼ), pars)
    α1 = alpha_bd(one(T), T(tᵢ), T(tⱼ), pars)
    return (one(T) - α1) * γ1^nmax
end

function n_marginal_tail(nmax::Integer, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return n_marginal_tail(nmax, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

function n_truncation(tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters; atol::Real=1e-12)
    atol >= 0 || throw(ArgumentError("atol must be non-negative."))
    tail0 = n_marginal_tail(0, tᵢ, tⱼ, pars)
    tail0 <= atol && return 0
    γ1 = gamma_bd(1.0, tᵢ, tⱼ, pars)
    γ1 <= 0 && return 0
    γ1 >= 1 && throw(ArgumentError("N marginal has no finite geometric truncation because γ(1) >= 1."))
    return max(0, ceil(Int, log(atol / tail0) / log(γ1)))
end

function n_truncation(tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real; atol::Real=1e-12)
    return n_truncation(tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r); atol=atol)
end

"""
    s_truncation(tᵢ, tⱼ, pars; atol=1e-12, max_smax=10_000)

Choose the smallest `smax` found such that `s_marginal_tail(smax, tᵢ, tⱼ, pars) <= atol`.
Throws an informative `ArgumentError` if the requested tail tolerance is not
reached by `max_smax`.
"""
function s_truncation(tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters; atol::Real=1e-12, max_smax::Integer=10_000)
    isfinite(atol) || throw(ArgumentError("atol must be finite."))
    atol >= 0 || throw(ArgumentError("atol must be non-negative."))
    _check_series_order(max_smax)

    s_marginal_tail(0, tᵢ, tⱼ, pars) <= atol && return 0

    hi = 1
    while hi < max_smax && s_marginal_tail(hi, tᵢ, tⱼ, pars) > atol
        hi = min(max_smax, 2hi)
    end

    hi_tail = s_marginal_tail(hi, tᵢ, tⱼ, pars)
    hi_tail <= atol || throw(ArgumentError("S marginal tail tolerance was not reached by max_smax=$max_smax; tail=$hi_tail."))

    lo = 0
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if s_marginal_tail(mid, tᵢ, tⱼ, pars) <= atol
            hi = mid
        else
            lo = mid
        end
    end
    return hi
end

function s_truncation(tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real; atol::Real=1e-12, max_smax::Integer=10_000)
    return s_truncation(tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r); atol=atol, max_smax=max_smax)
end

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
under the constant-rate generalized birth-death-sampling process.
"""
function unsampled_probability(tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    tⱼ <= tₖ || throw(ArgumentError("times must satisfy tⱼ <= tₖ."))
    γ0 = gamma_bd(zero(T), tⱼ, tₖ, pars)
    return one(T) - pars.ψ / pars.λ * γ0 / (one(T) - γ0)
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

"""
    reconstructed_alpha_bd(w, tᵢ, tⱼ, tₖ, pars)

Zero-count parameter `α(w,tᵢ,tⱼ,tₖ)` for the constant-rate reconstructed
process conditioned on the sampling horizon `tₖ`.
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
process conditioned on the sampling horizon `tₖ`.
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
process conditioned on the sampling horizon `tₖ`.
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
process.
"""
function reconstructed_pgf(z::Real, w::Real, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    wT, tiT, tjT, tkT, pT = _promote_reconstructed_inputs(w, tᵢ, tⱼ, tₖ, pars)
    zT = typeof(wT)(z)
    pars3 = _reconstructed_alpha_beta_gamma(wT, tiT, tjT, tkT, pT)
    return pars3.α + pars3.β * zT / _stabilize_denominator(one(typeof(wT)) - pars3.γ * zT)
end

"""
    reconstructed_xi(tᵢ, tⱼ, tₖ, pars)

Zero-count probability `ξ = α(1,tᵢ,tⱼ,tₖ)` for the reconstructed count.
"""
reconstructed_xi(tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters) =
    reconstructed_alpha_bd(1.0, tᵢ, tⱼ, tₖ, pars)

"""
    reconstructed_eta(tᵢ, tⱼ, tₖ, pars)

Geometric ratio `η = γ(1,tᵢ,tⱼ,tₖ)` for the reconstructed count.
"""
reconstructed_eta(tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters) =
    reconstructed_gamma_bd(1.0, tᵢ, tⱼ, tₖ, pars)

"""
    reconstructed_count_pmf(a, tᵢ, tⱼ, tₖ, pars)

PMF for the reconstructed lineage count `A(tⱼ)` from one lineage at `tᵢ` and
conditioning horizon `tₖ`.
"""
function reconstructed_count_pmf(a::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("a", a)
    ξ = reconstructed_xi(tᵢ, tⱼ, tₖ, pars)
    a == 0 && return ξ
    η = reconstructed_eta(tᵢ, tⱼ, tₖ, pars)
    return (1 - ξ) * (1 - η) * η^(a - 1)
end

"""
    reconstructed_pgf_series(smax, tᵢ, tⱼ, tₖ, pars)

Return truncated coefficient vectors for reconstructed `α(w)`, `β(w)`, and
`γ(w)` through `w^smax`.
"""
function reconstructed_pgf_series(smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_series_order(smax)
    _, tiT, tjT, tkT, pT = _promote_reconstructed_inputs(0.0, tᵢ, tⱼ, tₖ, pars)
    _check_time_order(tiT, tjT, tkT)
    αs, βs, γs = constant_rate_pgf_series(smax, tiT, tjT, pT)
    p = unsampled_probability(tjT, tkT, pT)
    den = _series_sub(_series_constant(one(eltype(γs)), smax), _series_scale(γs, p))
    inv_den = _series_inv(den)
    αr = _series_add(αs, _series_scale(_series_mul(βs, inv_den), p))
    βr = _series_scale(_series_mul(βs, _series_mul(inv_den, inv_den)), one(eltype(βs)) - p)
    γr = _series_sub(_series_constant(one(eltype(γs)), smax), _series_mul(_series_sub(_series_constant(one(eltype(γs)), smax), γs), inv_den))
    return αr, βr, γr
end

"""
    reconstructed_joint_pmf(a, s, tᵢ, tⱼ, tₖ, pars)

Joint PMF for reconstructed lineage count `A(tⱼ)=a` and cumulative samples
`S(tⱼ)=s`, from one lineage at `tᵢ` and sampling horizon `tₖ`.
"""
function reconstructed_joint_pmf(a::Integer, s::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("a", a)
    _check_count("s", s)
    αs, βs, γs = reconstructed_pgf_series(s, tᵢ, tⱼ, tₖ, pars)
    a == 0 && return αs[s + 1]
    f = copy(βs)
    for _ in 2:a
        f = _series_mul(γs, f)
    end
    return f[s + 1]
end

function _reconstructed_joint_pmf_table(amax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("amax", amax)
    _check_series_order(smax)
    αs, βs, γs = reconstructed_pgf_series(smax, tᵢ, tⱼ, tₖ, pars)
    return _joint_pmf_table_from_series(amax, αs, βs, γs)
end

function _reconstructed_joint_pmf_table_diagnostics(table::AbstractMatrix{T}, amax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters) where {T<:AbstractFloat}
    retained_mass = sum(table)
    count_tail = reconstructed_count_tail(amax, tᵢ, tⱼ, tₖ, pars)
    sampling_tail = reconstructed_sampling_tail(smax, tᵢ, tⱼ, tₖ, pars)
    overlap = _tail_overlap(T(count_tail), T(sampling_tail), retained_mass)
    return (
        table=table,
        amax=amax,
        smax=smax,
        retained_mass=retained_mass,
        missing_mass=max(zero(T), one(T) - retained_mass),
        count_tail_mass=T(count_tail),
        sampling_tail_mass=T(sampling_tail),
        count_only_tail_mass=max(zero(T), T(count_tail) - overlap),
        sampling_only_tail_mass=max(zero(T), T(sampling_tail) - overlap),
        joint_tail_overlap_mass=overlap,
    )
end

"""
    reconstructed_joint_pmf_table(amax, smax, tᵢ, tⱼ, tₖ, pars; diagnostics=false)

Rectangular table for `P(A(tⱼ)=a, S(tⱼ)=s)` with rows `a=0:amax` and columns
`s=0:smax`. With `diagnostics=true`, return a named tuple with retained and
tail mass accounting.
"""
function reconstructed_joint_pmf_table(amax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters; diagnostics::Bool=false)
    table = _reconstructed_joint_pmf_table(amax, smax, tᵢ, tⱼ, tₖ, pars)
    diagnostics || return table
    return _reconstructed_joint_pmf_table_diagnostics(table, amax, smax, tᵢ, tⱼ, tₖ, pars)
end

"""
    reconstructed_sampling_marginal_pmf(s, tᵢ, tⱼ, tₖ, pars)

Marginal PMF for cumulative samples `S(tⱼ)=s` under the reconstructed PGF.
"""
function reconstructed_sampling_marginal_pmf(s::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("s", s)
    αs, βs, γs = reconstructed_pgf_series(s, tᵢ, tⱼ, tₖ, pars)
    return _pgf_sampling_marginal_series(αs, βs, γs)[s + 1]
end

"""
    reconstructed_count_tail(amax, tᵢ, tⱼ, tₖ, pars)

Omitted reconstructed count tail `P(A(tⱼ) > amax)`.
"""
function reconstructed_count_tail(amax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("amax", amax)
    ξ = reconstructed_xi(tᵢ, tⱼ, tₖ, pars)
    η = reconstructed_eta(tᵢ, tⱼ, tₖ, pars)
    return (1 - ξ) * η^amax
end

"""
    reconstructed_count_truncation(tᵢ, tⱼ, tₖ, pars; atol=1e-12)

Smallest `amax` whose reconstructed count tail is at most `atol`.
"""
function reconstructed_count_truncation(tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters; atol::Real=1e-12)
    isfinite(atol) || throw(ArgumentError("atol must be finite."))
    atol >= 0 || throw(ArgumentError("atol must be non-negative."))
    tail0 = reconstructed_count_tail(0, tᵢ, tⱼ, tₖ, pars)
    tail0 <= atol && return 0
    η = reconstructed_eta(tᵢ, tⱼ, tₖ, pars)
    η <= 0 && return 0
    η < 1 || throw(ArgumentError("reconstructed count marginal has no finite geometric truncation because η >= 1."))
    return max(0, ceil(Int, log(atol / tail0) / log(η)))
end

"""
    reconstructed_sampling_tail(smax, tᵢ, tⱼ, tₖ, pars)

Omitted reconstructed sampling tail `P(S(tⱼ) > smax)`.
"""
function reconstructed_sampling_tail(smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_series_order(smax)
    αs, βs, γs = reconstructed_pgf_series(smax, tᵢ, tⱼ, tₖ, pars)
    marginal = _pgf_sampling_marginal_series(αs, βs, γs)
    tail = one(eltype(marginal)) - sum(marginal)
    roundoff = eps(eltype(marginal)) * max(one(eltype(marginal)), eltype(marginal)(length(marginal)))
    tail <= roundoff && return zero(eltype(marginal))
    return tail
end

"""
    reconstructed_sampling_truncation(tᵢ, tⱼ, tₖ, pars; atol=1e-12, max_smax=10_000)

Smallest `smax` whose reconstructed sampling tail is at most `atol`.
"""
function reconstructed_sampling_truncation(tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters; atol::Real=1e-12, max_smax::Integer=10_000)
    isfinite(atol) || throw(ArgumentError("atol must be finite."))
    atol >= 0 || throw(ArgumentError("atol must be non-negative."))
    _check_series_order(max_smax)
    reconstructed_sampling_tail(0, tᵢ, tⱼ, tₖ, pars) <= atol && return 0
    hi = 1
    while hi < max_smax && reconstructed_sampling_tail(hi, tᵢ, tⱼ, tₖ, pars) > atol
        hi = min(max_smax, 2hi)
    end
    hi_tail = reconstructed_sampling_tail(hi, tᵢ, tⱼ, tₖ, pars)
    hi_tail <= atol || throw(ArgumentError("reconstructed sampling tail tolerance was not reached by max_smax=$max_smax; tail=$hi_tail."))
    lo = 0
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if reconstructed_sampling_tail(mid, tᵢ, tⱼ, tₖ, pars) <= atol
            hi = mid
        else
            lo = mid
        end
    end
    return hi
end
