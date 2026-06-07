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
