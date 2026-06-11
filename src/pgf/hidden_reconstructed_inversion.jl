function _check_hidden_reconstructed_inputs(n::Integer, a::Integer, tᵢ::T, tⱼ::T, tₗ::T) where {T<:AbstractFloat}
    _check_count("n", n)
    _check_count("a", a)
    _check_time_order(tᵢ, tⱼ, tₗ)
    return nothing
end

function _logbinomial(n::Integer, k::Integer)
    0 <= k <= n || return -Inf
    kk = min(k, n - k)
    acc = 0.0
    @inbounds for i in 1:kk
        acc += log(n - kk + i) - log(i)
    end
    return acc
end

@inline function _clamp_probability(x::T; atol::T=sqrt(eps(T))) where {T<:AbstractFloat}
    if -atol <= x <= zero(T)
        return zero(T)
    elseif one(T) <= x <= one(T) + atol
        return one(T)
    end
    return x
end

function _nonnegative_difference(x::T, y::T; rtol::T=sqrt(eps(T)), atol::T=eps(T)) where {T<:AbstractFloat}
    diff = x - y
    scale = max(one(T), abs(x), abs(y))
    tol = max(atol, rtol * scale)
    diff >= -tol || throw(ArgumentError("subtractive term is materially negative; got $diff with tolerance $tol."))
    return max(zero(T), diff)
end

"""
    hidden_reconstructed_unsampled_probability(tⱼ, tₗ, pars)

Return `qⱼˡ = Gⱼˡ(1 - ρₗ, 0)`, the probability that one hidden lineage alive
at `tⱼ` leaves no sampled descendant by the observation horizon `tₗ`.
"""
hidden_reconstructed_unsampled_probability(tⱼ::Real, tₗ::Real, pars::ConstantRateBDParameters) =
    unsampled_probability(tⱼ, tₗ, pars)

"""
    hidden_count_given_reconstructed_count_logpmf(n, a, tᵢ, tⱼ, tₗ, pars)

Log probability for
`P(Nⱼ = n | Aⱼˡ = a, Aᵢˡ = 1)`, the hidden lineage count at `tⱼ`
conditional on the reconstructed count at `tⱼ` and on the initial lineage at
`tᵢ` having at least one sampled descendant by `tₗ`.

For `a >= 1`, `Nⱼ - a` is a negative binomial count of failures before
`a + 1` successes with success probability `1 - γᵢⱼ(1) qⱼˡ`.
"""
function hidden_count_given_reconstructed_count_logpmf(
    n::Integer,
    a::Integer,
    tᵢ::T,
    tⱼ::T,
    tₗ::T,
    pars::ConstantRateBDParameters{T};
    cancellation_rtol::T=sqrt(eps(T)),
    cancellation_atol::T=eps(T),
) where {T<:AbstractFloat}
    _check_hidden_reconstructed_inputs(n, a, tᵢ, tⱼ, tₗ)

    if a >= 1
        n < a && return T(-Inf)
        q = _clamp_probability(hidden_reconstructed_unsampled_probability(tⱼ, tₗ, pars))
        θ = _clamp_probability(gamma_bd(one(T), tᵢ, tⱼ, pars) * q)
        success = _clamp_probability(one(T) - θ)
        if θ == zero(T)
            return n == a ? zero(T) : T(-Inf)
        elseif success == zero(T)
            return T(-Inf)
        end
        return T(_logbinomial(n, a)) + (a + 1) * log(success) + (n - a) * log(θ)
    end

    q = _clamp_probability(hidden_reconstructed_unsampled_probability(tⱼ, tₗ, pars))
    α1 = alpha_bd(one(T), tᵢ, tⱼ, pars)
    β1 = beta_bd(one(T), tᵢ, tⱼ, pars)
    γ1 = gamma_bd(one(T), tᵢ, tⱼ, pars)
    α0 = alpha_bd(zero(T), tᵢ, tⱼ, pars)
    β0 = beta_bd(zero(T), tᵢ, tⱼ, pars)
    γ0 = gamma_bd(zero(T), tᵢ, tⱼ, pars)

    den1 = _stabilize_denominator(one(T) - γ1 * q)
    den0 = _stabilize_denominator(one(T) - γ0 * q)
    H0 = α1 - α0 + β1 * q / den1 - β0 * q / den0
    H0 > zero(T) || throw(ArgumentError("conditioning event Aⱼˡ = 0 has non-positive probability."))

    if n == 0
        mass = _nonnegative_difference(α1, α0; rtol=cancellation_rtol, atol=cancellation_atol)
        return mass == zero(T) ? T(-Inf) : log(mass / H0)
    end

    term1 = β1 * γ1^(n - 1)
    term0 = β0 * γ0^(n - 1)
    mass = q^n * _nonnegative_difference(term1, term0; rtol=cancellation_rtol, atol=cancellation_atol)
    return mass == zero(T) ? T(-Inf) : log(mass / H0)
end

function hidden_count_given_reconstructed_count_logpmf(n::Integer, a::Integer, tᵢ::Real, tⱼ::Real, tₗ::Real, pars::ConstantRateBDParameters; kwargs...)
    T = promote_type(typeof(tᵢ), typeof(tⱼ), typeof(tₗ), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    return hidden_count_given_reconstructed_count_logpmf(n, a, T(tᵢ), T(tⱼ), T(tₗ), pT; kwargs...)
end

"""
    hidden_count_given_reconstructed_count_pmf(n, a, tᵢ, tⱼ, tₗ, pars)

Probability mass
`P(Nⱼ = n | Aⱼˡ = a, Aᵢˡ = 1)`. Impossible states, such as `n < a` when
`a >= 1`, return zero probability.

For `a >= 1`, `Nⱼ - a ~ NegBin(a + 1, 1 - γᵢⱼ(1) qⱼˡ)`, where the negative
binomial counts failures before `a + 1` successes.
"""
function hidden_count_given_reconstructed_count_pmf(n::Integer, a::Integer, tᵢ::Real, tⱼ::Real, tₗ::Real, pars::ConstantRateBDParameters; kwargs...)
    ℓp = hidden_count_given_reconstructed_count_logpmf(n, a, tᵢ, tⱼ, tₗ, pars; kwargs...)
    return isfinite(ℓp) ? exp(ℓp) : 0.0
end

"""
    hidden_count_given_reconstructed_count_pmf_table(a, tᵢ, tⱼ, tₗ, pars; nmax)

Return a vector of named tuples with columns `n`, `a`, `probability`,
`log_probability`, and `case` for the truncated conditional distribution
`P(Nⱼ = n | Aⱼˡ = a, Aᵢˡ = 1)`.
"""
function hidden_count_given_reconstructed_count_pmf_table(
    a::Integer,
    tᵢ::Real,
    tⱼ::Real,
    tₗ::Real,
    pars::ConstantRateBDParameters;
    nmax::Integer,
    kwargs...,
)
    _check_count("a", a)
    _check_count("nmax", nmax)
    rows = NamedTuple[]
    for n in 0:nmax
        ℓp = hidden_count_given_reconstructed_count_logpmf(n, a, tᵢ, tⱼ, tₗ, pars; kwargs...)
        prob = isfinite(ℓp) ? exp(ℓp) : 0.0
        case = a >= 1 ? "a_ge_1" : (n == 0 ? "a0_n0" : "a0_nge1")
        push!(rows, (n=n, a=a, probability=prob, log_probability=ℓp, case=case))
    end
    return rows
end
