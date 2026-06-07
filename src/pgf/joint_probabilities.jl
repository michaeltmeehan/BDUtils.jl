"""
    reconstructed_pgf_series(smax, tᵢ, tⱼ, tₖ, pars)

Return truncated coefficient vectors for reconstructed `α(w)`, `β(w)`, and
`γ(w)` through `w^smax` under the raw thinned `reconstructed_*` convention.
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
    conditioned_reconstructed_pgf_series(smax, tᵢ, tⱼ, tₖ, pars)

Return truncated coefficient vectors for conditioned reconstructed `α̃(w)`,
`β̃(w)`, and `γ̃(w)` through `w^smax`, normalized by `qᵢᵏ` after conditioning
on `Aᵢᵏ = 1`.
"""
function conditioned_reconstructed_pgf_series(smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    αs, βs, γs = reconstructed_pgf_series(smax, tᵢ, tⱼ, tₖ, pars)
    _, tiT, _, tkT, pT = _promote_reconstructed_inputs(0.0, tᵢ, tⱼ, tₖ, pars)
    pᵢ = unsampled_probability(tiT, tkT, pT)
    qᵢ = _one_minus_unsampled_probability(tiT, tkT, pT)
    αc = copy(αs)
    αc[1] -= pᵢ
    αc ./= qᵢ
    βc = βs ./ qᵢ
    return αc, βc, γs
end

"""
    reconstructed_joint_pmf(a, s, tᵢ, tⱼ, tₖ, pars)

Joint PMF for reconstructed lineage count `A(tⱼ)=a` and cumulative samples
`S(tⱼ)=s`, from one lineage at `tᵢ` and sampling horizon `tₖ`, under the raw
thinned `reconstructed_*` convention. This is not conditioned on `Aᵢᵏ = 1`.
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

"""
    conditioned_reconstructed_joint_pmf(a, s, tᵢ, tⱼ, tₖ, pars)

Joint PMF for reconstructed lineage count `A(tⱼ)=a` and cumulative samples
`S(tⱼ)=s`, conditioned on the initial lineage being reconstructed
(`Aᵢᵏ = 1`).
"""
function conditioned_reconstructed_joint_pmf(a::Integer, s::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("a", a)
    _check_count("s", s)
    αs, βs, γs = conditioned_reconstructed_pgf_series(s, tᵢ, tⱼ, tₖ, pars)
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

function _conditioned_reconstructed_joint_pmf_table(amax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("amax", amax)
    _check_series_order(smax)
    αs, βs, γs = conditioned_reconstructed_pgf_series(smax, tᵢ, tⱼ, tₖ, pars)
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

function _conditioned_reconstructed_joint_pmf_table_diagnostics(table::AbstractMatrix{T}, amax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters) where {T<:AbstractFloat}
    retained_mass = sum(table)
    count_tail = conditioned_reconstructed_count_tail(amax, tᵢ, tⱼ, tₖ, pars)
    sampling_tail = conditioned_reconstructed_sampling_tail(smax, tᵢ, tⱼ, tₖ, pars)
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
`s=0:smax` under the raw thinned `reconstructed_*` convention. It is not
conditioned on `Aᵢᵏ = 1`. With `diagnostics=true`, return a named tuple with
retained and tail mass accounting.
"""
function reconstructed_joint_pmf_table(amax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters; diagnostics::Bool=false)
    table = _reconstructed_joint_pmf_table(amax, smax, tᵢ, tⱼ, tₖ, pars)
    diagnostics || return table
    return _reconstructed_joint_pmf_table_diagnostics(table, amax, smax, tᵢ, tⱼ, tₖ, pars)
end

"""
    conditioned_reconstructed_joint_pmf_table(amax, smax, tᵢ, tⱼ, tₖ, pars; diagnostics=false)

Rectangular table for `P(Aⱼᵏ=a, Sᵢⱼ=s | Aᵢᵏ=1)` with rows `a=0:amax` and
columns `s=0:smax`. With `diagnostics=true`, return a named tuple with
retained and tail mass accounting.
"""
function conditioned_reconstructed_joint_pmf_table(amax::Integer, smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters; diagnostics::Bool=false)
    table = _conditioned_reconstructed_joint_pmf_table(amax, smax, tᵢ, tⱼ, tₖ, pars)
    diagnostics || return table
    return _conditioned_reconstructed_joint_pmf_table_diagnostics(table, amax, smax, tᵢ, tⱼ, tₖ, pars)
end

"""
    reconstructed_sampling_marginal_pmf(s, tᵢ, tⱼ, tₖ, pars)

Marginal PMF for cumulative samples `S(tⱼ)=s` under the raw thinned
`reconstructed_*` PGF. This is not conditioned on `Aᵢᵏ = 1`.
"""
function reconstructed_sampling_marginal_pmf(s::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("s", s)
    αs, βs, γs = reconstructed_pgf_series(s, tᵢ, tⱼ, tₖ, pars)
    return _pgf_sampling_marginal_series(αs, βs, γs)[s + 1]
end

"""
    conditioned_reconstructed_sampling_marginal_pmf(s, tᵢ, tⱼ, tₖ, pars)

Marginal PMF for cumulative samples `S(tⱼ)=s` under the conditioned
reconstructed PGF, given `Aᵢᵏ = 1`.
"""
function conditioned_reconstructed_sampling_marginal_pmf(s::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("s", s)
    αs, βs, γs = conditioned_reconstructed_pgf_series(s, tᵢ, tⱼ, tₖ, pars)
    return _pgf_sampling_marginal_series(αs, βs, γs)[s + 1]
end

"""
    reconstructed_count_tail(amax, tᵢ, tⱼ, tₖ, pars)

Omitted raw thinned reconstructed count tail `P(Aⱼᵏ > amax)`. This is not
conditioned on `Aᵢᵏ = 1`.
"""
function reconstructed_count_tail(amax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("amax", amax)
    ξ = reconstructed_xi(tᵢ, tⱼ, tₖ, pars)
    η = reconstructed_eta(tᵢ, tⱼ, tₖ, pars)
    return (1 - ξ) * η^amax
end

"""
    conditioned_reconstructed_count_tail(amax, tᵢ, tⱼ, tₖ, pars)

Omitted conditioned reconstructed count tail
`P(Aⱼᵏ > amax | Aᵢᵏ=1)`.
"""
function conditioned_reconstructed_count_tail(amax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("amax", amax)
    ξ = conditioned_reconstructed_xi(tᵢ, tⱼ, tₖ, pars)
    η = conditioned_reconstructed_eta(tᵢ, tⱼ, tₖ, pars)
    return (1 - ξ) * η^amax
end

"""
    reconstructed_count_truncation(tᵢ, tⱼ, tₖ, pars; atol=1e-12)

Smallest `amax` whose raw thinned reconstructed count tail is at most `atol`.
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
    conditioned_reconstructed_count_truncation(tᵢ, tⱼ, tₖ, pars; atol=1e-12)

Smallest `amax` whose conditioned reconstructed count tail
`P(Aⱼᵏ > amax | Aᵢᵏ=1)` is at most `atol`.
"""
function conditioned_reconstructed_count_truncation(tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters; atol::Real=1e-12)
    isfinite(atol) || throw(ArgumentError("atol must be finite."))
    atol >= 0 || throw(ArgumentError("atol must be non-negative."))
    tail0 = conditioned_reconstructed_count_tail(0, tᵢ, tⱼ, tₖ, pars)
    tail0 <= atol && return 0
    η = conditioned_reconstructed_eta(tᵢ, tⱼ, tₖ, pars)
    η <= 0 && return 0
    η < 1 || throw(ArgumentError("conditioned reconstructed count marginal has no finite geometric truncation because η >= 1."))
    return max(0, ceil(Int, log(atol / tail0) / log(η)))
end

"""
    reconstructed_sampling_tail(smax, tᵢ, tⱼ, tₖ, pars)

Omitted raw thinned reconstructed sampling tail `P(S(tⱼ) > smax)`. This is
not conditioned on `Aᵢᵏ = 1`.
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
    conditioned_reconstructed_sampling_tail(smax, tᵢ, tⱼ, tₖ, pars)

Omitted conditioned reconstructed sampling tail
`P(S(tⱼ) > smax | Aᵢᵏ=1)`.
"""
function conditioned_reconstructed_sampling_tail(smax::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_series_order(smax)
    αs, βs, γs = conditioned_reconstructed_pgf_series(smax, tᵢ, tⱼ, tₖ, pars)
    marginal = _pgf_sampling_marginal_series(αs, βs, γs)
    tail = one(eltype(marginal)) - sum(marginal)
    roundoff = eps(eltype(marginal)) * max(one(eltype(marginal)), eltype(marginal)(length(marginal)))
    tail <= roundoff && return zero(eltype(marginal))
    return tail
end

"""
    reconstructed_sampling_truncation(tᵢ, tⱼ, tₖ, pars; atol=1e-12, max_smax=10_000)

Smallest `smax` whose raw thinned reconstructed sampling tail is at most
`atol`.
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

"""
    conditioned_reconstructed_sampling_truncation(tᵢ, tⱼ, tₖ, pars; atol=1e-12, max_smax=10_000)

Smallest `smax` whose conditioned reconstructed sampling tail is at most
`atol`, given `Aᵢᵏ = 1`.
"""
function conditioned_reconstructed_sampling_truncation(tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters; atol::Real=1e-12, max_smax::Integer=10_000)
    isfinite(atol) || throw(ArgumentError("atol must be finite."))
    atol >= 0 || throw(ArgumentError("atol must be non-negative."))
    _check_series_order(max_smax)
    conditioned_reconstructed_sampling_tail(0, tᵢ, tⱼ, tₖ, pars) <= atol && return 0
    hi = 1
    while hi < max_smax && conditioned_reconstructed_sampling_tail(hi, tᵢ, tⱼ, tₖ, pars) > atol
        hi = min(max_smax, 2hi)
    end
    hi_tail = conditioned_reconstructed_sampling_tail(hi, tᵢ, tⱼ, tₖ, pars)
    hi_tail <= atol || throw(ArgumentError("conditioned reconstructed sampling tail tolerance was not reached by max_smax=$max_smax; tail=$hi_tail."))
    lo = 0
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if conditioned_reconstructed_sampling_tail(mid, tᵢ, tⱼ, tₖ, pars) <= atol
            hi = mid
        else
            lo = mid
        end
    end
    return hi
end
