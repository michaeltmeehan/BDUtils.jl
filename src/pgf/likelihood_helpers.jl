function _binomial_coefficient(T::Type, n::Integer, k::Integer)
    _check_count("n", n)
    _check_count("k", k)
    k > n && return zero(T)
    kk = min(k, n - k)
    out = one(T)
    for j in 1:kk
        out *= T(n - kk + j) / T(j)
    end
    return out
end

function _falling_factorial(T::Type, b::Integer, c::Integer)
    _check_count("b", b)
    _check_count("c", c)
    c > b && return zero(T)
    out = one(T)
    for k in 0:(c - 1)
        out *= T(b - k)
    end
    return out
end

_falling_factorial(b::Integer, c::Integer) = _falling_factorial(Float64, b, c)

function _log_binomial_coefficient(T::Type, n::Integer, k::Integer)
    _check_count("n", n)
    _check_count("k", k)
    k > n && return T(-Inf)
    kk = min(k, n - k)
    out = zero(T)
    for j in 1:kk
        out += log(T(n - kk + j)) - log(T(j))
    end
    return out
end

function _log_falling_factorial(T::Type, b::Integer, c::Integer)
    _check_count("b", b)
    _check_count("c", c)
    c > b && return T(-Inf)
    out = zero(T)
    for k in 0:(c - 1)
        out += log(T(b - k))
    end
    return out
end

function _finite_nonnegative_or_throw(label::AbstractString, value)
    isfinite(value) && value >= zero(value) && return value
    throw(ErrorException("$label must be finite and nonnegative; got $value."))
end

function _finite_nonnegative_vector_or_throw(label::AbstractString, values)
    for value in values
        _finite_nonnegative_or_throw(label, value)
    end
    return values
end

function _normalize_likelihood_vector!(label::AbstractString, values::AbstractVector{T}) where {T<:AbstractFloat}
    _finite_nonnegative_vector_or_throw(label, values)
    scale = maximum(values; init=zero(T))
    scale > zero(T) || return zero(T)
    values ./= scale
    return log(scale)
end

function _log_power_nonnegative(x::T, n::Integer) where {T<:AbstractFloat}
    _check_count("exponent", n)
    n == 0 && return zero(T)
    x > zero(T) && return T(n) * log(x)
    return T(-Inf)
end

function _exp_log_nonnegative(label::AbstractString, logvalue::T) where {T<:AbstractFloat}
    logvalue == T(-Inf) && return zero(T)
    logvalue <= log(floatmax(T)) ||
        throw(ErrorException("$label overflows Float64 on the probability scale; use the cached log-likelihood path."))
    value = exp(logvalue)
    return _finite_nonnegative_or_throw(label, value)
end

function _no_sample_reconstructed_kernel(
    ti::Real,
    tj::Real,
    tℓ::Real,
    a::Integer,
    b::Integer,
    pars::ConstantRateBDParameters,
)
    _check_count("a", a)
    _check_count("b", b)
    b >= a >= 1 || return zero(promote_type(typeof(ti), typeof(tj), typeof(tℓ), typeof(pars.λ), Float64))
    _, tiT, tjT, tlT, pT = _promote_reconstructed_inputs(0.0, ti, tj, tℓ, pars)
    β = conditioned_reconstructed_beta_bd(zero(tiT), tiT, tjT, tlT, pT)
    γ = conditioned_reconstructed_gamma_bd(zero(tiT), tiT, tjT, tlT, pT)
    _finite_nonnegative_or_throw("conditioned reconstructed beta", β)
    _finite_nonnegative_or_throw("conditioned reconstructed gamma", γ)
    T = typeof(β)
    logkernel = _log_binomial_coefficient(T, b - 1, a - 1) +
        _log_power_nonnegative(β, a) +
        _log_power_nonnegative(γ, b - a)
    return _exp_log_nonnegative("no-sample reconstructed kernel", logkernel)
end

function _grouped_removal_sampling_jump(
    b::Integer,
    c::Integer,
    ψ̃::Real;
    labelled_samples::Bool=false,
)
    _check_count("b", b)
    _check_count("c", c)
    T = promote_type(typeof(ψ̃), Float64)
    c > b && return zero(T)
    ψT = T(ψ̃)
    _finite_nonnegative_or_throw("transformed sampling rate", ψT)
    logcoefficient = labelled_samples ? _log_falling_factorial(T, b, c) : _log_binomial_coefficient(T, b, c)
    logjump = logcoefficient + _log_power_nonnegative(ψT, c)
    return _exp_log_nonnegative("grouped removal sampling jump", logjump)
end

"""
    no_sample_probability_conditioned(ti, tj, tℓ, pars)

Compute `P(ΔS_ij = 0 | A_i^ℓ = 1)`, the probability of no serial samples in
`(ti,tj]` given that the lineage at `ti` is reconstructed by horizon `tℓ`.
"""
function no_sample_probability_conditioned(
    ti::Real,
    tj::Real,
    tℓ::Real,
    pars::ConstantRateBDParameters,
)
    _, tiT, tjT, tlT, pT = _promote_reconstructed_inputs(0.0, ti, tj, tℓ, pars)
    pᵢ = unsampled_probability(tiT, tlT, pT)
    qᵢ = one(tiT) - pᵢ
    α0 = alpha_bd(zero(tiT), tiT, tjT, pT)
    β0 = beta_bd(zero(tiT), tiT, tjT, pT)
    γ0 = gamma_bd(zero(tiT), tiT, tjT, pT)
    G10 = α0 + β0 / _stabilize_denominator(one(tiT) - γ0)
    return (G10 - pᵢ) / _stabilize_denominator(qᵢ)
end

"""
    terminal_count_transition(ti, tℓ, a, cℓ, pars)

Return `P(A_tℓ^ℓ = cℓ, ΔS_iℓ = 0 | A_i^ℓ = a)` for a terminal sampling
count at `tℓ`.
"""
function terminal_count_transition(
    ti::Real,
    tℓ::Real,
    a::Integer,
    cℓ::Integer,
    pars::ConstantRateBDParameters,
)
    _check_count("a", a)
    _check_count("cℓ", cℓ)
    ti == tℓ && return a == cℓ ? one(promote_type(typeof(ti), typeof(tℓ), typeof(pars.λ), Float64)) :
        zero(promote_type(typeof(ti), typeof(tℓ), typeof(pars.λ), Float64))
    return _no_sample_reconstructed_kernel(ti, tℓ, tℓ, a, cℓ, pars)
end

function _validate_grouped_sampling_inputs(
    t0::Real,
    sampling_times::AbstractVector{<:Real},
    sample_counts::AbstractVector{<:Integer},
    pars::ConstantRateBDParameters,
    tℓ::Real,
)
    length(sampling_times) == length(sample_counts) ||
        throw(ArgumentError("sampling_times and sample_counts must have equal length."))
    !isempty(sampling_times) || throw(ArgumentError("sampling_times must be nonempty."))
    all(isfinite, sampling_times) || throw(ArgumentError("sampling_times must be finite."))
    for i in 2:length(sampling_times)
        sampling_times[i - 1] < sampling_times[i] ||
            throw(ArgumentError("sampling_times must be strictly increasing unique grouped times."))
    end
    for c in sample_counts
        _check_count("sample count", c)
    end
    sum(sample_counts) >= 1 || throw(ArgumentError("sum(sample_counts) must be at least 1."))
    t0 < first(sampling_times) || throw(ArgumentError("t0 must be less than first(sampling_times)."))
    last(sampling_times) <= tℓ || throw(ArgumentError("last(sampling_times) must be <= tℓ."))
    isapprox(pars.r, one(pars.r)) ||
        throw(ArgumentError("grouped_sampling_time_likelihood currently supports removal sampling only; pars.r must be approximately 1."))
    return nothing
end

function _validate_sampling_time_likelihood_inputs(
    t0::Real,
    sampling_times::AbstractVector{<:Real},
    sample_counts::AbstractVector{<:Integer},
    terminal_count::Integer,
    pars::ConstantRateBDParameters,
    tℓ::Real,
    max_count::Union{Nothing,Integer},
)
    length(sampling_times) == length(sample_counts) ||
        throw(ArgumentError("sampling_times and sample_counts must have equal length."))
    all(isfinite, sampling_times) || throw(ArgumentError("sampling_times must be finite."))
    t0 < tℓ || throw(ArgumentError("t0 must be less than tℓ."))
    for i in 2:length(sampling_times)
        sampling_times[i - 1] < sampling_times[i] ||
            throw(ArgumentError("sampling_times must be strictly increasing unique grouped times."))
    end
    for c in sample_counts
        _check_count("sample count", c)
    end
    _check_count("terminal_count", terminal_count)
    if !isempty(sampling_times)
        t0 < first(sampling_times) ||
            throw(ArgumentError("t0 must be less than first(sampling_times)."))
        last(sampling_times) < tℓ ||
            throw(ArgumentError("last(sampling_times) must be less than tℓ for sampling_time_likelihood."))
    end
    isapprox(pars.r, one(pars.r)) ||
        throw(ArgumentError("sampling_time_likelihood currently supports removal sampling only; pars.r must be approximately 1."))
    if max_count !== nothing
        _check_count("max_count", max_count)
    end
    return nothing
end

struct SamplingTimeLikelihoodCache{T<:AbstractFloat,P<:ConstantRateBDParameters{T}}
    sampling_times::Vector{T}
    sample_counts::Vector{Int}
    terminal_count::Int
    first_sampling_time::T
    pars::P
    tℓ::T
    downstream::Vector{T}
    max_count::Int
    downstream_log_scale::T
    labelled_samples::Bool
    terminal_sampling::Bool
    terminal_condition::Symbol
    mode::Symbol
end

function SamplingTimeLikelihoodCache(
    sampling_times::Vector{T},
    sample_counts::Vector{Int},
    terminal_count::Int,
    first_sampling_time::T,
    pars::P,
    tℓ::T,
    downstream::Vector{T},
    max_count::Int,
    labelled_samples::Bool,
    terminal_sampling::Bool,
    terminal_condition::Symbol,
    mode::Symbol,
) where {T<:AbstractFloat,P<:ConstantRateBDParameters{T}}
    return SamplingTimeLikelihoodCache(
        sampling_times,
        sample_counts,
        terminal_count,
        first_sampling_time,
        pars,
        tℓ,
        downstream,
        max_count,
        zero(T),
        labelled_samples,
        terminal_sampling,
        terminal_condition,
        mode,
    )
end

struct OriginTimeMLEResult{T<:AbstractFloat}
    t0_hat::T
    loglikelihood::T
    lower::T
    upper::T
    converged::Bool
    iterations::Int
    n_evaluations::Int
    status::Symbol
end

function _sampling_time_remaining_counts(sample_counts::AbstractVector{<:Integer}, terminal_count::Integer)
    M = length(sample_counts)
    remaining = zeros(Int, M + 1)
    remaining[M + 1] = Int(terminal_count)
    for m in M:-1:1
        remaining[m] = remaining[m + 1] + Int(sample_counts[m])
    end
    return remaining
end

function _sampling_time_terminal_downstream(
    u::T,
    tℓ::T,
    terminal_count::Integer,
    after_max::Integer,
    pars::ConstantRateBDParameters{T};
    terminal_sampling::Bool,
    terminal_condition::Symbol,
) where {T<:AbstractFloat}
    h = zeros(T, after_max + 1)
    if terminal_sampling
        @inbounds for a in 1:after_max
            h[a + 1] = terminal_count_transition(u, tℓ, a, terminal_count, pars)
        end
        return h
    end
    if terminal_condition == :censored
        η = no_sample_probability_conditioned(u, tℓ, tℓ, pars)
        @inbounds for a in 0:after_max
            h[a + 1] = η^a
        end
        return h
    end
    terminal_condition == :any && begin
        fill!(h, one(T))
        return h
    end
    throw(ArgumentError("unsupported terminal_condition=$terminal_condition when terminal_sampling=false; expected :censored or :any."))
end

function _cached_exact_sampling_time_downstream(
    sampling_times::AbstractVector{T},
    sample_counts::AbstractVector{<:Integer},
    terminal_count::Integer,
    pars::ConstantRateBDParameters{T},
    tℓ::T;
    labelled_samples::Bool,
    terminal_sampling::Bool,
    terminal_condition::Symbol,
) where {T<:AbstractFloat}
    M = length(sampling_times)
    log_scale = zero(T)
    remaining = _sampling_time_remaining_counts(sample_counts, terminal_count)
    h = _sampling_time_terminal_downstream(
        sampling_times[end],
        tℓ,
        terminal_count,
        max(remaining[M + 1], 1),
        pars;
        terminal_sampling=terminal_sampling,
        terminal_condition=terminal_condition,
    )
    log_scale += _normalize_likelihood_vector!("terminal downstream vector", h)

    for m in M:-1:1
        c = sample_counts[m]
        before_max = max(remaining[m], 1)
        after_max = max(remaining[m + 1], 1)
        h_pre = zeros(T, before_max + 1)
        ψ̃ = transformed_sampling_rate(sampling_times[m], tℓ, pars)
        @inbounds for b in c:before_max
            d = b - c
            d <= after_max || continue
            h_pre[b + 1] += _grouped_removal_sampling_jump(b, c, ψ̃; labelled_samples=labelled_samples) *
                h[d + 1]
        end
        log_scale += _normalize_likelihood_vector!("sampling downstream vector", h_pre)
        m == 1 && return h_pre, log_scale

        h_prev = zeros(T, max(remaining[m], 1) + 1)
        u = sampling_times[m - 1]
        ti = sampling_times[m]
        @inbounds for a in 1:(length(h_prev) - 1)
            for b in a:before_max
                h_prev[a + 1] += _no_sample_reconstructed_kernel(u, ti, tℓ, a, b, pars) * h_pre[b + 1]
            end
        end
        log_scale += _normalize_likelihood_vector!("propagated downstream vector", h_prev)
        h = h_prev
    end
    throw(ArgumentError("sampling_times must be nonempty."))
end

function _cached_grouped_sampling_time_downstream(
    sampling_times::AbstractVector{T},
    sample_counts::AbstractVector{<:Integer},
    pars::ConstantRateBDParameters{T},
    tℓ::T;
    labelled_samples::Bool,
    terminal_condition::Symbol,
) where {T<:AbstractFloat}
    M = length(sampling_times)
    log_scale = zero(T)
    remaining = _sampling_time_remaining_counts(sample_counts, 0)
    h = zeros(T, 1)
    terminal_condition == :terminated && (h[1] = one(T))
    terminal_condition == :any && (h[1] = one(T))
    terminal_condition in (:terminated, :any) ||
        throw(ArgumentError("unsupported terminal_condition=$terminal_condition; expected :terminated or :any."))
    tl_work = sampling_times[end] == tℓ ? tℓ + 16sqrt(eps(T)) * max(one(T), abs(tℓ)) : tℓ
    log_scale += _normalize_likelihood_vector!("terminal downstream vector", h)

    for m in M:-1:1
        c = sample_counts[m]
        before_max = remaining[m]
        after_max = remaining[m + 1]
        h_pre = zeros(T, before_max + 1)
        ψ̃ = transformed_sampling_rate(sampling_times[m], tl_work, pars)
        @inbounds for b in c:before_max
            d = b - c
            d <= after_max || continue
            h_pre[b + 1] += _grouped_removal_sampling_jump(b, c, ψ̃; labelled_samples=labelled_samples) *
                h[d + 1]
        end
        log_scale += _normalize_likelihood_vector!("sampling downstream vector", h_pre)
        m == 1 && return h_pre, log_scale

        h_prev = zeros(T, remaining[m] + 1)
        u = sampling_times[m - 1]
        ti = sampling_times[m]
        @inbounds for a in 1:(length(h_prev) - 1)
            for b in a:before_max
                h_prev[a + 1] += _no_sample_reconstructed_kernel(u, ti, tl_work, a, b, pars) * h_pre[b + 1]
            end
        end
        log_scale += _normalize_likelihood_vector!("propagated downstream vector", h_prev)
        h = h_prev
    end
    throw(ArgumentError("sampling_times must be nonempty."))
end

function _grouped_sampling_time_filter(
    t0::Real,
    sampling_times::AbstractVector{<:Real},
    sample_counts::AbstractVector{<:Integer},
    pars::ConstantRateBDParameters;
    tℓ::Union{Nothing,Real}=nothing,
    labelled_samples::Bool=false,
)
    isempty(sampling_times) && throw(ArgumentError("sampling_times must be nonempty."))
    tl = tℓ === nothing ? last(sampling_times) : tℓ
    T = promote_type(typeof(t0), eltype(sampling_times), typeof(tl), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    times = T.(sampling_times)
    counts = Int.(sample_counts)
    t0T = T(t0)
    tlT = T(tl)

    _validate_grouped_sampling_inputs(t0T, times, counts, pT, tlT)
    tl_work = times[end] == tlT ? tlT + 16sqrt(eps(T)) * max(one(T), abs(tlT)) : tlT

    remaining = zeros(Int, length(counts) + 1)
    for i in length(counts):-1:1
        remaining[i] = remaining[i + 1] + counts[i]
    end

    f = zeros(T, remaining[1] + 1)
    f[2] = one(T)
    u = t0T

    for i in eachindex(times)
        ti = times[i]
        c = counts[i]
        before_max = remaining[i]
        g = zeros(T, before_max + 1)
        @inbounds for a in 0:(length(f) - 1)
            fa = f[a + 1]
            iszero(fa) && continue
            for b in a:before_max
                g[b + 1] += fa * _no_sample_reconstructed_kernel(u, ti, tl_work, a, b, pT)
            end
        end

        after_max = remaining[i + 1]
        next = zeros(T, after_max + 1)
        ψ̃ = transformed_sampling_rate(ti, tl_work, pT)
        @inbounds for b in c:before_max
            d = b - c
            d <= after_max || continue
            next[d + 1] += g[b + 1] *
                _grouped_removal_sampling_jump(b, c, ψ̃; labelled_samples=labelled_samples)
        end
        f = next
        u = ti
    end
    return f
end

"""
    grouped_sampling_time_likelihood(t0, sampling_times, sample_counts, pars;
        tℓ=nothing, labelled_samples=false, terminal_condition=:terminated)

Compute an unnormalized marginal likelihood/density for unique grouped
sampling times of the reconstructed process under removal sampling. The
`sample_counts` vector gives the number of samples at each grouped time in
`sampling_times`. The calculation marginalizes over unobserved reconstructed
births using no-sample propagators and finite forward filtering over
reconstructed lineage counts.

By default `tℓ` is set to the final sampling time, so the supplied grouped
sampling times are treated as the complete reconstructed sampling set over
`(t0,tℓ]`, and `terminal_condition=:terminated` returns the mass with zero
reconstructed lineages after the final grouped event; the terminal transformed
sampling rate is evaluated by a right-limit when the final grouped time equals
`tℓ`. Use
`terminal_condition=:any` to sum over the final filtering state. Grouped counts
are unlabelled by default, using `binomial(b,c)` in the sampling jump;
`labelled_samples=true` switches this coefficient to the falling factorial
`(b)_c`. Zero-count grouped times are allowed and are propagated harmlessly, but
the total sample count must be at least one. Currently only `pars.r ≈ 1` is
supported.
"""
function grouped_sampling_time_likelihood(
    t0::Real,
    sampling_times::AbstractVector{<:Real},
    sample_counts::AbstractVector{<:Integer},
    pars::ConstantRateBDParameters;
    tℓ::Union{Nothing,Real}=nothing,
    labelled_samples::Bool=false,
    terminal_condition::Symbol=:terminated,
)
    f = _grouped_sampling_time_filter(
        t0,
        sampling_times,
        sample_counts,
        pars;
        tℓ=tℓ,
        labelled_samples=labelled_samples,
    )
    terminal_condition == :terminated && return f[1]
    terminal_condition == :any && return sum(f)
    throw(ArgumentError("unsupported terminal_condition=$terminal_condition; expected :terminated or :any."))
end

"""
    cache_sampling_time_likelihood(sampling_times, sample_counts, terminal_count, pars; tℓ, ...)

Precompute the part of `sampling_time_likelihood` that is independent of the
origin time `t0`. The returned `SamplingTimeLikelihoodCache` stores a
downstream vector beginning immediately before the first grouped serial
sampling update, so repeated evaluation only recomputes propagation from `t0`
to the first sampling time.

The cache is intentionally conservative: it supports the same constant-rate,
single-initial-lineage, removal-sampling likelihood as `sampling_time_likelihood`
and requires at least one serial sampling time.
"""
function cache_sampling_time_likelihood(
    sampling_times::AbstractVector{<:Real},
    sample_counts::AbstractVector{<:Integer},
    terminal_count::Integer,
    pars::ConstantRateBDParameters;
    tℓ::Real,
    labelled_samples::Bool=false,
    terminal_sampling::Bool=true,
    terminal_condition::Symbol=:observed,
    atol::Real=1e-12,
    max_count::Union{Nothing,Integer}=nothing,
)
    T = promote_type(
        eltype(sampling_times),
        typeof(tℓ),
        typeof(atol),
        typeof(pars.λ),
        Float64,
    )
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    times = T.(sampling_times)
    counts = Int.(sample_counts)
    terminal = Int(terminal_count)
    tlT = T(tℓ)
    isempty(times) && throw(ArgumentError("sampling_times must be nonempty for cached origin-time evaluation."))
    _validate_sampling_time_likelihood_inputs(prevfloat(first(times)), times, counts, terminal, pT, tlT, max_count)
    if !terminal_sampling && terminal != 0
        throw(ArgumentError("terminal_count must be 0 when terminal_sampling=false."))
    end
    terminal_sampling && terminal_condition == :observed ||
        !terminal_sampling && terminal_condition in (:censored, :any) ||
        throw(ArgumentError("unsupported terminal/condition combination for sampling_time_likelihood cache."))
    remaining = _sampling_time_remaining_counts(counts, terminal)
    if max_count !== nothing && max_count < remaining[1]
        throw(ArgumentError("max_count must be at least the total observed sample count."))
    end
    downstream, downstream_log_scale = _cached_exact_sampling_time_downstream(
        times,
        counts,
        terminal,
        pT,
        tlT;
        labelled_samples=labelled_samples,
        terminal_sampling=terminal_sampling,
        terminal_condition=terminal_condition,
    )
    return SamplingTimeLikelihoodCache(
        times,
        counts,
        terminal,
        first(times),
        pT,
        tlT,
        downstream,
        length(downstream) - 1,
        downstream_log_scale,
        labelled_samples,
        terminal_sampling,
        terminal_condition,
        :sampling_time,
    )
end

"""
    cache_sampling_time_likelihood(sampling_times, sample_counts, pars;
        tℓ=nothing, labelled_samples=false, terminal_condition=:terminated)

Precompute the `t0`-independent downstream vector for
`grouped_sampling_time_likelihood`.
"""
function cache_sampling_time_likelihood(
    sampling_times::AbstractVector{<:Real},
    sample_counts::AbstractVector{<:Integer},
    pars::ConstantRateBDParameters;
    tℓ::Union{Nothing,Real}=nothing,
    labelled_samples::Bool=false,
    terminal_condition::Symbol=:terminated,
)
    isempty(sampling_times) && throw(ArgumentError("sampling_times must be nonempty."))
    tl = tℓ === nothing ? last(sampling_times) : tℓ
    T = promote_type(eltype(sampling_times), typeof(tl), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    times = T.(sampling_times)
    counts = Int.(sample_counts)
    tlT = T(tl)
    _validate_grouped_sampling_inputs(prevfloat(first(times)), times, counts, pT, tlT)
    terminal_condition in (:terminated, :any) ||
        throw(ArgumentError("unsupported terminal_condition=$terminal_condition; expected :terminated or :any."))
    downstream, downstream_log_scale = _cached_grouped_sampling_time_downstream(
        times,
        counts,
        pT,
        tlT;
        labelled_samples=labelled_samples,
        terminal_condition=terminal_condition,
    )
    return SamplingTimeLikelihoodCache(
        times,
        counts,
        0,
        first(times),
        pT,
        tlT,
        downstream,
        length(downstream) - 1,
        downstream_log_scale,
        labelled_samples,
        false,
        terminal_condition,
        :grouped_sampling_time,
    )
end

function _sampling_time_scaled_likelihood(cache::SamplingTimeLikelihoodCache{T}, t0::Real) where {T<:AbstractFloat}
    isfinite(t0) || return zero(T)
    t0T = T(t0)
    t0T < cache.first_sampling_time || return zero(T)
    t0T < cache.tℓ || return zero(T)
    horizon = cache.mode == :grouped_sampling_time && last(cache.sampling_times) == cache.tℓ ?
        cache.tℓ + 16sqrt(eps(T)) * max(one(T), abs(cache.tℓ)) :
        cache.tℓ
    lik = zero(T)
    @inbounds for b in 1:cache.max_count
        kernel = _no_sample_reconstructed_kernel(t0T, cache.first_sampling_time, horizon, 1, b, cache.pars)
        term = kernel * cache.downstream[b + 1]
        _finite_nonnegative_or_throw("propagator-vector product term", term)
        lik += term
    end
    _finite_nonnegative_or_throw("propagator-vector product", lik)
    return lik
end

function sampling_time_likelihood(cache::SamplingTimeLikelihoodCache{T}, t0::Real) where {T<:AbstractFloat}
    scaled_likelihood = _sampling_time_scaled_likelihood(cache, t0)
    scaled_likelihood > zero(T) || return zero(T)
    loglikelihood = cache.downstream_log_scale + log(scaled_likelihood)
    loglikelihood <= log(floatmax(T)) || return T(Inf)
    return exp(loglikelihood)
end

function sampling_time_loglikelihood(cache::SamplingTimeLikelihoodCache{T}, t0::Real) where {T<:AbstractFloat}
    scaled_likelihood = _sampling_time_scaled_likelihood(cache, t0)
    scaled_likelihood > zero(T) || return T(-Inf)
    loglikelihood = cache.downstream_log_scale + log(scaled_likelihood)
    isfinite(loglikelihood) || throw(ErrorException("sampling-time log-likelihood must be finite after scaled cache evaluation; got $loglikelihood."))
    return loglikelihood
end

function origin_time_loglikelihood_profile(cache::SamplingTimeLikelihoodCache, t0_grid)
    t0s = collect(t0_grid)
    loglikelihoods = [sampling_time_loglikelihood(cache, t0) for t0 in t0s]
    finite_loglikelihoods = filter(isfinite, loglikelihoods)
    max_loglikelihood = isempty(finite_loglikelihoods) ? -Inf : maximum(finite_loglikelihoods)
    delta_loglikelihoods = isfinite(max_loglikelihood) ?
        [isfinite(ll) ? ll - max_loglikelihood : -Inf for ll in loglikelihoods] :
        fill(-Inf, length(loglikelihoods))
    return (
        t0=t0s,
        loglikelihood=loglikelihoods,
        delta_loglikelihood=delta_loglikelihoods,
    )
end

function _origin_time_default_lower(cache::SamplingTimeLikelihoodCache{T}) where {T<:AbstractFloat}
    scale_floor = sqrt(eps(T)) * max(
        one(T),
        abs(cache.first_sampling_time),
        abs(last(cache.sampling_times)),
        abs(cache.tℓ),
    )
    observed_span = max(
        cache.tℓ - cache.first_sampling_time,
        last(cache.sampling_times) - cache.first_sampling_time,
        scale_floor,
    )
    return cache.first_sampling_time - T(10) * observed_span
end

function _origin_time_default_upper(cache::SamplingTimeLikelihoodCache{T}) where {T<:AbstractFloat}
    return prevfloat(cache.first_sampling_time)
end

function _origin_time_bounds(cache::SamplingTimeLikelihoodCache{T}, lower, upper) where {T<:AbstractFloat}
    lo = lower === nothing ? _origin_time_default_lower(cache) : T(lower)
    hi = upper === nothing ? _origin_time_default_upper(cache) : T(upper)
    isfinite(lo) || throw(ArgumentError("lower must be finite for origin_time_mle."))
    isfinite(hi) || throw(ArgumentError("upper must be finite for origin_time_mle."))
    lo < hi || throw(ArgumentError("lower must be less than upper for origin_time_mle."))
    hi < cache.first_sampling_time ||
        throw(ArgumentError("upper must be less than the first sampling time for origin_time_mle."))
    return lo, hi
end

function _origin_time_search(
    cache::SamplingTimeLikelihoodCache{T},
    lower::T,
    upper::T;
    tolerance::Real=sqrt(eps(T)),
    maxiter::Integer=1_000,
    boundary_tol::Union{Nothing,Real}=nothing,
) where {T<:AbstractFloat}
    tol = T(tolerance)
    tol > zero(T) || throw(ArgumentError("tolerance must be positive for origin_time_mle."))
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive for origin_time_mle."))
    btol = boundary_tol === nothing ? sqrt(tol) : T(boundary_tol)
    btol >= zero(T) || throw(ArgumentError("boundary_tol must be nonnegative for origin_time_mle."))

    n_evaluations = 0
    function objective(t0::T)
        n_evaluations += 1
        t0 < cache.first_sampling_time || return T(-Inf)
        value = sampling_time_loglikelihood(cache, t0)
        return isfinite(value) ? T(value) : T(-Inf)
    end

    φ = (sqrt(T(5)) - one(T)) / T(2)
    a = lower
    b = upper
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    fa = objective(a)
    fb = objective(b)
    fc = objective(c)
    fd = objective(d)
    iterations = 0

    while iterations < maxiter && (b - a) > tol * max(one(T), abs((a + b) / T(2)))
        iterations += 1
        if fc < fd
            a = c
            c = d
            fc = fd
            d = a + φ * (b - a)
            fd = objective(d)
        else
            b = d
            d = c
            fd = fc
            c = b - φ * (b - a)
            fc = objective(c)
        end
    end

    mid = (a + b) / T(2)
    fmid = objective(mid)
    candidates = ((lower, fa), (upper, fb), (c, fc), (d, fd), (mid, fmid))
    t0_hat, loglikelihood = candidates[1]
    for candidate in candidates[2:end]
        if candidate[2] > loglikelihood
            t0_hat, loglikelihood = candidate
        end
    end

    converged = (b - a) <= tol * max(one(T), abs(mid))
    status = converged ? :converged : :maxiter
    if !isfinite(loglikelihood)
        converged = false
        status = :nonfinite_objective
    else
        boundary_scale = max(one(T), abs(lower), abs(upper), upper - lower)
        if abs(t0_hat - lower) <= btol * boundary_scale
            status = :lower_bound
        elseif abs(t0_hat - upper) <= btol * boundary_scale
            status = :upper_bound
        end
    end

    return OriginTimeMLEResult(
        t0_hat,
        loglikelihood,
        lower,
        upper,
        converged,
        iterations,
        n_evaluations,
        status,
    )
end

"""
    origin_time_mle(cache::SamplingTimeLikelihoodCache; lower=nothing, upper=nothing,
        tolerance=sqrt(eps(T)), maxiter=1_000, boundary_tol=nothing)

Estimate the origin time `t0` by maximizing
`sampling_time_loglikelihood(cache, t0)` with fixed birth, death, sampling,
removal, and terminal sampling parameters. The optimizer uses the cached
factorized likelihood evaluator; it does not rebuild the full likelihood chain.

The search is bounded and requires `upper < cache.first_sampling_time`. If no
bounds are supplied, `upper` defaults to the previous floating-point value below
the first sampling time and `lower` defaults to ten observed time spans before
the first sampling time. These defaults are conservative; explicit scientific
bounds are recommended.
"""
function origin_time_mle(
    cache::SamplingTimeLikelihoodCache;
    lower=nothing,
    upper=nothing,
    tolerance::Real=sqrt(eps(eltype(cache.sampling_times))),
    maxiter::Integer=1_000,
    boundary_tol::Union{Nothing,Real}=nothing,
)
    lo, hi = _origin_time_bounds(cache, lower, upper)
    return _origin_time_search(
        cache,
        lo,
        hi;
        tolerance=tolerance,
        maxiter=maxiter,
        boundary_tol=boundary_tol,
    )
end

"""
    origin_time_mle(sampling_times, pars; sample_counts=nothing, terminal_count=nothing, kwargs...)

Construct a `SamplingTimeLikelihoodCache` and call `origin_time_mle(cache; ...)`.
When `sample_counts` is omitted, each supplied sampling time is assigned count
one. If `terminal_count` is omitted, the grouped sampling-time cache is used;
otherwise `tℓ` must be supplied and the exact sampling-time cache with terminal
sampling options is used.
"""
function origin_time_mle(
    sampling_times::AbstractVector{<:Real},
    pars::ConstantRateBDParameters;
    sample_counts=nothing,
    terminal_count=nothing,
    tℓ::Union{Nothing,Real}=nothing,
    labelled_samples::Bool=false,
    terminal_sampling::Bool=true,
    terminal_condition::Union{Nothing,Symbol}=nothing,
    atol::Real=1e-12,
    max_count::Union{Nothing,Integer}=nothing,
    lower=nothing,
    upper=nothing,
    tolerance::Real=sqrt(eps(Float64)),
    maxiter::Integer=1_000,
    boundary_tol::Union{Nothing,Real}=nothing,
)
    counts = sample_counts === nothing ? ones(Int, length(sampling_times)) : sample_counts
    condition = terminal_condition === nothing ?
        (terminal_count === nothing ? :terminated : :observed) :
        terminal_condition
    cache = if terminal_count === nothing
        cache_sampling_time_likelihood(
            sampling_times,
            counts,
            pars;
            tℓ=tℓ,
            labelled_samples=labelled_samples,
            terminal_condition=condition,
        )
    else
        tℓ === nothing && throw(ArgumentError("tℓ must be supplied when terminal_count is supplied."))
        cache_sampling_time_likelihood(
            sampling_times,
            counts,
            terminal_count,
            pars;
            tℓ=tℓ,
            labelled_samples=labelled_samples,
            terminal_sampling=terminal_sampling,
            terminal_condition=condition,
            atol=atol,
            max_count=max_count,
        )
    end
    return origin_time_mle(
        cache;
        lower=lower,
        upper=upper,
        tolerance=tolerance,
        maxiter=maxiter,
        boundary_tol=boundary_tol,
    )
end

"""
    sampling_time_likelihood(t0, sampling_times, sample_counts, terminal_count, pars;
        tℓ, labelled_samples=false, terminal_sampling=true,
        terminal_condition=:observed, atol=1e-12, max_count=nothing,
        diagnostics=false)

Compute the forward-filtering likelihood for grouped serial sampling times and
a terminal sample count under the constant-rate birth-death-sampling PGF. The
returned value is a density with respect to the serial sampling times and a
probability mass with respect to sample counts, conditioned on
`A_t0^ℓ = 1`.

`sampling_times` are strictly increasing serial grouped sampling times in
`(t0,tℓ)`. `sample_counts[m]` is the number of samples in the group at
`sampling_times[m]`; zero-count entries are interpreted as explicit checkpoints
where no serial sample was observed. Such checkpoints are propagated through
the no-sample kernel and therefore constrain the likelihood relative to
omitting the checkpoint. Grouped counts are unlabelled by default, using
`binomial(b,c)` in the removal jump. Set `labelled_samples=true` to use the
falling factorial `(b)_c`, appropriate when the members of each grouped sample
are labelled/order-distinguishable.

When `terminal_sampling=true`, `terminal_count` is the observed number of
terminal samples at exactly `tℓ`, including Bernoulli present-day sampling with
probability `pars.ρ₀`. Thus `pars.ρ₀` only enters through reconstruction to
the horizon and the terminal count transition. When `terminal_sampling=false`,
there is no observed terminal sampling event and `terminal_count` must be zero;
`terminal_condition=:censored` multiplies by the probability of no samples on
the final interval, while `terminal_condition=:any` ignores the final state.

The filter uses `f[a+1]` for state `a`, the number of reconstructed lineages.
The current implementation is intentionally restricted to constant-rate,
single-lineage initial conditioning (`A_t0^ℓ = 1`) and removal sampling only
(`pars.r ≈ 1`). Serial sampling times must lie strictly before `tℓ`.
`max_count` is currently a guard on the minimum feasible observed count rather
than an independent truncation control; the effective finite state caps are
derived from the remaining observed serial and terminal counts.

With `diagnostics=true`, return a named tuple containing `likelihood`,
`forward_vectors`, `serial_contributions`, `terminal_contribution`,
`effective_max_counts`, `max_count`, `retained_mass`, and `tail_mass`. The
reported `tail_mass` is `nothing` because this exact finite filter does not
estimate an omitted truncation tail.
"""
function sampling_time_likelihood(
    t0::Real,
    sampling_times::AbstractVector{<:Real},
    sample_counts::AbstractVector{<:Integer},
    terminal_count::Integer,
    pars::ConstantRateBDParameters;
    tℓ::Real,
    labelled_samples::Bool=false,
    terminal_sampling::Bool=true,
    terminal_condition::Symbol=:observed,
    atol::Real=1e-12,
    max_count::Union{Nothing,Integer}=nothing,
    diagnostics::Bool=false,
)
    T = promote_type(
        typeof(t0),
        eltype(sampling_times),
        typeof(tℓ),
        typeof(atol),
        typeof(pars.λ),
        Float64,
    )
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    times = T.(sampling_times)
    counts = Int.(sample_counts)
    terminal = Int(terminal_count)
    t0T = T(t0)
    tlT = T(tℓ)

    _validate_sampling_time_likelihood_inputs(t0T, times, counts, terminal, pT, tlT, max_count)
    if !terminal_sampling && terminal != 0
        throw(ArgumentError("terminal_count must be 0 when terminal_sampling=false."))
    end
    function _sampling_time_diagnostics(
        likelihood,
        forward_vectors,
        serial_contributions,
        terminal_contribution,
        effective_max_counts,
        retained_mass,
    )
        return (
            likelihood=likelihood,
            forward_vectors=forward_vectors,
            serial_contributions=serial_contributions,
            terminal_contribution=terminal_contribution,
            effective_max_counts=effective_max_counts,
            max_count=max_count,
            retained_mass=retained_mass,
            tail_mass=nothing,
            tℓ=tlT,
            terminal_sampling=terminal_sampling,
            terminal_condition=terminal_condition,
        )
    end

    if iszero(pT.ψ) && any(!iszero, counts)
        likelihood = zero(T)
        diagnostics || return likelihood
        return _sampling_time_diagnostics(likelihood, Vector{T}[], T[], likelihood, Int[], likelihood)
    end
    if isempty(counts) && terminal_sampling
        likelihood = conditioned_reconstructed_count_pmf(terminal, t0T, tlT, tlT, pT)
        diagnostics || return likelihood
        forward = zeros(T, max(terminal, 1) + 1)
        forward[2] = one(T)
        return _sampling_time_diagnostics(likelihood, [forward], T[], likelihood, Int[], sum(forward))
    end

    M = length(counts)
    remaining = zeros(Int, M + 1)
    remaining[M + 1] = terminal
    for m in M:-1:1
        remaining[m] = remaining[m + 1] + counts[m]
    end
    if max_count !== nothing && max_count < remaining[1]
        throw(ArgumentError("max_count must be at least the total observed sample count."))
    end

    # State a is stored at f[a+1], so index 1 is the zero-lineage state.
    f = zeros(T, max(remaining[1], 1) + 1)
    f[2] = one(T)
    forward_vectors = diagnostics ? [copy(f)] : Vector{T}[]
    serial_contributions = diagnostics ? T[] : T[]
    effective_max_counts = diagnostics ? Int[] : Int[]
    u = t0T

    for m in 1:M
        ti = times[m]
        c = counts[m]
        before_max = max(remaining[m], 1)
        diagnostics && push!(effective_max_counts, before_max)
        g = zeros(T, before_max + 1)
        @inbounds for a in 1:(length(f) - 1)
            fa = f[a + 1]
            iszero(fa) && continue
            for b in a:before_max
                g[b + 1] += fa * _no_sample_reconstructed_kernel(u, ti, tlT, a, b, pT)
            end
        end

        after_max = max(remaining[m + 1], 1)
        next = zeros(T, after_max + 1)
        ψ̃ = transformed_sampling_rate(ti, tlT, pT)
        @inbounds for b in c:before_max
            d = b - c
            d <= after_max || continue
            next[d + 1] += g[b + 1] *
                _grouped_removal_sampling_jump(b, c, ψ̃; labelled_samples=labelled_samples)
        end
        f = next
        if diagnostics
            push!(serial_contributions, sum(f))
            push!(forward_vectors, copy(f))
        end
        u = ti
    end

    if terminal_sampling
        lik = zero(T)
        @inbounds for a in 1:(length(f) - 1)
            fa = f[a + 1]
            iszero(fa) && continue
            lik += fa * terminal_count_transition(u, tlT, a, terminal, pT)
        end
        diagnostics || return lik
        return _sampling_time_diagnostics(lik, forward_vectors, serial_contributions, lik, effective_max_counts, sum(f))
    end

    terminal_condition == :censored && begin
        η = no_sample_probability_conditioned(u, tlT, tlT, pT)
        lik = zero(T)
        @inbounds for a in 0:(length(f) - 1)
            lik += f[a + 1] * η^a
        end
        diagnostics || return lik
        return _sampling_time_diagnostics(lik, forward_vectors, serial_contributions, lik, effective_max_counts, sum(f))
    end
    terminal_condition == :any && begin
        lik = sum(f)
        diagnostics || return lik
        return _sampling_time_diagnostics(lik, forward_vectors, serial_contributions, one(T), effective_max_counts, lik)
    end
    throw(ArgumentError("unsupported terminal_condition=$terminal_condition when terminal_sampling=false; expected :censored or :any."))
end
