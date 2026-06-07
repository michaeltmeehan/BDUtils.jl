function _falling_factorial(b::Integer, c::Integer)
    _check_count("b", b)
    _check_count("c", c)
    c > b && return 0
    out = 1
    for k in 0:(c - 1)
        out *= b - k
    end
    return out
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
    return binomial(b - 1, a - 1) * β^a * γ^(b - a)
end

function _grouped_removal_sampling_jump(
    b::Integer,
    c::Integer,
    ψ̃::Real;
    labelled_samples::Bool=false,
)
    _check_count("b", b)
    _check_count("c", c)
    c > b && return zero(promote_type(typeof(ψ̃), Float64))
    coefficient = labelled_samples ? _falling_factorial(b, c) : binomial(b, c)
    return coefficient * ψ̃^c
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
