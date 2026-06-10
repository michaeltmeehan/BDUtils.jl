#!/usr/bin/env julia

# Manual validation for cached origin-time likelihood profiles and the
# one-dimensional origin-time MLE.
#
# Run from the package root:
#
#   julia --project=. scripts/validation/origin_time_mle_validation.jl
#
# Optional controls:
#
#   BDUTILS_ORIGIN_TIME_MLE_NREPS=10
#   BDUTILS_ORIGIN_TIME_MLE_GRID_LENGTH=201
#   BDUTILS_ORIGIN_TIME_MLE_DEBUG_REPLICATE=3
#   BDUTILS_ORIGIN_TIME_MLE_DEBUG_TARGET=200
#   BDUTILS_ORIGIN_TIME_MLE_DEBUG_SCHEME=first_n
#
# The simulation section uses `simulate_bd` to generate a larger outbreak under
# a known origin time, then compares several ways of selecting n sampled removal
# times from that outbreak. The schemes answer different questions:
#
#   :first_n   early observed samples, matching the original script behaviour;
#   :random_n  a reproducible random subset of all sampled removals;
#   :even_n    approximately evenly spaced order statistics across removals.
#
# This is not exact simulation conditional on the total sample count. It is an
# estimator-behaviour diagnostic for the cached likelihood and one-dimensional
# origin-time optimizer under fixed, known birth-death-sampling parameters.
# The profile-likelihood intervals are asymptotic and conditional on those
# fixed parameters. Nonfinite or boundary cases usually indicate either weak
# information or numerical/truncation issues requiring inspection.

using BDUtils
using Printf
using Random
using Statistics

const PROFILE_CUTOFF_95 = 0.5 * 3.841458820694124
const DELTA_LOGLIK_CUTOFF_95 = -PROFILE_CUTOFF_95

const target_sample_sizes = [10, 50, 100, 200]
const selection_schemes = (:first_n, :random_n, :even_n)
const default_nreps = 5
const nreps = parse(Int, get(ENV, "BDUTILS_ORIGIN_TIME_MLE_NREPS", string(default_nreps)))
const debug_replicate = haskey(ENV, "BDUTILS_ORIGIN_TIME_MLE_DEBUG_REPLICATE") ?
    parse(Int, ENV["BDUTILS_ORIGIN_TIME_MLE_DEBUG_REPLICATE"]) : nothing
const debug_target = haskey(ENV, "BDUTILS_ORIGIN_TIME_MLE_DEBUG_TARGET") ?
    parse(Int, ENV["BDUTILS_ORIGIN_TIME_MLE_DEBUG_TARGET"]) : nothing
const debug_scheme = haskey(ENV, "BDUTILS_ORIGIN_TIME_MLE_DEBUG_SCHEME") ?
    Symbol(ENV["BDUTILS_ORIGIN_TIME_MLE_DEBUG_SCHEME"]) : nothing
const debug_mode = debug_replicate !== nothing || debug_target !== nothing || debug_scheme !== nothing
const seed = 20240609

const sim_pars = ConstantRateBDParameters(2.0, 0.2, 0.7, 1.0, 0.0)
const t0_true = -1.0
const simulation_tmax = 5.0
const max_simulation_attempts = 500
const default_profile_grid_length = 201
const profile_grid_length = parse(Int, get(ENV, "BDUTILS_ORIGIN_TIME_MLE_GRID_LENGTH", string(default_profile_grid_length)))
const grid_improvement_tolerance = 1e-6

if debug_scheme !== nothing && debug_scheme ∉ selection_schemes
    throw(ArgumentError("BDUTILS_ORIGIN_TIME_MLE_DEBUG_SCHEME must be one of $(selection_schemes); got $debug_scheme"))
end

function grouped_sampling_times(times::AbstractVector{<:Real})
    sorted = sort(Float64.(times))
    grouped = Float64[]
    counts = Int[]
    for t in sorted
        if !isempty(grouped) && t == grouped[end]
            counts[end] += 1
        else
            push!(grouped, t)
            push!(counts, 1)
        end
    end
    return grouped, counts
end

function profile_likelihood_interval(profile; delta_cutoff=DELTA_LOGLIK_CUTOFF_95)
    finite = findall(isfinite, profile.delta_loglikelihood)
    isempty(finite) && return (
        ci_lower=NaN,
        ci_upper=NaN,
        ci_width=NaN,
        hits_grid_edge=false,
        accepted_count=0,
    )

    accepted = [i for i in finite if profile.delta_loglikelihood[i] >= delta_cutoff]
    isempty(accepted) && return (
        ci_lower=NaN,
        ci_upper=NaN,
        ci_width=NaN,
        hits_grid_edge=false,
        accepted_count=0,
    )

    ci_lower = minimum(profile.t0[accepted])
    ci_upper = maximum(profile.t0[accepted])
    return (
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_upper - ci_lower,
        hits_grid_edge=first(accepted) == first(finite) || last(accepted) == last(finite),
        accepted_count=length(accepted),
    )
end

function grid_maximum(profile)
    finite = findall(isfinite, profile.loglikelihood)
    isempty(finite) && return (grid_t0=NaN, grid_loglikelihood=-Inf)
    i = finite[argmax(profile.loglikelihood[finite])]
    return (grid_t0=profile.t0[i], grid_loglikelihood=profile.loglikelihood[i])
end

function profile_diagnostics(profile, grid_best)
    finite_values = filter(isfinite, profile.loglikelihood)
    profile_range = isempty(finite_values) ? NaN : maximum(finite_values) - minimum(finite_values)
    finite_grid_fraction = length(finite_values) / length(profile.loglikelihood)
    return (
        profile_range=profile_range,
        finite_grid_fraction=finite_grid_fraction,
        grid_max_t0=grid_best.grid_t0,
        grid_max_loglikelihood=grid_best.grid_loglikelihood,
    )
end

function sample_time_geometry(times::AbstractVector{<:Real}; mle_t0=NaN)
    sorted = sort(Float64.(times))
    isempty(sorted) && return (
        first_time=NaN,
        last_time=NaN,
        sample_span=NaN,
        first_delay=NaN,
        mle_first_gap=NaN,
        median_time=NaN,
        q10_time=NaN,
        q90_time=NaN,
        iqr_time=NaN,
    )

    first_time = first(sorted)
    last_time = last(sorted)
    q25_time = quantile(sorted, 0.25)
    q75_time = quantile(sorted, 0.75)
    return (
        first_time=first_time,
        last_time=last_time,
        sample_span=last_time - first_time,
        first_delay=first_time - t0_true,
        mle_first_gap=isfinite(mle_t0) ? first_time - mle_t0 : NaN,
        median_time=median(sorted),
        q10_time=quantile(sorted, 0.10),
        q90_time=quantile(sorted, 0.90),
        iqr_time=q75_time - q25_time,
    )
end

function finite_correlation(xs, ys; min_n=3)
    pairs = [(Float64(x), Float64(y)) for (x, y) in zip(xs, ys) if isfinite(x) && isfinite(y)]
    n = length(pairs)
    n >= min_n || return (correlation=NaN, n=n)
    xmean = mean(first.(pairs))
    ymean = mean(last.(pairs))
    dx = first.(pairs) .- xmean
    dy = last.(pairs) .- ymean
    denom = sqrt(sum(abs2, dx) * sum(abs2, dy))
    denom > 0 || return (correlation=NaN, n=n)
    return (correlation=sum(dx .* dy) / denom, n=n)
end

function sampled_removal_times(log::BDEventLog)
    return [log.time[i] for i in eachindex(log.time) if log.kind[i] == SerialSampling]
end

function simulate_outbreak_with_samples(rng, pars, tmax, required_n)
    for attempt in 1:max_simulation_attempts
        log = simulate_bd(rng, pars, tmax; apply_ρ₀=false)
        times = sampled_removal_times(log)
        if length(times) >= required_n
            sort!(times)
            return (times=times, events=length(log), attempt=attempt)
        end
    end
    error("failed to simulate at least $required_n sampled removals after $max_simulation_attempts attempts")
end

function select_sample_times(times, target_n, scheme::Symbol, rng)
    target_n <= length(times) ||
        throw(ArgumentError("target_n=$target_n exceeds available sampled removals $(length(times))"))
    if scheme == :first_n
        return collect(@view times[1:target_n])
    elseif scheme == :random_n
        indices = randperm(rng, length(times))[1:target_n]
        return sort(times[indices])
    elseif scheme == :even_n
        if target_n == 1
            return [first(times)]
        end
        nall = length(times)
        indices = [1 + floor(Int, (k - 1) * (nall - 1) / (target_n - 1)) for k in 1:target_n]
        return times[indices]
    end
    throw(ArgumentError("unsupported selection_scheme=$scheme"))
end

function deterministic_validation()
    pars = ConstantRateBDParameters(1.35, 0.3, 0.55, 1.0, 0.4)
    tℓ = 1.4
    sampling_times = [0.35, 0.8]
    sample_counts = [1, 2]
    terminal_count = 1
    lower = -0.6
    upper = 0.3
    t0_grid = collect(range(lower, stop=upper, length=101))

    cache = cache_sampling_time_likelihood(
        sampling_times,
        sample_counts,
        terminal_count,
        pars;
        tℓ=tℓ,
    )
    profile = origin_time_loglikelihood_profile(cache, t0_grid)
    grid = grid_maximum(profile)
    fit = origin_time_mle(cache; lower=lower, upper=upper)

    grid_step = step(range(lower, stop=upper, length=101))
    ok = fit.loglikelihood + 1e-6 >= grid.grid_loglikelihood ||
        abs(fit.t0_hat - grid.grid_t0) <= 2grid_step

    println("deterministic origin-time MLE smoke check")
    @printf(
        "%10s  %10s  %14s  %14s  %18s  %18s  %12s\n",
        "lower",
        "upper",
        "grid_t0",
        "mle_t0",
        "grid_loglik",
        "mle_loglik",
        "status",
    )
    @printf(
        "%10.5f  %10.5f  %14.7f  %14.7f  %18.10f  %18.10f  %12s\n",
        lower,
        upper,
        grid.grid_t0,
        fit.t0_hat,
        grid.grid_loglikelihood,
        fit.loglikelihood,
        string(fit.status),
    )
    println("smoke_check_status = ", ok ? "ok" : "FAIL")
    println()
    ok || error("origin_time_mle did not match or improve on the deterministic grid profile")
    return nothing
end

function fit_selected_times(selected_times)
    grouped, counts = grouped_sampling_times(selected_times)
    geometry = sample_time_geometry(selected_times)
    first_time = geometry.first_time
    last_time = geometry.last_time
    lower = t0_true - 2.0
    upper = prevfloat(first_time)
    grid = collect(range(lower, stop=upper, length=profile_grid_length))

    cache = cache_sampling_time_likelihood(
        grouped,
        counts,
        sim_pars;
        tℓ=last_time,
        terminal_condition=:any,
    )
    profile = origin_time_loglikelihood_profile(cache, grid)
    fit = origin_time_mle(cache; lower=lower, upper=upper, tolerance=1e-8)
    interval = profile_likelihood_interval(profile)
    grid_best = grid_maximum(profile)
    diagnostics = profile_diagnostics(profile, grid_best)
    true_loglik = sampling_time_loglikelihood(cache, t0_true)
    mle_improves_grid = isfinite(fit.loglikelihood) &&
        fit.loglikelihood + grid_improvement_tolerance >= grid_best.grid_loglikelihood
    bound_scale = max(1.0, abs(lower), abs(upper), upper - lower)
    near_bound = abs(fit.t0_hat - lower) <= sqrt(1e-8) * bound_scale ||
        abs(fit.t0_hat - upper) <= sqrt(1e-8) * bound_scale
    return (
        cache=cache,
        fit=fit,
        profile=profile,
        interval=interval,
        grid_best=grid_best,
        diagnostics=diagnostics,
        true_loglik=true_loglik,
        mle_minus_true_loglik=fit.loglikelihood - true_loglik,
        mle_improves_grid=mle_improves_grid,
        lower=lower,
        upper=upper,
        geometry=sample_time_geometry(selected_times; mle_t0=fit.t0_hat),
        effective_count_cap=cache.max_count,
        near_bound=near_bound,
        retry_status=:not_run,
        retry_mle_t0=NaN,
        retry_loglikelihood=NaN,
    )
end

function debug_fit_report(rep, target_n, scheme, selected_times, result)
    debug_mode || return nothing
    cache = result.cache
    finite_profile = count(isfinite, result.profile.loglikelihood)
    scaled_true_lik = BDUtils._sampling_time_scaled_likelihood(cache, t0_true)
    println()
    println("debug origin-time MLE validation case")
    @printf("  replicate=%d target=%d selection_scheme=%s actual_n=%d\n", rep, target_n, string(scheme), length(selected_times))
    @printf("  first_time=%.17g last_time=%.17g min_gap=%.3e median_gap=%.3e\n",
        first(selected_times),
        last(selected_times),
        minimum(diff(selected_times)),
        median(diff(selected_times)),
    )
    @printf("  cache_mode=%s max_count=%d downstream_log_scale=%.10g\n",
        string(cache.mode),
        cache.max_count,
        cache.downstream_log_scale,
    )
    @printf("  downstream_finite=%s downstream_min=%.3e downstream_max=%.3e downstream_zeros=%d\n",
        string(all(isfinite, cache.downstream)),
        minimum(cache.downstream),
        maximum(cache.downstream),
        count(iszero, cache.downstream),
    )
    @printf("  scaled_true_likelihood=%.10g true_loglik=%.10g mle_loglik=%.10g\n",
        scaled_true_lik,
        result.true_loglik,
        result.fit.loglikelihood,
    )
    @printf("  finite_profile_grid=%d/%d grid_max_t0=%.10g grid_max_loglik=%.10g status=%s\n",
        finite_profile,
        length(result.profile.loglikelihood),
        result.grid_best.grid_t0,
        result.grid_best.grid_loglikelihood,
        string(result.fit.status),
    )
    println()
    return nothing
end

function print_fit_diagnostic(rep, target_n, scheme, actual_n, result)
    status = result.fit.status
    if status == :nonfinite_objective || status == :lower_bound || status == :upper_bound || result.near_bound
        println()
        println("diagnostic: origin_time_mle returned status=", status)
        @printf(
            "  target=%d replicate=%d selection_scheme=%s actual_n=%d first_time=%.10g last_time=%.10g\n",
            target_n,
            rep,
            string(scheme),
            actual_n,
            result.geometry.first_time,
            result.geometry.last_time,
        )
        @printf(
            "  bounds=[%.10g, %.10g] finite_grid_fraction=%.4f effective_count_cap=%d\n",
            result.lower,
            result.upper,
            result.diagnostics.finite_grid_fraction,
            result.effective_count_cap,
        )
        @printf(
            "  mle_t0=%.10g mle_loglik=%.10g grid_max_t0=%.10g grid_max_loglik=%.10g true_loglik=%.10g\n",
            result.fit.t0_hat,
            result.fit.loglikelihood,
            result.grid_best.grid_t0,
            result.grid_best.grid_loglikelihood,
            result.true_loglik,
        )
        if status == :nonfinite_objective
            println("  retry: not run; TODO expose an independent grouped-cache count-cap or truncation sensitivity knob.")
        end
        println()
    end
    return nothing
end

function empty_failure_row(target_n, actual_n, replicate, selection_scheme, sim, err, geometry)
    return (
        target_n=target_n,
        selection_scheme=selection_scheme,
        actual_n=actual_n,
        replicate=replicate,
        true_t0=t0_true,
        mle_t0=NaN,
        estimation_error=NaN,
        abs_error=NaN,
        lower_bound=NaN,
        upper_bound=NaN,
        first_time=geometry.first_time,
        last_time=geometry.last_time,
        sample_span=geometry.sample_span,
        first_delay=geometry.first_delay,
        mle_first_gap=geometry.mle_first_gap,
        median_time=geometry.median_time,
        q10_time=geometry.q10_time,
        q90_time=geometry.q90_time,
        iqr_time=geometry.iqr_time,
        ci_lower=NaN,
        ci_upper=NaN,
        ci_width=NaN,
        ci_covers_true=false,
        true_loglik=-Inf,
        mle_loglikelihood=-Inf,
        mle_minus_true_loglik=NaN,
        profile_range=NaN,
        finite_grid_fraction=NaN,
        grid_max_t0=NaN,
        grid_max_loglikelihood=-Inf,
        mle_improves_grid=false,
        optimizer_status=:failed,
        original_status=:failed,
        retry_status=:not_run,
        retry_mle_t0=NaN,
        retry_loglikelihood=NaN,
        effective_count_cap=missing,
        mle_near_bound=false,
        ci_hits_grid_edge=false,
        simulation_attempt=sim.attempt,
        simulation_event_count=sim.events,
        success=false,
        failure_reason=sprint(showerror, err),
    )
end

function simulation_rows()
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    max_target = maximum(target_sample_sizes)
    required_n = debug_target === nothing ? max_target : debug_target
    total_reps = debug_replicate === nothing ? nreps : max(nreps, debug_replicate)

    for rep in 1:total_reps
        sim = simulate_outbreak_with_samples(rng, sim_pars, simulation_tmax, required_n)
        if debug_replicate !== nothing && rep != debug_replicate
            continue
        end
        absolute_sample_times = t0_true .+ sim.times
        @printf(
            "replicate %d: selected from simulated outbreak with %d sampled removals, %d events, attempt %d\n",
            rep,
            length(absolute_sample_times),
            sim.events,
            sim.attempt,
        )

        for target_n in target_sample_sizes
            debug_target === nothing || target_n == debug_target || continue
            for scheme in selection_schemes
                debug_scheme === nothing || scheme == debug_scheme || continue
                selection_rng = MersenneTwister(seed + 100_000 * rep + 1_000 * target_n + findfirst(==(scheme), selection_schemes))
                selected = select_sample_times(absolute_sample_times, target_n, scheme, selection_rng)
                actual_n = length(selected)
                try
                    result = fit_selected_times(selected)
                    fit = result.fit
                    interval = result.interval
                    grid_best = result.grid_best
                    diagnostics = result.diagnostics
                    geometry = result.geometry
                    ci_covers_true = interval.ci_lower <= t0_true <= interval.ci_upper
                    debug_fit_report(rep, target_n, scheme, selected, result)
                    print_fit_diagnostic(rep, target_n, scheme, actual_n, result)
                    push!(rows, (
                        target_n=target_n,
                        selection_scheme=scheme,
                        actual_n=actual_n,
                        replicate=rep,
                        true_t0=t0_true,
                        mle_t0=fit.t0_hat,
                        estimation_error=fit.t0_hat - t0_true,
                        abs_error=abs(fit.t0_hat - t0_true),
                        lower_bound=result.lower,
                        upper_bound=result.upper,
                        first_time=geometry.first_time,
                        last_time=geometry.last_time,
                        sample_span=geometry.sample_span,
                        first_delay=geometry.first_delay,
                        mle_first_gap=geometry.mle_first_gap,
                        median_time=geometry.median_time,
                        q10_time=geometry.q10_time,
                        q90_time=geometry.q90_time,
                        iqr_time=geometry.iqr_time,
                        ci_lower=interval.ci_lower,
                        ci_upper=interval.ci_upper,
                        ci_width=interval.ci_width,
                        ci_covers_true=ci_covers_true,
                        true_loglik=result.true_loglik,
                        mle_loglikelihood=fit.loglikelihood,
                        mle_minus_true_loglik=result.mle_minus_true_loglik,
                        profile_range=diagnostics.profile_range,
                        finite_grid_fraction=diagnostics.finite_grid_fraction,
                        grid_max_t0=grid_best.grid_t0,
                        grid_max_loglikelihood=grid_best.grid_loglikelihood,
                        mle_improves_grid=result.mle_improves_grid,
                        optimizer_status=fit.status,
                        original_status=fit.status,
                        retry_status=result.retry_status,
                        retry_mle_t0=result.retry_mle_t0,
                        retry_loglikelihood=result.retry_loglikelihood,
                        effective_count_cap=result.effective_count_cap,
                        mle_near_bound=result.near_bound,
                        ci_hits_grid_edge=interval.hits_grid_edge,
                        simulation_attempt=sim.attempt,
                        simulation_event_count=sim.events,
                        success=isfinite(fit.loglikelihood),
                        failure_reason="",
                    ))
                catch err
                    geometry = sample_time_geometry(selected)
                    println()
                    println("diagnostic: origin_time_mle fit threw an exception")
                    @printf(
                        "  target=%d replicate=%d selection_scheme=%s actual_n=%d first_time=%.10g last_time=%.10g\n",
                        target_n,
                        rep,
                        string(scheme),
                        actual_n,
                        geometry.first_time,
                        geometry.last_time,
                    )
                    println("  failure_reason=", sprint(showerror, err))
                    println()
                    push!(rows, empty_failure_row(
                        target_n,
                        actual_n,
                        rep,
                        scheme,
                        sim,
                        err,
                        geometry,
                    ))
                end
            end
        end
    end
    return rows
end

function summarize_group(group, target_n, scheme)
    successes = [row for row in group if row.success]
    failure_count = length(group) - length(successes)
    abs_errors = [row.abs_error for row in successes if isfinite(row.abs_error)]
    coverages = [row.ci_covers_true for row in successes]
    widths = [row.ci_width for row in successes if isfinite(row.ci_width)]
    profile_ranges = [row.profile_range for row in successes if isfinite(row.profile_range)]
    finite_fractions = [row.finite_grid_fraction for row in successes if isfinite(row.finite_grid_fraction)]
    boundary_hits = [row.mle_near_bound || row.ci_hits_grid_edge for row in successes]
    first_delays = [row.first_delay for row in successes if isfinite(row.first_delay)]
    sample_spans = [row.sample_span for row in successes if isfinite(row.sample_span)]
    mle_first_gaps = [row.mle_first_gap for row in successes if isfinite(row.mle_first_gap)]
    return (
        target_n=target_n,
        selection_scheme=scheme,
        successful_replicates=length(successes),
        mean_abs_error=isempty(abs_errors) ? NaN : mean(abs_errors),
        median_abs_error=isempty(abs_errors) ? NaN : median(abs_errors),
        empirical_profile_ci_coverage=isempty(coverages) ? NaN : mean(coverages),
        median_ci_width=isempty(widths) ? NaN : median(widths),
        median_first_delay=isempty(first_delays) ? NaN : median(first_delays),
        median_sample_span=isempty(sample_spans) ? NaN : median(sample_spans),
        median_mle_first_gap=isempty(mle_first_gaps) ? NaN : median(mle_first_gaps),
        median_profile_range=isempty(profile_ranges) ? NaN : median(profile_ranges),
        median_finite_grid_fraction=isempty(finite_fractions) ? NaN : median(finite_fractions),
        boundary_hit_rate=isempty(boundary_hits) ? NaN : mean(boundary_hits),
        failure_count=failure_count,
    )
end

function summarize_rows(rows)
    summaries = NamedTuple[]
    for target_n in target_sample_sizes
        for scheme in selection_schemes
            group = [row for row in rows if row.target_n == target_n && row.selection_scheme == scheme]
            push!(summaries, summarize_group(group, target_n, scheme))
        end
    end
    for scheme in selection_schemes
        group = [row for row in rows if row.selection_scheme == scheme]
        push!(summaries, summarize_group(group, "all", scheme))
    end
    return summaries
end

function correlation_rows(rows)
    correlations = NamedTuple[]
    for scheme in selection_schemes
        group = [row for row in rows if row.selection_scheme == scheme && row.success]
        abs_err_first_delay = finite_correlation([row.abs_error for row in group], [row.first_delay for row in group])
        mle_first_time = finite_correlation([row.mle_t0 for row in group], [row.first_time for row in group])
        abs_err_sample_span = finite_correlation([row.abs_error for row in group], [row.sample_span for row in group])
        ci_width_first_delay = finite_correlation([row.ci_width for row in group], [row.first_delay for row in group])
        push!(correlations, (
            target_n="all",
            selection_scheme=scheme,
            cor_abs_err_first_delay=abs_err_first_delay.correlation,
            n_abs_err_first_delay=abs_err_first_delay.n,
            cor_mle_t0_first_time=mle_first_time.correlation,
            n_mle_t0_first_time=mle_first_time.n,
            cor_abs_err_sample_span=abs_err_sample_span.correlation,
            n_abs_err_sample_span=abs_err_sample_span.n,
            cor_ci_width_first_delay=ci_width_first_delay.correlation,
            n_ci_width_first_delay=ci_width_first_delay.n,
        ))
    end
    for target_n in target_sample_sizes
        for scheme in selection_schemes
            group = [row for row in rows if row.target_n == target_n && row.selection_scheme == scheme && row.success]
            length(group) >= 3 || continue
            abs_err_first_delay = finite_correlation([row.abs_error for row in group], [row.first_delay for row in group])
            mle_first_time = finite_correlation([row.mle_t0 for row in group], [row.first_time for row in group])
            abs_err_sample_span = finite_correlation([row.abs_error for row in group], [row.sample_span for row in group])
            ci_width_first_delay = finite_correlation([row.ci_width for row in group], [row.first_delay for row in group])
            push!(correlations, (
                target_n=target_n,
                selection_scheme=scheme,
                cor_abs_err_first_delay=abs_err_first_delay.correlation,
                n_abs_err_first_delay=abs_err_first_delay.n,
                cor_mle_t0_first_time=mle_first_time.correlation,
                n_mle_t0_first_time=mle_first_time.n,
                cor_abs_err_sample_span=abs_err_sample_span.correlation,
                n_abs_err_sample_span=abs_err_sample_span.n,
                cor_ci_width_first_delay=ci_width_first_delay.correlation,
                n_ci_width_first_delay=ci_width_first_delay.n,
            ))
        end
    end
    return correlations
end

function print_replicate_table(rows)
    println()
    println("simulation replicate-level results")
    @printf(
        "%6s  %8s  %3s  %3s  %9s  %9s  %9s  %8s  %8s  %8s  %9s  %7s  %11s  %8s\n",
        "target",
        "scheme",
        "rep",
        "n",
        "true_t0",
        "mle_t0",
        "abs_err",
        "first_d",
        "span",
        "mle_gap",
        "ci_w",
        "cover",
        "status",
        "boundhit",
    )
    for row in rows
        @printf(
            "%6d  %8s  %3d  %3d  %9.4f  %9.4f  %9.4f  %8.4f  %8.4f  %8.4f  %9.4f  %7s  %11s  %8s\n",
            row.target_n,
            string(row.selection_scheme),
            row.replicate,
            row.actual_n,
            row.true_t0,
            row.mle_t0,
            row.abs_error,
            row.first_delay,
            row.sample_span,
            row.mle_first_gap,
            row.ci_width,
            string(row.ci_covers_true),
            string(row.optimizer_status),
            string(row.mle_near_bound),
        )
    end
    println()
    println("Full replicate rows also include sampling-time quantiles, bounds, profile diagnostics, true/grid/MLE log-likelihoods, retry fields, effective count caps, and simulation diagnostics.")
end

function print_summary_table(summaries)
    println()
    println("simulation summary by target sample size and selection scheme, plus all-target rows")
    @printf(
        "%6s  %8s  %9s  %13s  %15s  %10s  %12s  %12s  %11s  %13s  %8s\n",
        "target",
        "scheme",
        "successes",
        "mean_abs_err",
        "median_abs_err",
        "coverage",
        "median_ci_w",
        "med_first_d",
        "med_span",
        "med_mle_gap",
        "failures",
    )
    for row in summaries
        @printf(
            "%6s  %8s  %9d  %13.5f  %15.5f  %10.3f  %12.5f  %12.5f  %11.5f  %13.5f  %8d\n",
            string(row.target_n),
            string(row.selection_scheme),
            row.successful_replicates,
            row.mean_abs_error,
            row.median_abs_error,
            row.empirical_profile_ci_coverage,
            row.median_ci_width,
            row.median_first_delay,
            row.median_sample_span,
            row.median_mle_first_gap,
            row.failure_count,
        )
    end
end

function print_correlation_table(correlations)
    println()
    println("sampling-time geometry correlations")
    @printf(
        "%6s  %8s  %10s  %3s  %10s  %3s  %10s  %3s  %10s  %3s\n",
        "target",
        "scheme",
        "err~first",
        "n",
        "mle~first",
        "n",
        "err~span",
        "n",
        "ciw~first",
        "n",
    )
    for row in correlations
        @printf(
            "%6s  %8s  %10.4f  %3d  %10.4f  %3d  %10.4f  %3d  %10.4f  %3d\n",
            string(row.target_n),
            string(row.selection_scheme),
            row.cor_abs_err_first_delay,
            row.n_abs_err_first_delay,
            row.cor_mle_t0_first_time,
            row.n_mle_t0_first_time,
            row.cor_abs_err_sample_span,
            row.n_abs_err_sample_span,
            row.cor_ci_width_first_delay,
            row.n_ci_width_first_delay,
        )
    end
end

function csv_escape(x)
    s = string(x)
    if occursin('"', s) || occursin(',', s) || occursin('\n', s) || occursin('\r', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_csv(path, rows)
    isempty(rows) && return nothing
    mkpath(dirname(path))
    names = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(string.(names), ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, name)) for name in names), ","))
        end
    end
    return nothing
end

function main()
    deterministic_validation()

    println("simulation-based origin-time MLE validation")
    println("parameters = ", sim_pars)
    println("true_t0 = ", t0_true)
    println("target_sample_sizes = ", target_sample_sizes)
    println("selection_schemes = ", selection_schemes)
    println("replicates = ", nreps)
    println("seed = ", seed)
    println("profile_grid_length = ", profile_grid_length)
    println("debug_mode = ", debug_mode)
    if debug_mode
        println("debug_replicate = ", debug_replicate)
        println("debug_target = ", debug_target)
        println("debug_scheme = ", debug_scheme)
    end
    println("profile_likelihood_delta_cutoff_95 = ", DELTA_LOGLIK_CUTOFF_95)
    println()
    println("Sampling convention: each replicate simulates a larger outbreak with simulate_bd,")
    println("then evaluates first_n, random_n, and even_n subsets for each target n.")
    println("These schemes probe early-sample behaviour, random subsampling behaviour,")
    println("and approximately quantile-spaced timing information, respectively.")
    println("This is an estimator-behaviour diagnostic, not exact simulation conditional on n.")
    println("The 95% profile interval uses loglikelihood >= maximum(loglikelihood) - ", PROFILE_CUTOFF_95, ".")
    println("Profile intervals are asymptotic and conditional on fixed parameters.")
    println("Nonfinite or boundary cases are reported for inspection rather than failing the script.")
    println()

    rows = simulation_rows()
    summaries = summarize_rows(rows)
    correlations = correlation_rows(rows)
    print_replicate_table(rows)
    print_summary_table(summaries)
    print_correlation_table(correlations)

    output_dir = joinpath(@__DIR__, "output")
    replicate_path = joinpath(output_dir, "origin_time_mle_simulation_replicates.csv")
    summary_path = joinpath(output_dir, "origin_time_mle_simulation_summary.csv")
    correlation_path = joinpath(output_dir, "origin_time_mle_simulation_correlations.csv")
    write_csv(replicate_path, rows)
    write_csv(summary_path, summaries)
    write_csv(correlation_path, correlations)
    println()
    println("wrote replicate CSV: ", replicate_path)
    println("wrote summary CSV: ", summary_path)
    println("wrote correlation CSV: ", correlation_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
