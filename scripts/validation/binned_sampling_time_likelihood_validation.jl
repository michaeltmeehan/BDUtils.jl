using Random
using Printf
using BDUtils

# Manual validation for interval-integrated, binned sampling-time likelihoods.
#
# This script compares a narrow validation-only PGF/filter implementation with
# direct Monte Carlo from simulate_bd. It conditions both calculations on one
# reconstructed lineage at the first bin edge, `A(first_edge, t_ell) = 1`, and
# estimates Monte Carlo probabilities as hits among only those conditioned
# simulations. It uses rho0 = 0 so every observed sample is a serial sample
# inside a bin, not a terminal atom. Terminal sampling is deliberately left out
# here because a final bin can ambiguously contain both serial samples inside
# the interval and Bernoulli terminal samples exactly at t_ell.
#
# Bin counts are interval-censored serial sampling observations. A bin
# [edges[i], edges[i+1]] is interpreted as the half-open-in-reverse interval
# `(edges[i], edges[i+1]]`, matching the package's no-sampling interval
# convention. For partial-removal sampling (`0 < r < 1`), simulate_bd records
# sampled-and-removed lineages as `SerialSampling` and sampled-but-not-removed
# lineages as `FossilizedSampling`; both are serial sampling events for the
# interval-count likelihood and both are counted in bins. By default bins cover
# the complete observation horizon, so there is no final interval to constrain.
# If `t_ell` extends beyond the last bin, `final_interval=:marginalized` treats
# later samples as unobserved; use `final_interval=:censored` to require no
# later samples.
#
# Increase the simulation budget with, for example:
#   BDUTILS_BINNED_VALIDATION_NSIMS=100000 julia --project=. scripts/validation/binned_sampling_time_likelihood_validation.jl

const DEFAULT_NSIMS = 50_000
const NSIMS = parse(Int, get(ENV, "BDUTILS_BINNED_VALIDATION_NSIMS", string(DEFAULT_NSIMS)))
const SEED = parse(Int, get(ENV, "BDUTILS_BINNED_VALIDATION_SEED", "20260608"))
const ATOL = parse(Float64, get(ENV, "BDUTILS_BINNED_VALIDATION_ATOL", "0.02"))
const MAX_A = parse(Int, get(ENV, "BDUTILS_BINNED_VALIDATION_MAX_A", "120"))

function validate_binned_inputs(bin_edges, counts, pars, tℓ, max_a)
    length(counts) == length(bin_edges) - 1 ||
        throw(ArgumentError("length(counts) must equal length(bin_edges) - 1."))
    length(bin_edges) >= 2 || throw(ArgumentError("bin_edges must contain at least two entries."))
    all(isfinite, bin_edges) || throw(ArgumentError("bin_edges must be finite."))
    for i in 2:length(bin_edges)
        bin_edges[i - 1] < bin_edges[i] ||
            throw(ArgumentError("bin_edges must be strictly increasing."))
    end
    for c in counts
        c isa Integer || throw(ArgumentError("all counts must be non-negative integers."))
        c >= 0 || throw(ArgumentError("all counts must be non-negative integers."))
    end
    first(bin_edges) < tℓ || throw(ArgumentError("first(bin_edges) must be less than tℓ."))
    last(bin_edges) <= tℓ || throw(ArgumentError("last(bin_edges) must be <= tℓ."))
    max_a >= 1 || throw(ArgumentError("max_a must be positive."))
    iszero(pars.ρ₀) ||
        throw(ArgumentError("binned validation keeps terminal sampling disabled; pars.ρ₀ must be 0."))
    return nothing
end

function convolve_joint_by_lineage(single::AbstractMatrix{T}, ai::Int, m::Int, max_a::Int) where {T<:AbstractFloat}
    ai == 0 && begin
        out = zeros(T, max_a + 1, m + 1)
        out[1, 1] = one(T)
        return out
    end

    dist = zeros(T, max_a + 1, m + 1)
    dist[1, 1] = one(T)
    for _ in 1:ai
        next = zeros(T, max_a + 1, m + 1)
        @inbounds for a0 in 0:max_a
            for s0 in 0:m
                mass = dist[a0 + 1, s0 + 1]
                iszero(mass) && continue
                for da in 0:(max_a - a0)
                    for ds in 0:(m - s0)
                        next[a0 + da + 1, s0 + ds + 1] += mass * single[da + 1, ds + 1]
                    end
                end
            end
        end
        dist = next
    end
    return dist
end

function binned_transition_column(ai::Int, m::Int, ti, tj, tℓ, pars, max_a::Int; diagnostics=false)
    ai == 0 && begin
        column = [aj == 0 && m == 0 ? 1.0 : 0.0 for aj in 0:max_a]
        diagnostics || return column
        return (column=column, count_tail_mass=0.0)
    end
    single_result = conditioned_reconstructed_joint_pmf_table(max_a, m, ti, tj, tℓ, pars; diagnostics=diagnostics)
    single = diagnostics ? single_result.table : single_result
    powered = convolve_joint_by_lineage(single, ai, m, max_a)
    column = powered[:, m + 1]
    diagnostics || return column
    return (column=column, count_tail_mass=single_result.count_tail_mass)
end

function binned_sampling_time_likelihood(
    bin_edges,
    counts,
    pars;
    tℓ=last(bin_edges),
    conditioned=true,
    final_interval=:marginalized,
    atol=1e-12,
    max_amax=MAX_A,
    diagnostics=false,
)
    conditioned || throw(ArgumentError("only conditioned=true is implemented in this validation utility."))
    final_interval in (:marginalized, :censored) ||
        throw(ArgumentError("unsupported final_interval=$final_interval; expected :marginalized or :censored."))
    validate_binned_inputs(bin_edges, counts, pars, tℓ, max_amax)

    T = promote_type(eltype(float.(bin_edges)), typeof(tℓ), typeof(atol), typeof(pars.λ), Float64)
    edges = T.(bin_edges)
    tl = T(tℓ)
    p = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))

    f = zeros(T, max_amax + 1)
    f[2] = one(T)
    forward_mass = T[sum(f)]
    count_tail_mass = T[]
    max_count_tail_mass = zero(T)

    for i in eachindex(counts)
        m = Int(counts[i])
        next = zeros(T, max_amax + 1)
        for ai in 0:max_amax
            fa = f[ai + 1]
            iszero(fa) && continue
            q_result = binned_transition_column(ai, m, edges[i], edges[i + 1], tl, p, max_amax; diagnostics=diagnostics)
            q = diagnostics ? q_result.column : q_result
            if diagnostics
                tail = T(q_result.count_tail_mass)
                push!(count_tail_mass, tail)
                max_count_tail_mass = max(max_count_tail_mass, tail)
            end
            @inbounds for aj in 0:max_amax
                next[aj + 1] += fa * q[aj + 1]
            end
        end
        f = next
        push!(forward_mass, sum(f))
    end

    if final_interval == :censored && last(edges) < tl
        η = no_sample_probability_conditioned(last(edges), tl, tl, p)
        likelihood = sum(f[a + 1] * η^a for a in 0:max_amax)
    else
        likelihood = sum(f)
    end
    diagnostics || return likelihood
    return (
        likelihood=likelihood,
        forward_mass=forward_mass,
        final_distribution=f,
        max_a=max_amax,
        max_count_tail_mass=max_count_tail_mass,
        count_tail_mass=count_tail_mass,
        atol=atol,
        tℓ=tl,
        final_interval=final_interval,
    )
end

function shifted_edges(edges)
    offset = first(edges)
    # The integer-like case is translated so simulation starts at time zero.
    # Only coordinates move: duration, rates, horizon, bin counts, and the
    # absence/presence of a final interval are unchanged.
    return (shifted=[x - offset for x in edges], offset=offset)
end

function bin_index(t, edges)
    for b in 1:(length(edges) - 1)
        edges[b] < t <= edges[b + 1] && return b
    end
    return nothing
end

function bin_counts_from_log(log, edges)
    counts = zeros(Int, length(edges) - 1)
    for i in eachindex(log.time)
        # Under partial removal, non-removing serial samples remain active in
        # the simulator and are logged as FossilizedSampling. The PGF's w marker
        # counts all serial sampling events, so both simulator event kinds enter
        # the binned count observation.
        log.kind[i] in (SerialSampling, FossilizedSampling) || continue
        b = bin_index(log.time[i], edges)
        b === nothing || (counts[b] += 1)
    end
    return counts
end

function monte_carlo_binned(case; nsims=NSIMS)
    shifted = shifted_edges(case.bin_edges).shifted
    tl = last(shifted)
    rng = MersenneTwister(case.seed)
    conditioned = 0
    hits = 0

    for _ in 1:nsims
        log = simulate_bd(rng, case.pars, tl; apply_ρ₀=false)
        A_at(log, 0.0, tl) == 1 || continue
        conditioned += 1
        if bin_counts_from_log(log, shifted) == case.target_counts
            hits += 1
        end
    end

    conditioned > 0 || throw(ArgumentError("no simulations satisfied A(first_edge, t_ell) = 1; increase nsims."))
    estimate = hits / conditioned
    se = sqrt(max(estimate * (1 - estimate), 0.0) / conditioned)
    return (
        nsims=nsims,
        conditioned=conditioned,
        hits=hits,
        estimate=estimate,
        se=se,
        ci_low=estimate - 1.96se,
        ci_high=estimate + 1.96se,
    )
end

function analytical_for_case(case; max_a=MAX_A, diagnostics=false)
    shifted = shifted_edges(case.bin_edges).shifted
    return binned_sampling_time_likelihood(
        shifted,
        case.target_counts,
        case.pars;
        tℓ=last(shifted),
        max_amax=max_a,
        atol=ATOL,
        diagnostics=diagnostics,
    )
end

function deterministic_cross_checks(pars)
    checks = Tuple{String,String,Bool,Float64,Float64}[]
    rlabel = @sprintf("r=%.1f", pars.r)

    merged = binned_sampling_time_likelihood([0.0, 1.0], [2], pars; max_amax=MAX_A)
    allocated = sum(
        binned_sampling_time_likelihood([0.0, 0.5, 1.0], [k, 2 - k], pars; max_amax=MAX_A)
        for k in 0:2
    )
    push!(checks, (rlabel, "merge allocation", isapprox(merged, allocated; atol=2ATOL, rtol=5e-3), merged, allocated))

    no_sample_transition = binned_sampling_time_likelihood([0.0, 0.4], [0], pars; tℓ=1.0, max_amax=MAX_A)
    no_sample_kernel = no_sample_probability_conditioned(0.0, 0.4, 1.0, pars)
    push!(checks, (rlabel, "zero-sample bin", isapprox(no_sample_transition, no_sample_kernel; atol=1e-8, rtol=1e-8),
                  no_sample_transition, no_sample_kernel))

    diagnostic = binned_sampling_time_likelihood([0.0, 0.5, 1.0], [1, 0], pars; max_amax=MAX_A, diagnostics=true)
    finite_nonnegative = isfinite(diagnostic.likelihood) && diagnostic.likelihood >= 0.0
    push!(checks, (rlabel, "finite nonnegative", finite_nonnegative, diagnostic.likelihood, 0.0))
    tail_ok = diagnostic.max_count_tail_mass <= ATOL
    push!(checks, (rlabel, "truncation tail", tail_ok, diagnostic.max_count_tail_mass, ATOL))

    if isapprox(pars.r, one(pars.r))
        narrow_edges = [0.0, 0.49, 0.51, 1.0]
        narrow_binned = binned_sampling_time_likelihood(narrow_edges, [0, 1, 0], pars; max_amax=MAX_A)
        exact_density = sampling_time_likelihood(0.0, [0.5], [1], 0, pars;
            tℓ=1.0,
            terminal_sampling=false,
            terminal_condition=:censored,
        )
        narrow_approx = exact_density * (narrow_edges[3] - narrow_edges[2])
        push!(checks, (rlabel, "narrow-bin diagnostic", isapprox(narrow_binned, narrow_approx; atol=0.02, rtol=0.25),
                      narrow_binned, narrow_approx))
    else
        # The exact grouped-time density helper is still removal-only, so for
        # partial removal this remains a finite/positive binned diagnostic.
        narrow_edges = [0.0, 0.49, 0.51, 1.0]
        narrow_binned = binned_sampling_time_likelihood(narrow_edges, [0, 1, 0], pars; max_amax=MAX_A)
        push!(checks, (rlabel, "narrow-bin diagnostic", isfinite(narrow_binned) && narrow_binned >= 0.0,
                      narrow_binned, 0.0))
    end

    return checks
end

function validation_cases()
    removal_pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 0.0)
    partial_pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 0.5, 0.0)
    return [
        (name="one non-empty bin", bin_edges=[0.0, 0.5, 1.0], target_counts=[1, 0],
         pars=removal_pars, seed=SEED + 1),
        (name="two non-empty bins", bin_edges=[0.0, 0.5, 1.0, 1.5], target_counts=[1, 0, 1],
         pars=removal_pars, seed=SEED + 2),
        (name="multiple samples in one bin", bin_edges=[0.0, 0.5, 1.0, 1.5], target_counts=[0, 2, 0],
         pars=removal_pars, seed=SEED + 3),
        (name="empty interior bin", bin_edges=[0.0, 0.4, 0.8, 1.2], target_counts=[1, 0, 1],
         pars=removal_pars, seed=SEED + 4),
        (name="integer-like bins", bin_edges=[-0.5, 0.5, 1.5, 2.5], target_counts=[1, 1, 0],
         pars=removal_pars, seed=SEED + 5),
        (name="partial removal: one non-empty bin", bin_edges=[0.0, 0.5, 1.0], target_counts=[1, 0],
         pars=partial_pars, seed=SEED + 6),
        (name="partial removal: multiple in one bin", bin_edges=[0.0, 0.5, 1.0, 1.5], target_counts=[0, 2, 0],
         pars=partial_pars, seed=SEED + 7),
        (name="partial removal: interior empty bin", bin_edges=[0.0, 0.4, 0.8, 1.2], target_counts=[1, 0, 1],
         pars=partial_pars, seed=SEED + 8),
    ]
end

function fmt(x; digits=6)
    return @sprintf("%.*g", digits, x)
end

function run_case(case)
    analytical = analytical_for_case(case; diagnostics=true)
    likelihood = analytical.likelihood
    mc = monte_carlo_binned(case)
    inside_ci = mc.ci_low <= likelihood <= mc.ci_high
    close_abs = abs(likelihood - mc.estimate) <= ATOL
    status = inside_ci || close_abs ? "ok" : "check"
    return merge(case, (likelihood=likelihood, analytical=analytical, mc=mc, status=status))
end

function print_results(results, checks)
    println("binned_sampling_time_likelihood validation")
    println("------------------------------------------")
    @printf("nsims=%d  seed=%d  atol=%.4g  max_a=%d\n", NSIMS, SEED, ATOL, MAX_A)
    println("Monte Carlo estimates are conditional on A(first_edge, t_ell) = 1; rho0 = 0.")
    println("For r < 1, simulator SerialSampling and FossilizedSampling events are both counted as serial samples.")
    println("Simulation and PGF bins use `(left, right]` endpoints; later samples are marginalized when last_edge < t_ell unless final_interval=:censored is requested.")
    println()
    println("case                                      likelihood    mc_est      mc_se       95% CI                  matches/conditioned  count_tail  status")
    println("----------------------------------------  ------------  ----------  ----------  ----------------------  -------------------  ----------  ------")
    for row in results
        mc = row.mc
        ci = @sprintf("[%.5g, %.5g]", mc.ci_low, mc.ci_high)
        matches = @sprintf("%d/%d", mc.hits, mc.conditioned)
        @printf(
            "%-40s  %12s  %10s  %10s  %-22s  %-19s  %10s  %s\n",
            row.name,
            fmt(row.likelihood),
            fmt(mc.estimate),
            fmt(mc.se),
            ci,
            matches,
            fmt(row.analytical.max_count_tail_mass),
            row.status,
        )
    end

    println()
    println("deterministic cross-checks")
    println("pars   check                  lhs          rhs          status")
    println("-----  ---------------------  -----------  -----------  ------")
    for (rlabel, name, ok, lhs, rhs) in checks
        @printf("%-5s  %-21s  %11s  %11s  %s\n", rlabel, name, fmt(lhs), fmt(rhs), ok ? "ok" : "check")
    end
end

cases = validation_cases()
results = run_case.(cases)
checks = vcat(
    deterministic_cross_checks(ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 0.0)),
    deterministic_cross_checks(ConstantRateBDParameters(1.2, 0.4, 0.5, 0.5, 0.0)),
)
print_results(results, checks)

if any(row.status != "ok" for row in results) || any(!check[3] for check in checks)
    exit(1)
end
