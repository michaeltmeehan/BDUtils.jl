using Random
using Printf
using BDUtils

# Manual Monte Carlo validation for the conditioned reconstructed joint PMF.
#
# This script is intentionally runnable by hand and is not part of the default
# package test suite. It conditions simulated histories on
# `A_at(log, ti, tk) == 1` before comparing empirical frequencies with
# `conditioned_reconstructed_joint_pmf(a, s, ti, tj, tk, pars)`.
#
# The runner defaults to `apply_ρ₀ = false` because these analytical
# probabilities describe sampling over the simulated interval, not an extra
# terminal present-day sampling pass at the horizon.

# ------------------------------------------------------------
# Empirical conditioned reconstructed joint counts
# ------------------------------------------------------------

function empirical_conditioned_reconstructed_joint_counts(logs, ti, tj, tk)
    counts = Dict{Tuple{Int,Int},Int}()

    n_total = length(logs)
    n_retained = 0
    n_conditioned_out = 0

    for log in logs
        ai = A_at(log, ti, tk)

        if ai == 1
            aj = A_at(log, tj, tk)
            sij = S_at(log, tj) - S_at(log, ti)
            key = (aj, sij)
            counts[key] = get(counts, key, 0) + 1
            n_retained += 1
        else
            n_conditioned_out += 1
        end
    end

    return (
        counts = counts,
        n_total = n_total,
        n_retained = n_retained,
        n_conditioned_out = n_conditioned_out,
    )
end

# ------------------------------------------------------------
# Compare empirical and analytical conditional probabilities
# ------------------------------------------------------------

function compare_conditioned_reconstructed_joint_pmf(
    logs,
    ti,
    tj,
    tk,
    pars;
    amax = nothing,
    smax = nothing,
)
    empirical = empirical_conditioned_reconstructed_joint_counts(logs, ti, tj, tk)

    if empirical.n_retained == 0
        return (
            n_total = empirical.n_total,
            n_retained = empirical.n_retained,
            n_conditioned_out = empirical.n_conditioned_out,
            summaries = NamedTuple[],
        )
    end

    observed_pairs = collect(keys(empirical.counts))
    if isnothing(amax)
        amax = maximum(first.(observed_pairs))
    end
    if isnothing(smax)
        smax = maximum(last.(observed_pairs))
    end

    diagnostic = conditioned_reconstructed_joint_pmf_table(
        amax, smax, ti, tj, tk, pars;
        diagnostics = true,
    )

    summaries = NamedTuple[]
    total_abs_error = 0.0
    max_abs_error = 0.0
    empirical_retained_mass = 0.0

    for a in 0:amax
        for s in 0:smax
            count = get(empirical.counts, (a, s), 0)
            emp = count / empirical.n_retained
            ana = diagnostic.table[a + 1, s + 1]
            err = abs(emp - ana)

            empirical_retained_mass += emp
            total_abs_error += err
            max_abs_error = max(max_abs_error, err)

            if count > 0 || ana > 0
                push!(summaries, (
                    a = a,
                    s = s,
                    count = count,
                    empirical = emp,
                    analytical = ana,
                    abs_error = err,
                ))
            end
        end
    end

    return (
        n_total = empirical.n_total,
        n_retained = empirical.n_retained,
        n_conditioned_out = empirical.n_conditioned_out,
        retained_fraction = empirical.n_retained / empirical.n_total,
        empirical_counts = empirical.counts,
        diagnostic = diagnostic,
        amax = amax,
        smax = smax,
        empirical_retained_mass = empirical_retained_mass,
        analytical_retained_mass = diagnostic.retained_mass,
        analytical_missing_mass = diagnostic.missing_mass,
        analytical_count_tail_mass = diagnostic.count_tail_mass,
        analytical_sampling_tail_mass = diagnostic.sampling_tail_mass,
        analytical_count_only_tail_mass = diagnostic.count_only_tail_mass,
        analytical_sampling_only_tail_mass = diagnostic.sampling_only_tail_mass,
        analytical_joint_tail_overlap_mass = diagnostic.joint_tail_overlap_mass,
        total_variation_retained = 0.5 * total_abs_error,
        max_abs_error = max_abs_error,
        summaries = summaries,
    )
end

# ------------------------------------------------------------
# Convenience runners
# ------------------------------------------------------------

function run_conditioned_reconstructed_joint_validation(;
    seed = 1234,
    nrep = 100_000,
    pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 0.8),
    ti = 0.0,
    tj = 1.0,
    tk = 2.0,
    initial_lineages = 1,
    apply_ρ₀ = false,
    amax = nothing,
    smax = nothing,
)
    rng = MersenneTwister(seed)

    logs = [
        simulate_bd(rng, pars, tk;
            initial_lineages = initial_lineages,
            apply_ρ₀ = apply_ρ₀,
        )
        for _ in 1:nrep
    ]

    result = compare_conditioned_reconstructed_joint_pmf(
        logs, ti, tj, tk, pars;
        amax = amax,
        smax = smax,
    )

    println()
    println("Conditioned reconstructed joint validation")
    println("------------------------------------------")
    @printf("nrep                  = %d\n", result.n_total)
    @printf("retained after A_i=1  = %d\n", result.n_retained)
    @printf("conditioned out       = %d\n", result.n_conditioned_out)
    @printf("retained fraction     = %.8f\n", result.retained_fraction)
    @printf("parameters            = %s\n", string(pars))
    @printf("t_i, t_j, t_k          = %.3f, %.3f, %.3f\n", ti, tj, tk)
    @printf("initial_lineages      = %d\n", initial_lineages)
    @printf("apply_rho0            = %s\n", string(apply_ρ₀))
    @printf("amax, smax            = %d, %d\n", result.amax, result.smax)
    @printf("empirical retained    = %.8f\n", result.empirical_retained_mass)
    @printf("analytical retained   = %.8f\n", result.analytical_retained_mass)
    @printf("analytical missing    = %.8g\n", result.analytical_missing_mass)
    @printf("analytical count tail = %.8g\n", result.analytical_count_tail_mass)
    @printf("analytical samp tail  = %.8g\n", result.analytical_sampling_tail_mass)
    @printf("count-only tail       = %.8g\n", result.analytical_count_only_tail_mass)
    @printf("samp-only tail        = %.8g\n", result.analytical_sampling_only_tail_mass)
    @printf("joint tail overlap    = %.8g\n", result.analytical_joint_tail_overlap_mass)
    @printf("TV retained           = %.6g\n", result.total_variation_retained)
    @printf("max abs error         = %.6g\n", result.max_abs_error)

    result.n_retained < 1_000 &&
        println("WARNING: few retained simulations; increase nrep for a sharper Monte Carlo check.")

    println()
    println("Largest absolute errors")
    println("a\ts\tempirical\tanalytical\tabs_error")
    for row in Iterators.take(sort(result.summaries; by = row -> row.abs_error, rev = true), 30)
        @printf("%d\t%d\t%.8g\t%.8g\t%.8g\n", row.a, row.s, row.empirical, row.analytical, row.abs_error)
    end

    return result
end

function run_conditioned_reconstructed_joint_regimes(; seed = 1234, nrep = 100_000)
    regimes = (
        (name = "moderate sampling, complete removal",
         pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 0.8)),
        (name = "moderate sampling, incomplete removal",
         pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 0.4, 0.8)),
        (name = "lower sampling",
         pars = ConstantRateBDParameters(1.4, 0.3, 0.25, 0.7, 0.8)),
    )

    results = NamedTuple[]
    for (i, regime) in enumerate(regimes)
        println()
        println("Regime: ", regime.name)
        push!(results, run_conditioned_reconstructed_joint_validation(
            seed = seed + i - 1,
            nrep = nrep,
            pars = regime.pars,
        ))
    end
    return results
end

# result = run_conditioned_reconstructed_joint_validation()
# results = run_conditioned_reconstructed_joint_regimes()
