using Random
using Printf
using BDUtils

# Manual Monte Carlo validation for the conditioned reconstructed count PMF.
#
# This script is intentionally runnable by hand and is not part of the default
# package test suite. It conditions simulated histories on
# `A_at(log, ti, tk) == 1` before comparing empirical frequencies with
# `conditioned_reconstructed_count_pmf(a, ti, tj, tk, pars)`.
#
# The runner defaults to `apply_ρ₀ = false` because these analytical
# probabilities describe sampling over the simulated interval, not an extra
# terminal present-day sampling pass at the horizon.

# ------------------------------------------------------------
# Empirical conditioned reconstructed count counts
# ------------------------------------------------------------

function empirical_conditioned_reconstructed_count_counts(logs, ti, tj, tk)
    counts = Dict{Int,Int}()

    n_total = length(logs)
    n_retained = 0
    n_conditioned_out = 0

    for log in logs
        ai = A_at(log, ti, tk)

        if ai == 1
            aj = A_at(log, tj, tk)
            counts[aj] = get(counts, aj, 0) + 1
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

function compare_conditioned_reconstructed_count_pmf(
    logs,
    ti,
    tj,
    tk,
    pars;
    amax = nothing,
)
    empirical = empirical_conditioned_reconstructed_count_counts(logs, ti, tj, tk)

    if empirical.n_retained == 0
        return (
            n_total = empirical.n_total,
            n_retained = empirical.n_retained,
            n_conditioned_out = empirical.n_conditioned_out,
            summaries = NamedTuple[],
        )
    end

    observed_as = collect(keys(empirical.counts))
    if isnothing(amax)
        amax = maximum(observed_as)
    end

    summaries = NamedTuple[]
    total_abs_error = 0.0
    max_abs_error = 0.0
    empirical_retained_mass = 0.0
    analytical_retained_mass = 0.0

    for a in 0:amax
        count = get(empirical.counts, a, 0)
        emp = count / empirical.n_retained
        ana = conditioned_reconstructed_count_pmf(a, ti, tj, tk, pars)
        err = abs(emp - ana)

        empirical_retained_mass += emp
        analytical_retained_mass += ana
        total_abs_error += err
        max_abs_error = max(max_abs_error, err)

        if count > 0 || ana > 0
            push!(summaries, (
                a = a,
                count = count,
                empirical = emp,
                analytical = ana,
                abs_error = err,
            ))
        end
    end

    analytical_tail = conditioned_reconstructed_count_tail(amax, ti, tj, tk, pars)

    return (
        n_total = empirical.n_total,
        n_retained = empirical.n_retained,
        n_conditioned_out = empirical.n_conditioned_out,
        retained_fraction = empirical.n_retained / empirical.n_total,
        empirical_counts = empirical.counts,
        amax = amax,
        empirical_retained_mass = empirical_retained_mass,
        analytical_retained_mass = analytical_retained_mass,
        analytical_tail_beyond_amax = analytical_tail,
        total_variation_retained = 0.5 * total_abs_error,
        max_abs_error = max_abs_error,
        summaries = summaries,
    )
end

# ------------------------------------------------------------
# Convenience runners
# ------------------------------------------------------------

function run_conditioned_reconstructed_count_validation(;
    seed = 1234,
    nrep = 100_000,
    pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 0.8),
    ti = 0.0,
    tj = 1.0,
    tk = 2.0,
    initial_lineages = 1,
    apply_ρ₀ = false,
    amax = nothing,
)
    rng = MersenneTwister(seed)

    logs = [
        simulate_bd(rng, pars, tk;
            initial_lineages = initial_lineages,
            apply_ρ₀ = apply_ρ₀,
        )
        for _ in 1:nrep
    ]

    result = compare_conditioned_reconstructed_count_pmf(
        logs, ti, tj, tk, pars;
        amax = amax,
    )

    println()
    println("Conditioned reconstructed count validation")
    println("------------------------------------------")
    @printf("nrep                  = %d\n", result.n_total)
    @printf("retained after A_i=1  = %d\n", result.n_retained)
    @printf("conditioned out       = %d\n", result.n_conditioned_out)
    @printf("retained fraction     = %.8f\n", result.retained_fraction)
    @printf("parameters            = %s\n", string(pars))
    @printf("t_i, t_j, t_k          = %.3f, %.3f, %.3f\n", ti, tj, tk)
    @printf("initial_lineages      = %d\n", initial_lineages)
    @printf("apply_rho0            = %s\n", string(apply_ρ₀))
    @printf("amax                  = %d\n", result.amax)
    @printf("empirical retained    = %.8f\n", result.empirical_retained_mass)
    @printf("analytical retained   = %.8f\n", result.analytical_retained_mass)
    @printf("analytical tail       = %.8g\n", result.analytical_tail_beyond_amax)
    @printf("TV retained           = %.6g\n", result.total_variation_retained)
    @printf("max abs error         = %.6g\n", result.max_abs_error)

    result.n_retained < 1_000 &&
        println("WARNING: few retained simulations; increase nrep for a sharper Monte Carlo check.")

    println()
    println("a\tempirical\tanalytical\tabs_error")
    for row in result.summaries
        @printf("%d\t%.8g\t%.8g\t%.8g\n", row.a, row.empirical, row.analytical, row.abs_error)
    end

    println()
    println("Largest absolute errors")
    println("a\tempirical\tanalytical\tabs_error")
    for row in Iterators.take(sort(result.summaries; by = row -> row.abs_error, rev = true), 20)
        @printf("%d\t%.8g\t%.8g\t%.8g\n", row.a, row.empirical, row.analytical, row.abs_error)
    end

    return result
end

function run_conditioned_reconstructed_count_regimes(; seed = 1234, nrep = 100_000)
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
        push!(results, run_conditioned_reconstructed_count_validation(
            seed = seed + i - 1,
            nrep = nrep,
            pars = regime.pars,
        ))
    end
    return results
end

# result = run_conditioned_reconstructed_count_validation()
# results = run_conditioned_reconstructed_count_regimes()
