using Random
using Printf

# ------------------------------------------------------------
# Analytical joint coefficient
# ------------------------------------------------------------

function reconstructed_transition_joint_mass(
    aj::Integer,
    ak::Integer,
    ti::Real,
    tj::Real,
    tk::Real,
    tl::Real,
    pars,
)
    aj >= 0 || throw(ArgumentError("aj must be non-negative."))
    ak >= 0 || throw(ArgumentError("ak must be non-negative."))

    βij = beta_bd(1.0, ti, tj, pars)
    γij = gamma_bd(1.0, ti, tj, pars)
    αij = alpha_bd(1.0, ti, tj, pars)

    αjk0 = alpha_bd(0.0, tj, tk, pars)
    βjk0 = beta_bd(0.0, tj, tk, pars)
    γjk0 = gamma_bd(0.0, tj, tk, pars)

    pkl = unsampled_probability(tk, tl, pars)

    q = αjk0 + βjk0 * pkl / (1.0 - γjk0 * pkl)

    if aj == 0
        return ak == 0 ? αij + βij * q / (1.0 - γij * q) : 0.0
    end

    if ak < aj
        return 0.0
    end

    βtilde = βjk0 * (1.0 - pkl) / (1.0 - γjk0 * pkl)^2
    γtilde = γjk0 * (1.0 - pkl) / (1.0 - γjk0 * pkl)

    return βij *
           βtilde^aj *
           binomial(ak - 1, aj - 1) *
           γtilde^(ak - aj) *
           γij^(aj - 1) /
           (1.0 - γij * q)^(aj + 1)
end

# ------------------------------------------------------------
# Empirical joint counts
# ------------------------------------------------------------

function empirical_joint_transition_counts(logs, tj, tk, tl)
    counts = Dict{Tuple{Int,Int},Int}()

    n_no_sampling = 0
    n_total = length(logs)

    for log in logs
        ΔS_jk = S_at(log, tk) - S_at(log, tj)

        if ΔS_jk == 0
            aj = A_at(log, tj, tl)
            ak = A_at(log, tk, tl)

            counts[(aj, ak)] = get(counts, (aj, ak), 0) + 1
            n_no_sampling += 1
        end
    end

    return (
        counts = counts,
        n_total = n_total,
        n_no_sampling = n_no_sampling,
    )
end

# ------------------------------------------------------------
# Compare empirical and analytical joint probabilities
# ------------------------------------------------------------

function compare_joint_transition_mass(
    logs,
    ti,
    tj,
    tk,
    tl,
    pars;
    amax_j = nothing,
    amax_k = nothing,
)
    empirical = empirical_joint_transition_counts(logs, tj, tk, tl)

    observed_pairs = collect(keys(empirical.counts))

    if isempty(observed_pairs)
        return (
            n_total = empirical.n_total,
            n_no_sampling = empirical.n_no_sampling,
            summaries = NamedTuple[],
        )
    end

    if isnothing(amax_j)
        amax_j = maximum(first.(observed_pairs))
    end

    if isnothing(amax_k)
        amax_k = maximum(last.(observed_pairs))
    end

    summaries = NamedTuple[]

    total_abs_error = 0.0
    max_abs_error = 0.0
    empirical_retained_mass = 0.0
    analytical_retained_mass = 0.0

    for aj in 0:amax_j
        for ak in 0:amax_k
            emp = get(empirical.counts, (aj, ak), 0) / empirical.n_total

            ana = reconstructed_transition_joint_mass(
                aj, ak, ti, tj, tk, tl, pars
            )

            err = abs(emp - ana)

            empirical_retained_mass += emp
            analytical_retained_mass += ana
            total_abs_error += err
            max_abs_error = max(max_abs_error, err)

            if emp > 0 || ana > 0
                push!(summaries, (
                    aj = aj,
                    ak = ak,
                    empirical = emp,
                    analytical = ana,
                    abs_error = err,
                ))
            end
        end
    end

    return (
        n_total = empirical.n_total,
        n_no_sampling = empirical.n_no_sampling,
        empirical_counts = empirical.counts,
        empirical_retained_mass = empirical_retained_mass,
        analytical_retained_mass = analytical_retained_mass,
        total_variation_retained = 0.5 * total_abs_error,
        max_abs_error = max_abs_error,
        summaries = summaries,
    )
end

# ------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------

function run_joint_transition_diagnostic(;
    seed = 1234,
    nrep = 100_000,
    pars = ConstantRateBDParameters(1.2, 0.4, 0.35, 0.7, 0.8),
    ti = 0.0,
    tj = 1.0,
    tk = 2.0,
    tl = 4.0,
    initial_lineages = 1,
    amax_j = nothing,
    amax_k = nothing,
)
    rng = MersenneTwister(seed)

    logs = [
        simulate_bd(rng, pars, tl;
            initial_lineages = initial_lineages,
            apply_ρ₀ = false,
        )
        for _ in 1:nrep
    ]

    result = compare_joint_transition_mass(
        logs, ti, tj, tk, tl, pars;
        amax_j = amax_j,
        amax_k = amax_k,
    )

    println()
    println("Joint transition-mass diagnostic")
    println("--------------------------------")
    @printf("nrep                  = %d\n", result.n_total)
    @printf("no-sampling count      = %d\n", result.n_no_sampling)
    @printf("t_i, t_j, t_k, t_l     = %.3f, %.3f, %.3f, %.3f\n", ti, tj, tk, tl)
    @printf("empirical retained     = %.8f\n", result.empirical_retained_mass)
    @printf("analytical retained    = %.8f\n", result.analytical_retained_mass)
    @printf("TV retained            = %.6g\n", result.total_variation_retained)
    @printf("max abs error          = %.6g\n", result.max_abs_error)
    println()

    println("Largest absolute errors")
    println("aj\tak\tempirical\tanalytical\tabs_error")

    sorted_rows = sort(result.summaries; by = row -> row.abs_error, rev = true)

    for row in Iterators.take(sorted_rows, 20)
        @printf(
            "%d\t%d\t%.8g\t%.8g\t%.8g\n",
            row.aj,
            row.ak,
            row.empirical,
            row.analytical,
            row.abs_error,
        )
    end

    return result
end


using Random
using Printf

# ------------------------------------------------------------
# Analytical conditional kernel
# ------------------------------------------------------------

function phi_jkl(tj, tk, tl, pars)
    γjk0 = gamma_bd(0.0, tj, tk, pars)
    pkl = unsampled_probability(tk, tl, pars)
    return γjk0 * (1.0 - pkl) / (1.0 - γjk0 * pkl)
end

function reconstructed_transition_kernel(ak::Integer, aj::Integer, ϕ::Real)
    aj >= 1 || throw(ArgumentError("aj must be >= 1."))
    ak < aj && return 0.0

    return binomial(ak - 1, aj - 1) *
           (1.0 - ϕ)^aj *
           ϕ^(ak - aj)
end

# ------------------------------------------------------------
# Empirical conditional counts
# ------------------------------------------------------------

function empirical_reconstructed_transition_counts(logs, tj, tk, tl)
    counts = Dict{Tuple{Int,Int},Int}()
    row_totals = Dict{Int,Int}()

    n_used = 0
    n_conditioned_out = 0

    for log in logs
        ΔS_jk = S_at(log, tk) - S_at(log, tj)

        if ΔS_jk == 0
            aj = A_at(log, tj, tl)
            ak = A_at(log, tk, tl)

            if aj >= 1
                counts[(aj, ak)] = get(counts, (aj, ak), 0) + 1
                row_totals[aj] = get(row_totals, aj, 0) + 1
                n_used += 1
            else
                n_conditioned_out += 1
            end
        else
            n_conditioned_out += 1
        end
    end

    return (
        counts = counts,
        row_totals = row_totals,
        n_used = n_used,
        n_conditioned_out = n_conditioned_out,
    )
end

# ------------------------------------------------------------
# Pairwise comparison table
# ------------------------------------------------------------

function compare_reconstructed_transition_kernel_pairs(
    logs,
    tj,
    tk,
    tl,
    pars;
    min_row_n = 1,
)
    empirical = empirical_reconstructed_transition_counts(logs, tj, tk, tl)
    ϕ = phi_jkl(tj, tk, tl, pars)

    rows = sort(collect(keys(empirical.row_totals)))
    pair_summaries = NamedTuple[]
    row_summaries = NamedTuple[]

    for aj in rows
        total = empirical.row_totals[aj]
        total < min_row_n && continue

        observed_aks = sort([ak for ((a, ak), _) in empirical.counts if a == aj])
        max_ak = maximum(observed_aks)

        tv_retained = 0.0
        max_abs_error = 0.0
        analytical_retained_mass = 0.0
        empirical_retained_mass = 0.0

        for ak in aj:max_ak
            emp_count = get(empirical.counts, (aj, ak), 0)
            emp = emp_count / total
            ana = reconstructed_transition_kernel(ak, aj, ϕ)

            empirical_retained_mass += emp
            analytical_retained_mass += ana

            err = emp - ana
            abs_err = abs(err)

            tv_retained += abs_err
            max_abs_error = max(max_abs_error, abs_err)

            push!(pair_summaries, (
                aj = aj,
                ak = ak,
                row_n = total,
                count = emp_count,
                empirical = emp,
                analytical = ana,
                error = err,
                abs_error = abs_err,
            ))
        end

        push!(row_summaries, (
            aj = aj,
            n = total,
            max_ak = max_ak,
            phi = ϕ,
            empirical_retained_mass = empirical_retained_mass,
            analytical_retained_mass = analytical_retained_mass,
            analytical_tail_beyond_max_ak = max(0.0, 1.0 - analytical_retained_mass),
            tv_retained = 0.5 * tv_retained,
            max_abs_error = max_abs_error,
        ))
    end

    return (
        phi = ϕ,
        n_used = empirical.n_used,
        n_conditioned_out = empirical.n_conditioned_out,
        empirical_counts = empirical.counts,
        row_summaries = row_summaries,
        pair_summaries = pair_summaries,
    )
end

# ------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------

function run_reconstructed_transition_kernel_pair_validation(;
    seed = 1234,
    nrep = 1_000_000,
    pars = ConstantRateBDParameters(1.2, 0.4, 0.35, 0.7, 0.8),
    tj = 1.0,
    tk = 2.0,
    tl = 4.0,
    initial_lineages = 1,
    min_row_n = 1,
    max_pairs_print = nothing,
)
    rng = MersenneTwister(seed)

    logs = [
        simulate_bd(rng, pars, tl;
            initial_lineages = initial_lineages,
            apply_ρ₀ = false,
        )
        for _ in 1:nrep
    ]

    result = compare_reconstructed_transition_kernel_pairs(
        logs, tj, tk, tl, pars;
        min_row_n = min_row_n,
    )

    println()
    println("Conditional reconstructed transition-kernel validation")
    println("------------------------------------------------------")
    @printf("nrep                  = %d\n", nrep)
    @printf("used after condition  = %d\n", result.n_used)
    @printf("conditioned out       = %d\n", result.n_conditioned_out)
    @printf("t_j, t_k, t_l          = %.3f, %.3f, %.3f\n", tj, tk, tl)
    @printf("phi_jk^l              = %.8f\n", result.phi)
    println()

    println("Row-wise summary by A_j^l = a_j")
    println("a_j\tn\tmax_ak\tTV(retained)\tmax_abs_error\tanalytic_tail")

    for row in result.row_summaries
        @printf(
            "%d\t%d\t%d\t%.6g\t%.6g\t%.6g\n",
            row.aj,
            row.n,
            row.max_ak,
            row.tv_retained,
            row.max_abs_error,
            row.analytical_tail_beyond_max_ak,
        )
    end

    println()
    println("Pairwise empirical - analytical comparison")
    println("a_j\ta_k\trow_n\tcount\tempirical\tanalytical\terror\tabs_error")

    rows_to_print = isnothing(max_pairs_print) ?
        result.pair_summaries :
        Iterators.take(result.pair_summaries, max_pairs_print)

    for row in rows_to_print
        @printf(
            "%d\t%d\t%d\t%d\t%.8g\t%.8g\t%.8g\t%.8g\n",
            row.aj,
            row.ak,
            row.row_n,
            row.count,
            row.empirical,
            row.analytical,
            row.error,
            row.abs_error,
        )
    end

    return result
end

# result_conditional = run_reconstructed_transition_kernel_pair_validation()


using Random
using Printf

# ------------------------------------------------------------
# Analytical backward / inverted bridge kernel
# ------------------------------------------------------------

function bridge_pi_unsimplified(ti, tj, tk, tl, pars)
    γij = gamma_bd(1.0, ti, tj, pars)

    αjk0 = alpha_bd(0.0, tj, tk, pars)
    βjk0 = beta_bd(0.0, tj, tk, pars)
    γjk0 = gamma_bd(0.0, tj, tk, pars)

    pkl = unsampled_probability(tk, tl, pars)

    q = αjk0 + βjk0 * pkl / (1.0 - γjk0 * pkl)

    βtilde = βjk0 * (1.0 - pkl) / (1.0 - γjk0 * pkl)^2
    γtilde = γjk0 * (1.0 - pkl) / (1.0 - γjk0 * pkl)

    ρ = βtilde * γij / (1.0 - γij * q)

    return ρ / (ρ + γtilde)
end

function bridge_pi(ti, tj, tk, tl, pars)
    x = gamma_bd(1.0, ti, tj, pars)

    a = alpha_bd(0.0, tj, tk, pars)
    b = beta_bd(0.0, tj, tk, pars)
    g = gamma_bd(0.0, tj, tk, pars)

    p = unsampled_probability(tk, tl, pars)

    return (b * x) /
           ((1.0 - g * p) * (g + x * (b - g * a)))
end

function check_bridge_pi(ti, tj, tk, tl, pars; atol = 1e-12, rtol = 1e-10)
    π1 = bridge_pi_unsimplified(ti, tj, tk, tl, pars)
    π2 = bridge_pi(ti, tj, tk, tl, pars)

    @assert isapprox(π1, π2; atol = atol, rtol = rtol)

    return π2
end

function inverted_transition_kernel(aj::Integer, ak::Integer, π::Real)
    ak >= 1 || throw(ArgumentError("ak must be >= 1."))
    aj < 1 && return 0.0
    aj > ak && return 0.0

    return binomial(ak - 1, aj - 1) *
           π^(aj - 1) *
           (1.0 - π)^(ak - aj)
end

# ------------------------------------------------------------
# Empirical inverted transition counts
#
# Estimates:
#
#   P(A_j^l = aj | A_k^l = ak, ΔS_jk = 0)
#
# starting from N_i = 1.
# ------------------------------------------------------------

function empirical_inverted_transition_counts(logs, tj, tk, tl)
    counts = Dict{Tuple{Int,Int},Int}()      # (aj, ak) => count
    col_totals = Dict{Int,Int}()             # ak => count

    n_used = 0
    n_conditioned_out = 0

    for log in logs
        ΔS_jk = S_at(log, tk) - S_at(log, tj)

        if ΔS_jk == 0
            aj = A_at(log, tj, tl)
            ak = A_at(log, tk, tl)

            if ak >= 1
                counts[(aj, ak)] = get(counts, (aj, ak), 0) + 1
                col_totals[ak] = get(col_totals, ak, 0) + 1
                n_used += 1
            else
                n_conditioned_out += 1
            end
        else
            n_conditioned_out += 1
        end
    end

    return (
        counts = counts,
        col_totals = col_totals,
        n_used = n_used,
        n_conditioned_out = n_conditioned_out,
    )
end

# ------------------------------------------------------------
# Column-wise comparison against binomial bridge kernel
# ------------------------------------------------------------

function compare_inverted_transition_kernel(
    logs,
    ti,
    tj,
    tk,
    tl,
    pars;
    min_col_n = 50,
)
    empirical = empirical_inverted_transition_counts(logs, tj, tk, tl)
    π = check_bridge_pi(ti, tj, tk, tl, pars)

    cols = sort(collect(keys(empirical.col_totals)))

    col_summaries = NamedTuple[]
    pair_summaries = NamedTuple[]

    for ak in cols
        total = empirical.col_totals[ak]
        total < min_col_n && continue

        tv_retained = 0.0
        max_abs_error = 0.0
        empirical_retained_mass = 0.0
        analytical_retained_mass = 0.0

        for aj in 1:ak
            emp_count = get(empirical.counts, (aj, ak), 0)
            emp = emp_count / total
            ana = inverted_transition_kernel(aj, ak, π)

            empirical_retained_mass += emp
            analytical_retained_mass += ana

            err = emp - ana
            abs_err = abs(err)

            tv_retained += abs_err
            max_abs_error = max(max_abs_error, abs_err)

            push!(pair_summaries, (
                ak = ak,
                aj = aj,
                col_n = total,
                count = emp_count,
                empirical = emp,
                analytical = ana,
                error = err,
                abs_error = abs_err,
            ))
        end

        push!(col_summaries, (
            ak = ak,
            n = total,
            pi = π,
            empirical_retained_mass = empirical_retained_mass,
            analytical_retained_mass = analytical_retained_mass,
            tv_retained = 0.5 * tv_retained,
            max_abs_error = max_abs_error,
        ))
    end

    return (
        pi = π,
        n_used = empirical.n_used,
        n_conditioned_out = empirical.n_conditioned_out,
        empirical_counts = empirical.counts,
        col_summaries = col_summaries,
        pair_summaries = pair_summaries,
    )
end

# ------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------

function run_inverted_transition_kernel_validation(;
    seed = 1234,
    nrep = 1_000_000,
    pars = ConstantRateBDParameters(1.2, 0.4, 0.35, 0.7, 0.8),
    ti = 0.0,
    tj = 1.0,
    tk = 2.0,
    tl = 4.0,
    initial_lineages = 1,
    min_col_n = 50,
    max_pairs_print = nothing,
)
    rng = MersenneTwister(seed)

    logs = [
        simulate_bd(rng, pars, tl;
            initial_lineages = initial_lineages,
            apply_ρ₀ = false,
        )
        for _ in 1:nrep
    ]

    result = compare_inverted_transition_kernel(
        logs,
        ti,
        tj,
        tk,
        tl,
        pars;
        min_col_n = min_col_n,
    )

    println()
    println("Inverted reconstructed transition-kernel validation")
    println("---------------------------------------------------")
    @printf("nrep                  = %d\n", nrep)
    @printf("used after condition  = %d\n", result.n_used)
    @printf("conditioned out       = %d\n", result.n_conditioned_out)
    @printf("t_i, t_j, t_k, t_l     = %.3f, %.3f, %.3f, %.3f\n", ti, tj, tk, tl)
    @printf("pi                    = %.8f\n", result.pi)
    println()

    println("Column-wise comparison by A_k^l = a_k")
    println("a_k\tn\tTV(retained)\tmax_abs_error")

    for row in result.col_summaries
        @printf(
            "%d\t%d\t%.6g\t%.6g\n",
            row.ak,
            row.n,
            row.tv_retained,
            row.max_abs_error,
        )
    end

    println()
    println("Pairwise empirical - analytical comparison")
    println("a_k\ta_j\tcol_n\tcount\tempirical\tanalytical\terror\tabs_error")

    rows_to_print = isnothing(max_pairs_print) ?
        result.pair_summaries :
        Iterators.take(result.pair_summaries, max_pairs_print)

    for row in rows_to_print
        @printf(
            "%d\t%d\t%d\t%d\t%.8g\t%.8g\t%.8g\t%.8g\n",
            row.ak,
            row.aj,
            row.col_n,
            row.count,
            row.empirical,
            row.analytical,
            row.error,
            row.abs_error,
        )
    end

    return result
end

# result_inverted = run_inverted_transition_kernel_validation()