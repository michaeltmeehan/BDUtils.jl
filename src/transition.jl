using Random
using Printf

# If running from package context, uncomment/edit:
# using BDUtils

# ------------------------------------------------------------
# Analytical kernel
# ------------------------------------------------------------

function phi_jkl(tj, tk, tl, pars)
    ϕ_direct = reconstructed_gamma_bd(0.0, tj, tk, tl, pars)

    # Optional explicit cross-check:
    # ϕ = ((1 - p_k^l) * γ_jk(0)) / (1 - p_k^l * γ_jk(0))
    g_jk = gamma_bd(0.0, tj, tk, pars)
    p_kl = unsampled_probability(tk, tl, pars)
    ϕ_explicit = ((1 - p_kl) * g_jk) / (1 - p_kl * g_jk)

    @assert isapprox(ϕ_direct, ϕ_explicit; rtol=1e-10, atol=1e-12)

    return ϕ_direct
end

function transition_kernel_no_sampling(ak::Integer, aj::Integer, ϕ::Real)
    aj >= 1 || throw(ArgumentError("aj must be >= 1."))
    ak < aj && return 0.0

    return binomial(ak - 1, aj - 1) *
           (1 - ϕ)^aj *
           ϕ^(ak - aj)
end

# ------------------------------------------------------------
# Empirical conditional transition counts
# ------------------------------------------------------------

function empirical_transition_counts(logs, tj, tk, tl)
    counts = Dict{Tuple{Int,Int},Int}()
    row_totals = Dict{Int,Int}()

    n_used = 0
    n_conditioned_out = 0

    for log in logs
        ΔS_jk = S_at(log, tk) - S_at(log, tj)

        if ΔS_jk == 0
            nj = N_at(log, tj)
            aj = A_at(log, tj, tl)
            ak = A_at(log, tk, tl)

            # Kernel is for aj >= 1.
            if aj >= 1 && nj ==  1
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

function compare_transition_kernel(logs, tj, tk, tl, pars)
    empirical = empirical_transition_counts(logs, tj, tk, tl)
    ϕ = phi_jkl(tj, tk, tl, pars)

    rows = sort(collect(keys(empirical.row_totals)))
    summaries = NamedTuple[]

    for aj in rows
        total = empirical.row_totals[aj]
        observed_aks = sort([ak for ((a, ak), _) in empirical.counts if a == aj])
        max_ak = maximum(observed_aks)

        tv_retained = 0.0
        max_abs_error = 0.0
        analytical_retained_mass = 0.0

        for ak in aj:max_ak
            emp = get(empirical.counts, (aj, ak), 0) / total
            ana = transition_kernel_no_sampling(ak, aj, ϕ)

            analytical_retained_mass += ana
            err = abs(emp - ana)
            tv_retained += err
            max_abs_error = max(max_abs_error, err)
        end

        push!(summaries, (
            aj = aj,
            n = total,
            max_ak = max_ak,
            phi = ϕ,
            tv_retained = 0.5 * tv_retained,
            max_abs_error = max_abs_error,
            analytical_retained_mass = analytical_retained_mass,
            analytical_tail_beyond_max_ak = max(0.0, 1.0 - analytical_retained_mass),
        ))
    end

    return (
        phi = ϕ,
        n_used = empirical.n_used,
        n_conditioned_out = empirical.n_conditioned_out,
        empirical_counts = empirical.counts,
        row_summaries = summaries,
    )
end

# ------------------------------------------------------------
# Main validation run
# ------------------------------------------------------------

function run_transition_kernel_validation(;
    seed = 1234,
    nrep = 100_000,
    pars = ConstantRateBDParameters(1.2, 0.4, 0.35, 0.7, 0.8),
    tj = 1.0,
    tk = 2.0,
    tl = 4.0,
    initial_lineages = 1,
)
    rng = MersenneTwister(seed)

    logs = [
        simulate_bd(rng, pars, tl;
            initial_lineages = initial_lineages,
            apply_ρ₀ = false,
        )
        for _ in 1:nrep
    ]

    result = compare_transition_kernel(logs, tj, tk, tl, pars)

    println()
    println("No-sampling reconstructed transition-kernel validation")
    println("------------------------------------------------------")
    @printf("nrep                 = %d\n", nrep)
    @printf("used after condition = %d\n", result.n_used)
    @printf("conditioned out      = %d\n", result.n_conditioned_out)
    @printf("t_j, t_k, t_l         = %.3f, %.3f, %.3f\n", tj, tk, tl)
    @printf("phi_jk^l             = %.8f\n", result.phi)
    println()

    println("Row-wise comparison by A_j^l = a_j")
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

    return result
end

result = run_transition_kernel_validation()