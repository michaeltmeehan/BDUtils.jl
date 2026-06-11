using Random
using Printf
using BDUtils

# Manual stochastic validation for
# P(N_j = n | A_j^l = a, A_i^l = 1).
#
# This is a scientific Monte Carlo diagnostic, not a deterministic unit test.
# Discrepancies should be interpreted relative to Monte Carlo standard errors.
# The empirical A_j^l values are computed from simulated ancestry via `A_at`,
# including terminal rho sampling at t_l when `apply_ρ₀=true`.

function _hidden_inversion_nmax(a, ti, tj, tl, pars; tail_atol=1e-8, max_nmax=10_000)
    cumulative = 0.0
    start = a >= 1 ? a : 0
    for n in start:max_nmax
        cumulative += hidden_count_given_reconstructed_count_pmf(n, a, ti, tj, tl, pars)
        1 - cumulative <= tail_atol && return n
    end
    return max_nmax
end

function _mean_variance_from_probs(probs)
    mean = sum(n * p for (n, p) in probs)
    var = sum((n - mean)^2 * p for (n, p) in probs)
    return mean, var
end

function _csv_escape(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function _write_hidden_inversion_csv(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        columns = (:regime, :a, :n, :retained, :empirical, :analytical, :mcse, :abs_error, :z)
        println(io, join(columns, ","))
        for row in rows
            println(io, join((_csv_escape(getproperty(row, c)) for c in columns), ","))
        end
    end
    return path
end

function hidden_reconstructed_inversion_validation_regime(;
    regime = "moderate",
    seed = 20240611,
    nrep = 100_000,
    pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 0.8),
    ti = 0.0,
    tj = 1.0,
    tl = 2.0,
    apply_ρ₀ = true,
    strata = (0, 1, 2, 3),
    tail_atol = 1e-8,
)
    rng = MersenneTwister(seed)
    counts_by_a = Dict{Int,Dict{Int,Int}}()
    retained = 0
    conditioned_out = 0

    for _ in 1:nrep
        log = simulate_bd(rng, pars, tl; initial_lineages=1, apply_ρ₀=apply_ρ₀)
        if A_at(log, ti, tl) == 1
            retained += 1
            a = A_at(log, tj, tl)
            n = N_at(log, tj)
            counts = get!(counts_by_a, a, Dict{Int,Int}())
            counts[n] = get(counts, n, 0) + 1
        else
            conditioned_out += 1
        end
    end

    summaries = NamedTuple[]
    rows = NamedTuple[]
    for a in strata
        counts = get(counts_by_a, a, Dict{Int,Int}())
        Ra = sum(values(counts))
        if Ra == 0
            push!(summaries, (
                regime=regime, a=a, retained=0, tv=NaN, max_abs_error=NaN,
                max_abs_z=NaN, empirical_mean=NaN, analytical_mean=NaN,
                empirical_variance=NaN, analytical_variance=NaN,
            ))
            continue
        end

        nmax = max(maximum(keys(counts)), _hidden_inversion_nmax(a, ti, tj, tl, pars; tail_atol=tail_atol))
        analytical_probs = Dict(n => hidden_count_given_reconstructed_count_pmf(n, a, ti, tj, tl, pars) for n in 0:nmax)
        empirical_probs = Dict(n => get(counts, n, 0) / Ra for n in 0:nmax)
        tv = 0.5 * sum(abs(empirical_probs[n] - analytical_probs[n]) for n in 0:nmax)
        max_abs_error = maximum(abs(empirical_probs[n] - analytical_probs[n]) for n in 0:nmax)
        z_values = Float64[]

        for n in 0:nmax
            p = analytical_probs[n]
            phat = empirical_probs[n]
            mcse = sqrt(p * (1 - p) / Ra)
            z = mcse > 0 ? (phat - p) / mcse : NaN
            Ra * p >= 5 && isfinite(z) && push!(z_values, abs(z))
            push!(rows, (
                regime=regime, a=a, n=n, retained=Ra, empirical=phat,
                analytical=p, mcse=mcse, abs_error=abs(phat - p), z=z,
            ))
        end

        empirical_mean, empirical_var = _mean_variance_from_probs(empirical_probs)
        analytical_mean, analytical_var = _mean_variance_from_probs(analytical_probs)
        push!(summaries, (
            regime=regime,
            a=a,
            retained=Ra,
            tv=tv,
            max_abs_error=max_abs_error,
            max_abs_z=isempty(z_values) ? NaN : maximum(z_values),
            empirical_mean=empirical_mean,
            analytical_mean=analytical_mean,
            empirical_variance=empirical_var,
            analytical_variance=analytical_var,
        ))
    end

    return (
        regime=regime,
        seed=seed,
        nrep=nrep,
        retained=retained,
        conditioned_out=conditioned_out,
        retained_fraction=retained / nrep,
        pars=pars,
        ti=ti,
        tj=tj,
        tl=tl,
        apply_ρ₀=apply_ρ₀,
        summaries=summaries,
        rows=rows,
    )
end

function print_hidden_reconstructed_inversion_validation(result)
    println()
    println("Hidden reconstructed inversion validation")
    println("-----------------------------------------")
    @printf("regime             = %s\n", result.regime)
    @printf("seed               = %d\n", result.seed)
    @printf("nrep               = %d\n", result.nrep)
    @printf("retained A_i^l=1   = %d\n", result.retained)
    @printf("conditioned out    = %d\n", result.conditioned_out)
    @printf("retained fraction  = %.8f\n", result.retained_fraction)
    @printf("parameters         = %s\n", string(result.pars))
    @printf("t_i, t_j, t_l       = %.3f, %.3f, %.3f\n", result.ti, result.tj, result.tl)
    @printf("apply_rho0         = %s\n", string(result.apply_ρ₀))

    println()
    println("a\tretained\tTV\tmax_abs_error\tmax_abs_z\temp_mean\tana_mean\temp_var\tana_var")
    for row in result.summaries
        @printf("%d\t%d\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\n",
            row.a, row.retained, row.tv, row.max_abs_error, row.max_abs_z,
            row.empirical_mean, row.analytical_mean, row.empirical_variance,
            row.analytical_variance)
    end
    return nothing
end

function run_hidden_reconstructed_inversion_validation(;
    seed = 20240611,
    nrep = 100_000,
    write_csv = true,
    csv_path = joinpath(@__DIR__, "output", "hidden_reconstructed_inversion_validation.csv"),
)
    regimes = (
        (regime="moderate", pars=ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 0.8)),
        (regime="high_sampling", pars=ConstantRateBDParameters(1.2, 0.4, 1.2, 1.0, 0.95)),
        (regime="low_sampling", pars=ConstantRateBDParameters(1.4, 0.3, 0.12, 0.7, 0.1)),
    )

    results = [
        hidden_reconstructed_inversion_validation_regime(
            regime=regime.regime,
            seed=seed + i - 1,
            nrep=nrep,
            pars=regime.pars,
        )
        for (i, regime) in enumerate(regimes)
    ]

    foreach(print_hidden_reconstructed_inversion_validation, results)

    if write_csv
        rows = reduce(vcat, [collect(result.rows) for result in results]; init=NamedTuple[])
        _write_hidden_inversion_csv(csv_path, rows)
        println()
        println("Wrote CSV diagnostics to ", csv_path)
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    nrep = parse(Int, get(ENV, "NREP", "100000"))
    run_hidden_reconstructed_inversion_validation(nrep=nrep)
end
