using Random
using Printf
using BDUtils

# Manual validation for sampling_time_likelihood.
#
# This script is intentionally kept outside the normal test suite. It compares
# the public likelihood with an independent latent-count summation and with a
# fixed-seed Monte Carlo window-density estimate from simulate_bd.
#
# Increase the simulation budget with:
#   BDUTILS_SAMPLING_TIME_VALIDATION_NSIMS=100000 julia --project=. scripts/validation/sampling_time_likelihood_validation.jl

const DEFAULT_NSIMS = 20_000
const NSIMS = parse(Int, get(ENV, "BDUTILS_SAMPLING_TIME_VALIDATION_NSIMS", string(DEFAULT_NSIMS)))
const HALF_WIDTH = parse(Float64, get(ENV, "BDUTILS_SAMPLING_TIME_VALIDATION_HALF_WIDTH", "0.035"))
const SEED = 20240607

function conditioned_coeff(n, ti, tj, tl, pars)
    n < 0 && return 0.0
    alpha = conditioned_reconstructed_alpha_bd(0.0, ti, tj, tl, pars)
    beta = conditioned_reconstructed_beta_bd(0.0, ti, tj, tl, pars)
    gamma = conditioned_reconstructed_gamma_bd(0.0, ti, tj, tl, pars)
    n == 0 && return alpha
    return beta * gamma^(n - 1)
end

function coeff_power(a, b, ti, tj, tl, pars)
    a < 0 && return 0.0
    a == 0 && return b == 0 ? 1.0 : 0.0
    masses = zeros(Float64, b + 1)
    masses[1] = 1.0
    for _ in 1:a
        next = zeros(Float64, b + 1)
        for current in 0:b
            iszero(masses[current + 1]) && continue
            for added in 0:(b - current)
                next[current + added + 1] += masses[current + 1] *
                    conditioned_coeff(added, ti, tj, tl, pars)
            end
        end
        masses = next
    end
    return masses[b + 1]
end

function removal_jump(b, c, psi_tilde; labelled_samples=false)
    c > b && return 0.0
    coefficient = labelled_samples ? prod((b - k for k in 0:(c - 1)); init=1) : binomial(b, c)
    return coefficient * psi_tilde^c
end

function brute_force_sampling_time_likelihood(
    t0,
    sampling_times,
    sample_counts,
    terminal_count,
    pars;
    tl,
    max_count,
    labelled_samples=false,
)
    f = Dict{Int,Float64}(1 => 1.0)
    u = t0
    for i in eachindex(sampling_times)
        ti = sampling_times[i]
        c = sample_counts[i]
        g = Dict{Int,Float64}()
        for (a, mass) in f
            for b in 0:max_count
                k = coeff_power(a, b, u, ti, tl, pars)
                iszero(k) && continue
                g[b] = get(g, b, 0.0) + mass * k
            end
        end

        psi_tilde = transformed_sampling_rate(ti, tl, pars)
        next = Dict{Int,Float64}()
        for (b, mass) in g
            d = b - c
            d >= 0 || continue
            jump = removal_jump(b, c, psi_tilde; labelled_samples=labelled_samples)
            next[d] = get(next, d, 0.0) + mass * jump
        end
        f = next
        u = ti
    end

    return sum(mass * terminal_count_transition(u, tl, a, terminal_count, pars) for (a, mass) in f)
end

function terminal_samples(log, tl)
    return count(log.kind[i] == SerialSampling && log.time[i] == tl for i in eachindex(log.time))
end

function serial_window_counts(log, times, half_width)
    return [
        count(
            log.kind[i] == SerialSampling &&
            t - half_width < log.time[i] < t + half_width
            for i in eachindex(log.time)
        )
        for t in times
    ]
end

function outside_serial_count(log, t0, tl, times, half_width)
    return count(
        log.kind[i] == SerialSampling &&
        t0 < log.time[i] < tl &&
        all(!(t - half_width < log.time[i] < t + half_width) for t in times)
        for i in eachindex(log.time)
    )
end

function monte_carlo_window_density(case; nsims=NSIMS, half_width=HALF_WIDTH)
    any(iszero, case.counts) && return nothing
    case.pars.r == 1.0 || return nothing

    rng = MersenneTwister(case.seed)
    conditioned = 0
    hits = 0

    for _ in 1:nsims
        log = simulate_bd(rng, case.pars, case.tl; apply_ρ₀=true)
        A_at(log, case.t0, case.tl) == 1 || continue
        conditioned += 1

        serial_match = isempty(case.times) || serial_window_counts(log, case.times, half_width) == case.counts
        outside_match = isempty(case.times) ||
            outside_serial_count(log, case.t0, case.tl, case.times, half_width) == 0
        if serial_match &&
                terminal_samples(log, case.tl) == case.terminal_count &&
                outside_match
            hits += 1
        end
    end

    volume = (2half_width)^length(case.times)
    p_hat = hits / conditioned
    density = p_hat / volume
    se = sqrt(max(p_hat * (1 - p_hat), 0.0) / conditioned) / volume
    return (
        nsims=nsims,
        conditioned=conditioned,
        hits=hits,
        estimate=density,
        se=se,
        ci_low=density - 1.96se,
        ci_high=density + 1.96se,
    )
end

function fmt(x; digits=6)
    x === nothing && return "-"
    return @sprintf("%.*g", digits, x)
end

function diagnostic_summary(diagnostic)
    diagnostic === nothing && return "-"
    return @sprintf(
        "fv=%d; serial=%s; eff=%s; retained=%.6g",
        length(diagnostic.forward_vectors),
        string(round.(diagnostic.serial_contributions; sigdigits=4)),
        string(diagnostic.effective_max_counts),
        diagnostic.retained_mass,
    )
end

function run_case(case)
    likelihood = nothing
    brute = nothing
    diagnostic = nothing
    mc = nothing
    status = "ok"

    try
        likelihood = sampling_time_likelihood(
            case.t0,
            case.times,
            case.counts,
            case.terminal_count,
            case.pars;
            tℓ=case.tl,
            max_count=case.max_count,
        )
        diagnostic = sampling_time_likelihood(
            case.t0,
            case.times,
            case.counts,
            case.terminal_count,
            case.pars;
            tℓ=case.tl,
            max_count=case.max_count,
            diagnostics=true,
        )
        if case.brute
            brute = brute_force_sampling_time_likelihood(
                case.t0,
                case.times,
                case.counts,
                case.terminal_count,
                case.pars;
                tl=case.tl,
                max_count=something(case.max_count, sum(case.counts) + case.terminal_count + 8),
            )
        end
        mc = monte_carlo_window_density(case)
    catch err
        status = err isa ArgumentError ? "unsupported: $(err.msg)" : "error: $(sprint(showerror, err))"
    end

    return merge(case, (
        likelihood=likelihood,
        brute=brute,
        diagnostic=diagnostic,
        mc=mc,
        status=status,
    ))
end

function validation_cases()
    base = ConstantRateBDParameters(0.8, 0.25, 0.45, 1.0, 0.7)
    terminal_base = ConstantRateBDParameters(1.1, 0.35, 0.35, 1.0, 0.55)
    return [
        (name="one serial", pars=base, t0=0.0, times=[0.55], counts=[1], terminal_count=1,
         tl=1.0, max_count=9, brute=true, seed=SEED + 1),
        (name="two serial", pars=ConstantRateBDParameters(1.25, 0.25, 0.5, 1.0, 0.35),
         t0=0.0, times=[0.3, 0.75], counts=[1, 1], terminal_count=1,
         tl=1.1, max_count=9, brute=true, seed=SEED + 2),
        (name="terminal count", pars=terminal_base, t0=0.0, times=Float64[], counts=Int[],
         terminal_count=2, tl=1.2, max_count=nothing, brute=false, seed=SEED + 3),
        (name="rho0 = 0", pars=ConstantRateBDParameters(0.9, 0.2, 0.5, 1.0, 0.0),
         t0=0.0, times=[0.45], counts=[1], terminal_count=1,
         tl=1.0, max_count=8, brute=true, seed=SEED + 4),
        (name="rho0 = 1", pars=ConstantRateBDParameters(0.9, 0.2, 0.5, 1.0, 1.0),
         t0=0.0, times=[0.45], counts=[1], terminal_count=1,
         tl=1.0, max_count=8, brute=true, seed=SEED + 5),
        (name="r = 0", pars=ConstantRateBDParameters(0.9, 0.2, 0.5, 0.0, 0.6),
         t0=0.0, times=[0.45], counts=[1], terminal_count=1,
         tl=1.0, max_count=8, brute=false, seed=SEED + 6),
        (name="r = 1", pars=ConstantRateBDParameters(0.9, 0.2, 0.5, 1.0, 0.6),
         t0=0.0, times=[0.45], counts=[1], terminal_count=1,
         tl=1.0, max_count=8, brute=true, seed=SEED + 7),
        (name="zero checkpoint", pars=ConstantRateBDParameters(1.8, 0.5, 0.7, 1.0, 0.25),
         t0=0.0, times=[0.6], counts=[0], terminal_count=1,
         tl=1.4, max_count=8, brute=true, seed=SEED + 8),
    ]
end

function print_results(results)
    println("sampling_time_likelihood validation")
    println("-----------------------------------")
    @printf("nsims=%d  half_width=%.4f  seed=%d\n\n", NSIMS, HALF_WIDTH, SEED)
    println("case              likelihood    brute        abs_diff     mc_est      mc_se       95% CI                  cond/hits  status")
    println("----------------  ------------  -----------  -----------  ----------  ----------  ----------------------  ---------  ------")
    for row in results
        abs_diff = row.likelihood === nothing || row.brute === nothing ? nothing : abs(row.likelihood - row.brute)
        mc_est = row.mc === nothing ? nothing : row.mc.estimate
        mc_se = row.mc === nothing ? nothing : row.mc.se
        ci = row.mc === nothing ? "-" : @sprintf("[%.5g, %.5g]", row.mc.ci_low, row.mc.ci_high)
        condhits = row.mc === nothing ? "-" : @sprintf("%d/%d", row.mc.conditioned, row.mc.hits)
        @printf(
            "%-16s  %12s  %11s  %11s  %10s  %10s  %-22s  %-9s  %s\n",
            row.name,
            fmt(row.likelihood),
            fmt(row.brute),
            fmt(abs_diff),
            fmt(mc_est),
            fmt(mc_se),
            ci,
            condhits,
            row.status,
        )
    end

    println()
    println("diagnostics=true summaries")
    for row in results
        @printf("%-16s  %s\n", row.name, diagnostic_summary(row.diagnostic))
    end
end

results = run_case.(validation_cases())
print_results(results)
