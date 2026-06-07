using Test
using BDUtils
using TreeSim
using Random

function tiny_tree()
    return Tree(
        [0.0, 0.6, 1.0, 1.4],
        [2, 3, 0, 0],
        [0, 4, 0, 0],
        [0, 1, 2, 2],
        [Root, Binary, SampledLeaf, SampledLeaf],
        [0, 0, 0, 0],
        [0, 0, 101, 102],
    )
end

function unsampled_unary_tree()
    return Tree(
        [0.0, 0.5, 1.0],
        [2, 3, 0],
        [0, 0, 0],
        [0, 1, 2],
        [Root, UnsampledUnary, SampledLeaf],
        [0, 0, 0],
        [0, 0, 101],
    )
end

function root_only_tree()
    return Tree(
        [0.0],
        [0],
        [0],
        [0],
        [Root],
        [0],
        [0],
    )
end

function scalar_joint_pgf(z, w, ti, tj, pars)
    γw = gamma_bd(w, ti, tj, pars)
    return alpha_bd(w, ti, tj, pars) + beta_bd(w, ti, tj, pars) * z / (1 - γw * z)
end

bd_a(w, pars) = pars.μ + pars.r * pars.ψ * w
bd_b_backward(w, pars) = -(pars.λ + bd_a(w, pars) + pars.ψ * (1 - w))

function backward_generator(y, w, pars)
    return (y - 1) * (pars.λ * y - bd_a(w, pars)) - pars.ψ * (1 - w) * y
end

function p_unsampled(tj, tk, pars)
    z = 1 - pars.ρ₀
    α0 = alpha_bd(0.0, tj, tk, pars)
    β0 = beta_bd(0.0, tj, tk, pars)
    γ0 = gamma_bd(0.0, tj, tk, pars)
    return α0 + β0 * z / (1 - γ0 * z)
end

function transformed_rates(tj, tk, pars)
    p = p_unsampled(tj, tk, pars)
    one_minus_p = 1 - p
    return (
        λ=pars.λ * one_minus_p,
        μ=pars.ψ * (pars.r + (1 - pars.r) * p) / one_minus_p,
        ψ=pars.ψ / one_minus_p,
    )
end

function transformed_alpha_beta_gamma(w, ti, tj, tk, pars)
    p = p_unsampled(tj, tk, pars)
    αij = alpha_bd(w, ti, tj, pars)
    βij = beta_bd(w, ti, tj, pars)
    γij = gamma_bd(w, ti, tj, pars)
    den = 1 - γij * p
    return (
        α=αij + βij * p / den,
        β=βij * (1 - p) / den^2,
        γ=1 - (1 - γij) / den,
    )
end

function central_first_derivative(f, x; h=1e-5)
    return (f(x + h) - f(x - h)) / (2h)
end

function central_second_derivative(f, x; h=1e-4)
    return (f(x + h) - 2f(x) + f(x - h)) / (h^2)
end

function table_pgf_sum(table, z, w)
    total = zero(eltype(table))
    for n in 0:(size(table, 1) - 1), s in 0:(size(table, 2) - 1)
        total += table[n + 1, s + 1] * z^n * w^s
    end
    return total
end

function total_variation_on_support(empirical::Dict, analytical::Dict, support)
    return 0.5 * sum(abs(get(empirical, key, 0.0) - get(analytical, key, 0.0)) for key in support)
end

function max_abs_error_on_support(empirical::Dict, analytical::Dict, support)
    isempty(support) && return 0.0
    return maximum(abs(get(empirical, key, 0.0) - get(analytical, key, 0.0)) for key in support)
end

function simulate_original_process(seed, pars, tmax, nsims)
    rng = MersenneTwister(seed)
    return [simulate_bd(rng, pars, tmax; apply_ρ₀=false) for _ in 1:nsims]
end

function analytical_joint_dict(table)
    return Dict((n - 1, s - 1) => table[n, s] for n in axes(table, 1), s in axes(table, 2))
end

function analytical_marginal_dict(values)
    return Dict(i - 1 => values[i] for i in eachindex(values))
end

function original_process_validation_summary(seed, pars, tj, nsims; tail_atol=2e-4, max_smax=1_000)
    logs = simulate_original_process(seed, pars, tj, nsims)
    nmax = n_truncation(0.0, tj, pars; atol=tail_atol)
    smax = s_truncation(0.0, tj, pars; atol=tail_atol, max_smax=max_smax)
    diagnostic = joint_pmf_NS_table(nmax, smax, 0.0, tj, pars; diagnostics=true)
    analytical_joint = analytical_joint_dict(diagnostic.table)
    support = collect(keys(analytical_joint))

    empirical_counts = joint_counts_NS(logs, tj)
    empirical_joint = joint_pmf_NS(empirical_counts)
    empirical_marginals = marginal_pmf_NS(empirical_counts)

    analytical_n = analytical_marginal_dict([n_marginal_pmf(n, 0.0, tj, pars) for n in 0:nmax])
    analytical_s = analytical_marginal_dict([s_marginal_pmf(s, 0.0, tj, pars) for s in 0:smax])
    n_support = collect(keys(analytical_n))
    s_support = collect(keys(analytical_s))

    empirical_retained = sum(get(empirical_joint, key, 0.0) for key in support)
    empirical_n_retained = sum(get(empirical_marginals.N, n, 0.0) for n in n_support)
    empirical_s_retained = sum(get(empirical_marginals.S, s, 0.0) for s in s_support)

    return (
        diagnostic=diagnostic,
        empirical_joint=empirical_joint,
        analytical_joint=analytical_joint,
        empirical_marginals=empirical_marginals,
        analytical_n=analytical_n,
        analytical_s=analytical_s,
        support=support,
        n_support=n_support,
        s_support=s_support,
        empirical_retained=empirical_retained,
        empirical_n_retained=empirical_n_retained,
        empirical_s_retained=empirical_s_retained,
        joint_tv=total_variation_on_support(empirical_joint, analytical_joint, support),
        joint_maxerr=max_abs_error_on_support(empirical_joint, analytical_joint, support),
        n_tv=total_variation_on_support(empirical_marginals.N, analytical_n, n_support),
        n_maxerr=max_abs_error_on_support(empirical_marginals.N, analytical_n, n_support),
        s_tv=total_variation_on_support(empirical_marginals.S, analytical_s, s_support),
        s_maxerr=max_abs_error_on_support(empirical_marginals.S, analytical_s, s_support),
    )
end

function reconstructed_validation_summary(seed, pars, tj, tk, nsims; tail_atol=2e-4, max_smax=1_000, apply_ρ₀=false)
    rng = MersenneTwister(seed)
    logs = [simulate_bd(rng, pars, tk; apply_ρ₀=apply_ρ₀) for _ in 1:nsims]

    amax = reconstructed_count_truncation(0.0, tj, tk, pars; atol=tail_atol)
    smax = reconstructed_sampling_truncation(0.0, tj, tk, pars; atol=tail_atol, max_smax=max_smax)
    diagnostic = reconstructed_joint_pmf_table(amax, smax, 0.0, tj, tk, pars; diagnostics=true)
    analytical_joint = analytical_joint_dict(diagnostic.table)
    support = collect(keys(analytical_joint))

    empirical_a = reconstructed_pmf_A(logs, tj)
    empirical_joint = reconstructed_joint_pmf_AS(logs, tj)
    analytical_a = analytical_marginal_dict([reconstructed_count_pmf(a, 0.0, tj, tk, pars) for a in 0:amax])
    analytical_s = analytical_marginal_dict([reconstructed_sampling_marginal_pmf(s, 0.0, tj, tk, pars) for s in 0:smax])
    s_counts = marginal_counts_NS(joint_counts_NS(logs, tj)).S
    total = length(logs)
    empirical_s = Dict(k => v / total for (k, v) in s_counts)

    a_support = collect(keys(analytical_a))
    s_support = collect(keys(analytical_s))
    empirical_retained = sum(get(empirical_joint, key, 0.0) for key in support)
    empirical_a_retained = sum(get(empirical_a, a, 0.0) for a in a_support)
    empirical_s_retained = sum(get(empirical_s, s, 0.0) for s in s_support)
    empirical_a_mean = sum(a * p for (a, p) in empirical_a)
    η = reconstructed_eta(0.0, tj, tk, pars)
    count_tail = reconstructed_count_tail(amax, 0.0, tj, tk, pars)
    analytical_a_mean = sum(a * reconstructed_count_pmf(a, 0.0, tj, tk, pars) for a in 0:amax) +
                        count_tail * ((amax + 1) + η / (1 - η))

    return (
        logs=logs,
        diagnostic=diagnostic,
        empirical_a=empirical_a,
        empirical_s=empirical_s,
        empirical_joint=empirical_joint,
        analytical_a=analytical_a,
        analytical_s=analytical_s,
        analytical_joint=analytical_joint,
        support=support,
        a_support=a_support,
        s_support=s_support,
        empirical_retention_probability=empirical_retention_probability(logs, tj, tk),
        analytical_retention_probability=1 - unsampled_probability(tj, tk, pars),
        empirical_a_zero=get(empirical_a, 0, 0.0),
        analytical_a_zero=reconstructed_count_pmf(0, 0.0, tj, tk, pars),
        empirical_a_mean=empirical_a_mean,
        analytical_a_mean=analytical_a_mean,
        empirical_retained=empirical_retained,
        empirical_a_retained=empirical_a_retained,
        empirical_s_retained=empirical_s_retained,
        joint_tv=total_variation_on_support(empirical_joint, analytical_joint, support),
        joint_maxerr=max_abs_error_on_support(empirical_joint, analytical_joint, support),
        a_tv=total_variation_on_support(empirical_a, analytical_a, a_support),
        a_maxerr=max_abs_error_on_support(empirical_a, analytical_a, a_support),
        s_tv=total_variation_on_support(empirical_s, analytical_s, s_support),
        s_maxerr=max_abs_error_on_support(empirical_s, analytical_s, s_support),
    )
end

function assert_reconstructed_validation(summary;
                                         joint_tv_atol,
                                         marginal_tv_atol,
                                         maxerr_atol,
                                         tail_slack,
                                         retention_atol)
    @test summary.diagnostic.retained_mass >= 1.0 - summary.diagnostic.count_tail_mass - summary.diagnostic.sampling_tail_mass - 1e-10
    @test summary.diagnostic.retained_mass <= 1.0 + 1e-10
    @test abs(summary.empirical_retention_probability - summary.analytical_retention_probability) <= retention_atol
    @test abs(summary.empirical_a_zero - summary.analytical_a_zero) <= maxerr_atol
    @test abs(summary.empirical_a_mean - summary.analytical_a_mean) <= 2 * maxerr_atol
    @test abs(summary.empirical_retained - summary.diagnostic.retained_mass) <= tail_slack
    @test abs(summary.empirical_a_retained - (1 - summary.diagnostic.count_tail_mass)) <= tail_slack
    @test abs(summary.empirical_s_retained - (1 - summary.diagnostic.sampling_tail_mass)) <= tail_slack
    @test summary.joint_tv <= joint_tv_atol
    @test summary.a_tv <= marginal_tv_atol
    @test summary.s_tv <= marginal_tv_atol
    @test summary.joint_maxerr <= maxerr_atol
    @test summary.a_maxerr <= maxerr_atol
    @test summary.s_maxerr <= maxerr_atol
end

function assert_original_process_validation(summary;
                                            joint_tv_atol,
                                            marginal_tv_atol,
                                            maxerr_atol,
                                            tail_slack)
    @test summary.diagnostic.retained_mass >= 1.0 - summary.diagnostic.n_tail_mass - summary.diagnostic.s_tail_mass - 1e-10
    @test summary.diagnostic.retained_mass <= 1.0 + 1e-10
    @test abs(summary.empirical_retained - summary.diagnostic.retained_mass) <= tail_slack
    @test abs(summary.empirical_n_retained - (1 - summary.diagnostic.n_tail_mass)) <= tail_slack
    @test abs(summary.empirical_s_retained - (1 - summary.diagnostic.s_tail_mass)) <= tail_slack
    @test summary.joint_tv <= joint_tv_atol
    @test summary.n_tv <= marginal_tv_atol
    @test summary.s_tv <= marginal_tv_atol
    @test summary.joint_maxerr <= maxerr_atol
    @test summary.n_maxerr <= maxerr_atol
    @test summary.s_maxerr <= maxerr_atol
end

# Constant-rate analytical regression fixtures.
#
# Core invariants protected below:
# - closed-form α/β/γ agree with their formal power-series coefficients,
# - joint PMF tables agree with scalar PMFs, marginals, tails, and PGFs,
# - forward and backward triangular ODE residuals vanish,
# - scalar PGF residuals satisfy the backward Kolmogorov equation,
# - Kendall, Riccati, and transformed-rate identities remain coherent.
#
# The stress grid intentionally spans small/large intervals, near-critical
# dynamics, low/no sampling, r = 0, and high sampling/removal cases.
const ODE_REGIMES = (
    ConstantRateBDParameters(0.9, 1.1, 0.4, 0.6),
    ConstantRateBDParameters(1.02, 0.8, 0.4, 0.5),
    ConstantRateBDParameters(1.8, 0.5, 0.7, 0.4),
    ConstantRateBDParameters(1.3, 0.6, 0.0, 0.0),
    ConstantRateBDParameters(1.4, 0.6, 0.7, 0.0),
    ConstantRateBDParameters(1.5, 0.2, 2.5, 0.95),
)

const KENDALL_REGIMES = (
    ODE_REGIMES[1],
    ODE_REGIMES[2],
    ODE_REGIMES[3],
    ODE_REGIMES[5],
    ODE_REGIMES[6],
)

const STRESS_REGIMES = (
    (name="small_t", pars=ConstantRateBDParameters(2.0, 0.5, 0.4, 0.7), ti=0.0, tj=1e-6),
    (name="moderate_t", pars=ConstantRateBDParameters(2.0, 0.5, 0.4, 0.7), ti=0.0, tj=1.5),
    (name="near_critical", pars=ConstantRateBDParameters(1.02, 0.8, 0.4, 0.5), ti=0.0, tj=2.0),
    (name="low_sampling", pars=ConstantRateBDParameters(1.4, 0.8, 1e-8, 0.3), ti=0.0, tj=1.0),
    (name="no_sampling", pars=ConstantRateBDParameters(1.4, 0.8, 0.0, 0.0), ti=0.0, tj=1.0),
    (name="high_sampling_removal", pars=ConstantRateBDParameters(1.5, 0.2, 3.0, 0.95), ti=0.0, tj=1.2),
    (name="zero_removal", pars=ConstantRateBDParameters(1.5, 0.6, 0.8, 0.0), ti=0.0, tj=1.4),
    (name="larger_t_subcritical", pars=ConstantRateBDParameters(0.9, 1.1, 0.6, 0.8), ti=0.0, tj=4.0),
)

const FORWARD_W_VALUES = (0.0, 0.3, 0.75, 1.0)
const BACKWARD_W_VALUES = (0.0, 0.25, 0.7, 1.0)
const FORWARD_TJ_VALUES = (0.55, 1.1, 1.8)
const BACKWARD_T_PAIRS = ((0.1, 0.45), (0.2, 1.2), (0.7, 2.0))
const PGF_T_PAIRS = ((0.1, 0.6), (0.3, 1.4), (0.8, 2.2))
