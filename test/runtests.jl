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
    γ0 = gamma_bd(0.0, tj, tk, pars)
    return 1 - pars.ψ / pars.λ * γ0 / (1 - γ0)
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

function reconstructed_validation_summary(seed, pars, tj, tk, nsims; tail_atol=2e-4, max_smax=1_000)
    rng = MersenneTwister(seed)
    logs = [simulate_bd(rng, pars, tk; apply_ρ₀=false) for _ in 1:nsims]

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

@testset "BDUtils constant-rate core" begin
    λ = 2.0
    μ = 0.5
    ψ = 0.4
    r = 0.7
    pars = ConstantRateBDParameters(λ, μ, ψ, r, 0.25)

    @testset "derived helpers" begin
        @test compute_R0(λ, μ, ψ, r) == λ / (μ + ψ * r)
        @test compute_delta(λ, μ, ψ, r) == μ + ψ * r
        @test compute_sampled_removal_rate(pars) == r * ψ
        @test compute_sampling_fraction(pars) == (r * ψ) / (μ + ψ * r)
        @test compute_R0([λ], [μ], [ψ], r) == [λ / (μ + ψ * r)]
        @test compute_R0(pars) == compute_R0(λ, μ, ψ, r)
        @test compute_delta(pars) == compute_delta(λ, μ, ψ, r)

        alt = reparameterize_R0_delta_s(pars)
        @test keys(alt) == (:R0, :δ, :s, :r, :ρ₀)
        @test alt.R0 == compute_R0(pars)
        @test alt.δ == compute_delta(pars)
        @test alt.s == compute_sampling_fraction(pars)
        @test alt.r == pars.r
        @test alt.ρ₀ == pars.ρ₀

        roundtrip = parameters_from_R0_delta_s_r(alt.R0, alt.δ, alt.s, alt.r, alt.ρ₀)
        @test roundtrip.λ ≈ pars.λ
        @test roundtrip.μ ≈ pars.μ
        @test roundtrip.ψ ≈ pars.ψ
        @test roundtrip.r == pars.r
        @test roundtrip.ρ₀ == pars.ρ₀

        hand = parameters_from_R0_delta_s_r(2.0, 1.5, 0.2, 0.5, 0.1)
        @test hand.λ == 3.0
        @test hand.μ ≈ 1.2
        @test hand.ψ ≈ 0.6
        @test hand.r == 0.5
        @test hand.ρ₀ == 0.1

        zero_r = parameters_from_R0_delta_s_r(2.0, 1.5, 0.0, 0.0, 0.25)
        @test zero_r.λ == 3.0
        @test zero_r.μ == 1.5
        @test zero_r.ψ == 0.0
        @test zero_r.r == 0.0
        @test zero_r.ρ₀ == 0.25

        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 1.5, 0.1, 0.0, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(0.0, 1.5, 0.0, 0.0, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 0.0, 0.0, 0.0, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 1.5, -0.1, 0.5, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 1.5, 1.1, 0.5, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 1.5, 0.1, -0.1, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 1.5, 0.1, 0.5, 1.1)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(Inf, 1.5, 0.1, 0.5, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, NaN, 0.1, 0.5, 0.0)
    end

    @testset "constant-rate parameter object" begin
        @test pars.λ == λ
        @test pars.μ == μ
        @test pars.ψ == ψ
        @test pars.r == r
        @test pars.ρ₀ == 0.25
        @test ConstantRateBDParameters(λ, μ, ψ, r).ρ₀ == 0.0
        @test ConstantRateBDParameters(2, 0, 0, 0, 1) isa ConstantRateBDParameters{Float64}
        @test sprint(show, pars) == "ConstantRateBDParameters(λ=2.0, μ=0.5, ψ=0.4, r=0.7, ρ₀=0.25)"

        @test ConstantRateBDParameters(λ, μ, ψ, 0.0, 0.0).r == 0.0
        @test ConstantRateBDParameters(λ, μ, ψ, 1.0, 1.0).ρ₀ == 1.0

        @test_throws ArgumentError ConstantRateBDParameters(0.0, μ, ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, -μ, ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, -ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, ψ, 1.1)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, ψ, r, -0.1)
        @test_throws ArgumentError ConstantRateBDParameters(Inf, μ, ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(NaN, μ, ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, Inf, ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, NaN, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, ψ, NaN)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, ψ, r, Inf)
    end

    @testset "constant-rate native simulation and empirical extraction" begin
        handcrafted = BDEventLog(
            [0.2, 0.4, 0.6, 0.8],
            [2, 2, 1, 3],
            [1, 0, 0, 2],
            [Birth, FossilizedSampling, SerialSampling, Death],
            1,
            1.0,
        )

        @test length(handcrafted) == 4
        @test handcrafted[1] == BDEventRecord(0.2, 2, 1, Birth)
        @test collect(record.kind for record in handcrafted) == [Birth, FossilizedSampling, SerialSampling, Death]
        @test sprint(show, handcrafted) == "BDEventLog(4 events, initial_lineages=1, tmax=1.0)"

        @test NS_at(handcrafted, 0.0) == (N=1, S=0)
        @test NS_at(handcrafted, 0.2) == (N=2, S=0)
        @test NS_at(handcrafted, 0.5) == (N=2, S=1)
        @test NS_at(handcrafted, 0.7) == (N=1, S=2)
        @test NS_at(handcrafted, 1.0) == (N=0, S=2)
        @test N_at(handcrafted, 0.7) == 1
        @test S_at(handcrafted, 0.7) == 2
        @test N_over_time(handcrafted, [0.0, 0.5, 1.0]) == [1, 2, 0]
        @test S_over_time(handcrafted, [0.0, 0.5, 1.0]) == [0, 1, 2]
        @test NS_over_time(handcrafted, [0.0, 0.7]) == [(N=1, S=0), (N=1, S=2)]

        @test extant_lineages_at(handcrafted, 0.0) == [1]
        @test extant_lineages_at(handcrafted, 0.2) == [1, 2]
        @test extant_lineages_at(handcrafted, 0.7) == [2]
        @test retained_lineages_at(handcrafted, 0.0) == [1]
        @test retained_lineages_at(handcrafted, 0.2) == [1, 2]
        @test retained_lineages_at(handcrafted, 0.5) == [1]
        @test retained_lineages_at(handcrafted, 0.7) == Int[]
        @test A_at(handcrafted, 0.0) == 1
        @test A_over_time(handcrafted, [0.0, 0.2, 0.7, 1.0]) == [1, 2, 0, 0]

        terminal_sample = BDEventLog([0.25, 1.0], [2, 2], [1, 0], [Birth, SerialSampling], 1, 1.0)
        @test extant_lineages_at(terminal_sample, 0.5) == [1, 2]
        @test retained_lineages_at(terminal_sample, 0.5) == [2]
        @test retained_lineages_at(terminal_sample, 1.0) == Int[]
        @test A_at(terminal_sample, 0.5) == 1

        fossil_at_tj = BDEventLog([0.5], [1], [0], [FossilizedSampling], 1, 1.0)
        @test retained_lineages_at(fossil_at_tj, 0.5) == Int[]
        @test A_at(fossil_at_tj, 0.49, 0.5) == 1

        serial_at_tj = BDEventLog([0.5], [1], [0], [SerialSampling], 1, 1.0)
        @test extant_lineages_at(serial_at_tj, 0.5) == Int[]
        @test A_at(serial_at_tj, 0.5) == 0
        @test A_at(serial_at_tj, 0.49, 0.5) == 1

        sample_after_tj = BDEventLog([0.500001], [1], [0], [FossilizedSampling], 1, 1.0)
        @test A_at(sample_after_tj, 0.5) == 1
        @test A_at(sample_after_tj, 0.500001) == 0

        truncated = BDEventLog([0.8, 0.9], [1, 1], [0, 0], [FossilizedSampling, SerialSampling], 1, 1.0)
        @test A_at(truncated, 0.5, 0.7) == 0
        @test A_at(truncated, 0.5, 0.8) == 1
        @test A_at(truncated, 0.85, 0.87) == 0
        @test A_at(truncated, 0.85, 0.9) == 1
        @test A_at(truncated, 0.8, 0.8) == 0
        @test A_over_time(truncated, [0.5, 0.8, 0.85]; tk=0.9) == [1, 1, 1]
        @test A_over_time(truncated, [0.5, 0.8]; tk=0.8) == [1, 0]

        extinct_unsampled = BDEventLog([0.1], [1], [0], [Death], 1, 1.0)
        counts = joint_counts_NS([handcrafted, extinct_unsampled], 1.0)
        @test counts == Dict((0, 2) => 1, (0, 0) => 1)
        @test joint_pmf_NS(counts) == Dict((0, 2) => 0.5, (0, 0) => 0.5)
        @test marginal_counts_NS(counts) == (N=Dict(0 => 2), S=Dict(2 => 1, 0 => 1))
        @test marginal_pmf_NS(counts) == (N=Dict(0 => 1.0), S=Dict(2 => 0.5, 0 => 0.5))
        @test reconstructed_counts_A([handcrafted, extinct_unsampled], 0.0) == Dict(1 => 1, 0 => 1)
        @test reconstructed_pmf_A([handcrafted, extinct_unsampled], 0.0) == Dict(1 => 0.5, 0 => 0.5)
        @test reconstructed_joint_counts_AS([handcrafted, extinct_unsampled], 0.5) == Dict((1, 1) => 1, (0, 0) => 1)
        @test reconstructed_joint_pmf_AS([handcrafted, extinct_unsampled], 0.5) == Dict((1, 1) => 0.5, (0, 0) => 0.5)
        @test empirical_retention_probability([handcrafted, extinct_unsampled], 0.0) == 0.5
        @test joint_counts_NS([handcrafted, extinct_unsampled], [0.0, 1.0]) == [
            Dict((1, 0) => 2),
            Dict((0, 2) => 1, (0, 0) => 1),
        ]

        birth_only = simulate_bd(MersenneTwister(1), ConstantRateBDParameters(1.0, 0.0, 0.0, 0.0), 0.5)
        @test birth_only isa BDEventLog
        @test all(==(Birth), birth_only.kind)
        @test issorted(birth_only.time)
        @test N_at(birth_only, 0.5) == birth_only.initial_lineages + length(birth_only)
        @test S_at(birth_only, 0.5) == 0

        serial_only = simulate_bd(MersenneTwister(2), ConstantRateBDParameters(1.0, 0.0, 10.0, 1.0), 10.0; apply_ρ₀=false)
        @test SerialSampling in serial_only.kind
        @test all(kind -> kind == Birth || kind == SerialSampling, serial_only.kind)
        @test N_at(serial_only, 10.0) >= 0
        @test S_at(serial_only, 10.0) == count(==(SerialSampling), serial_only.kind)

        contemp = simulate_bd(MersenneTwister(3), ConstantRateBDParameters(1.0, 0.0, 0.0, 0.0, 1.0), 0.0)
        @test length(contemp) == 1
        @test contemp.time == [0.0]
        @test contemp.kind == [SerialSampling]
        @test NS_at(contemp, 0.0) == (N=0, S=1)

        @test_throws ArgumentError simulate_bd(ConstantRateBDParameters(1.0, 0.0, 0.0, 0.0), -0.1)
        @test_throws ArgumentError simulate_bd(ConstantRateBDParameters(1.0, 0.0, 0.0, 0.0), 1.0; initial_lineages=-1)
        @test_throws ArgumentError NS_at(handcrafted, -0.1)
        @test_throws ArgumentError joint_pmf_NS(Dict{Tuple{Int,Int},Int}())
        @test_throws ArgumentError A_at(handcrafted, -0.1)
        @test_throws ArgumentError A_at(handcrafted, 0.8, 0.7)
        @test_throws ArgumentError A_at(handcrafted, 0.5, 1.1)
        @test_throws ArgumentError retained_lineages_at(handcrafted, 0.8, 0.7)
        @test_throws ArgumentError A_over_time(handcrafted, [0.1]; tk=1.1)
        @test_throws ArgumentError reconstructed_pmf_A(Dict{Int,Int}())
        @test_throws ArgumentError reconstructed_joint_pmf_AS(Dict{Tuple{Int,Int},Int}())
        @test_throws ArgumentError empirical_retention_probability([extinct_unsampled], 0.5)
    end

    @testset "TreeSim reconstructed extraction from BDEventLog" begin
        simple = BDEventLog([0.2, 0.5, 0.7], [2, 1, 2], [1, 0, 0], [Birth, FossilizedSampling, SerialSampling], 1, 1.0)

        forest = forest_from_eventlog(simple; tj=0.0, tk=1.0)
        @test length(forest) == A_at(simple, 0.0, 1.0) == 1
        @test validate_tree(only(forest); require_single_root=true, require_reachable=true)
        @test only(forest).kind == [Root, Binary, SampledLeaf, SampledLeaf]
        @test only(forest).host == [1, 1, 1, 2]

        after_birth = forest_from_eventlog(simple; tj=0.2, tk=1.0)
        @test length(after_birth) == A_at(simple, 0.2, 1.0) == 2
        @test sort([tree.host[TreeSim.root(tree)] for tree in after_birth]) == retained_lineages_at(simple, 0.2, 1.0)
        @test all(tree -> validate_tree(tree; require_single_root=true, require_reachable=true), after_birth)
        @test_throws ErrorException tree_from_eventlog(simple; tj=0.2, tk=1.0)

        fossil_chain = BDEventLog([0.3, 0.6], [1, 1], [0, 0], [FossilizedSampling, SerialSampling], 1, 1.0)
        chain_tree = tree_from_eventlog(fossil_chain; tj=0.0, tk=1.0)
        @test validate_tree(chain_tree; require_single_root=true, require_reachable=true)
        @test chain_tree.kind == [Root, SampledUnary, SampledLeaf]
        @test tree_from_eventlog(fossil_chain; tj=0.3, tk=1.0).kind == [Root, SampledLeaf]

        sample_at_tj = BDEventLog([0.5], [1], [0], [FossilizedSampling], 1, 1.0)
        @test forest_from_eventlog(sample_at_tj; tj=0.5, tk=1.0) == Tree[]
        @test length(forest_from_eventlog(sample_at_tj; tj=0.49, tk=0.5)) == A_at(sample_at_tj, 0.49, 0.5) == 1

        sample_at_tk = BDEventLog([0.5], [1], [0], [SerialSampling], 1, 1.0)
        @test length(forest_from_eventlog(sample_at_tk; tj=0.49, tk=0.5)) == A_at(sample_at_tk, 0.49, 0.5) == 1
        @test forest_from_eventlog(sample_at_tk; tj=0.5, tk=1.0) == Tree[]

        extinct_unsampled = BDEventLog([0.1], [1], [0], [Death], 1, 1.0)
        @test forest_from_eventlog(extinct_unsampled; tj=0.0, tk=1.0) == Tree[]
        @test isempty(tree_from_eventlog(extinct_unsampled; tj=0.0, tk=1.0))

        rng = MersenneTwister(20260417)
        sim_pars = ConstantRateBDParameters(1.4, 0.35, 0.8, 0.55, 0.25)
        for _ in 1:200
            log = simulate_bd(rng, sim_pars, 1.5; initial_lineages=2)
            for (tj, tk) in ((0.0, 1.5), (0.4, 1.2), (0.9, 1.5))
                forest = forest_from_eventlog(log; tj, tk)
                @test length(forest) == A_at(log, tj, tk)
                @test sort([tree.host[TreeSim.root(tree)] for tree in forest]) == retained_lineages_at(log, tj, tk)
                @test all(tree -> validate_tree(tree; require_single_root=true, require_reachable=true), forest)
                if length(forest) > 1
                    @test_throws ErrorException tree_from_eventlog(log; tj, tk)
                end
            end
        end

        @test_throws ArgumentError forest_from_eventlog(simple; tj=-0.1, tk=1.0)
        @test_throws ArgumentError forest_from_eventlog(simple; tj=0.8, tk=0.7)
        @test_throws ArgumentError forest_from_eventlog(simple; tj=0.0, tk=1.1)
    end

    @testset "TreeSim full sampled-ancestry extraction from BDEventLog" begin
        same_tree(a, b) = a.time == b.time &&
                          a.left == b.left &&
                          a.right == b.right &&
                          a.parent == b.parent &&
                          a.kind == b.kind &&
                          a.host == b.host &&
                          a.label == b.label
        same_forest(a, b) = length(a) == length(b) && all(same_tree(x, y) for (x, y) in zip(a, b))

        unary_birth = BDEventLog([0.2, 0.7], [2, 2], [1, 0], [Birth, SerialSampling], 1, 1.0)
        full = full_forest_from_eventlog(unary_birth; tj=0.0, tk=1.0)
        reconstructed = forest_from_eventlog(unary_birth; tj=0.0, tk=1.0)
        @test length(full) == length(reconstructed) == A_at(unary_birth, 0.0, 1.0) == 1
        @test validate_tree(only(full); require_single_root=true, require_reachable=true)
        @test only(full).kind == [Root, UnsampledUnary, SampledLeaf]
        @test only(full).host == [1, 1, 2]
        @test only(reconstructed).kind == [Root, SampledLeaf]
        @test same_forest([BDUtils._bd_collapse_unsampled_unary(tree) for tree in full], reconstructed)

        full_single = full_tree_from_eventlog(unary_birth; tj=0.0, tk=1.0)
        @test same_tree(full_single, only(full))

        sampled_parent_and_child = BDEventLog([0.2, 0.5, 0.7], [2, 1, 2], [1, 0, 0], [Birth, FossilizedSampling, SerialSampling], 1, 1.0)
        full_binary = full_forest_from_eventlog(sampled_parent_and_child; tj=0.0, tk=1.0)
        @test same_forest(full_binary, forest_from_eventlog(sampled_parent_and_child; tj=0.0, tk=1.0))
        @test only(full_binary).kind == [Root, Binary, SampledLeaf, SampledLeaf]

        after_birth = full_forest_from_eventlog(sampled_parent_and_child; tj=0.2, tk=1.0)
        @test length(after_birth) == A_at(sampled_parent_and_child, 0.2, 1.0) == 2
        @test all(tree -> validate_tree(tree; require_single_root=true, require_reachable=true), after_birth)
        @test_throws ErrorException full_tree_from_eventlog(sampled_parent_and_child; tj=0.2, tk=1.0)

        sample_at_tj = BDEventLog([0.5], [1], [0], [FossilizedSampling], 1, 1.0)
        @test full_forest_from_eventlog(sample_at_tj; tj=0.5, tk=1.0) == Tree[]
        @test length(full_forest_from_eventlog(sample_at_tj; tj=0.49, tk=0.5)) == 1

        sample_at_tk = BDEventLog([0.5], [1], [0], [SerialSampling], 1, 1.0)
        @test length(full_forest_from_eventlog(sample_at_tk; tj=0.49, tk=0.5)) == 1
        @test full_forest_from_eventlog(sample_at_tk; tj=0.5, tk=0.5) == Tree[]

        rng = MersenneTwister(20260418)
        sim_pars = ConstantRateBDParameters(1.6, 0.4, 0.9, 0.5, 0.2)
        for _ in 1:200
            log = simulate_bd(rng, sim_pars, 1.5; initial_lineages=2)
            for (tj, tk) in ((0.0, 1.5), (0.3, 1.1), (0.8, 1.5), (1.0, 1.0))
                full = full_forest_from_eventlog(log; tj, tk)
                reconstructed = forest_from_eventlog(log; tj, tk)
                @test length(full) == length(reconstructed) == A_at(log, tj, tk)
                @test sort([tree.host[TreeSim.root(tree)] for tree in full]) == retained_lineages_at(log, tj, tk)
                @test all(tree -> validate_tree(tree; require_single_root=true, require_reachable=true), full)
                @test same_forest([BDUtils._bd_collapse_unsampled_unary(tree) for tree in full], reconstructed)
                if length(full) > 1
                    @test_throws ErrorException full_tree_from_eventlog(log; tj, tk)
                end
            end
        end

        @test_throws ArgumentError full_forest_from_eventlog(unary_birth; tj=-0.1, tk=1.0)
        @test_throws ArgumentError full_forest_from_eventlog(unary_birth; tj=0.8, tk=0.7)
        @test_throws ArgumentError full_forest_from_eventlog(unary_birth; tj=0.0, tk=1.1)
    end

    # Core analytical invariants: scalar closed forms, coefficients, PMFs,
    # marginals, and tails. These are the first line of defense for refactors.
    @testset "core invariant: PGF/probability helpers" begin
        a, b, Δ = bd_coefficients(1.0, λ, μ, ψ, r)
        a_struct, b_struct, Δ_struct = bd_coefficients(1.0, pars)
        @test isfinite(a)
        @test isfinite(b)
        @test isfinite(Δ)
        @test Δ >= 0
        @test (a_struct, b_struct, Δ_struct) == (a, b, Δ)

        @test γ(1.0, 0.0, 0.0, λ, μ, ψ, r) == 0.0
        @test α(1.0, 0.0, 0.0, λ, μ, ψ, r) == 0.0
        @test β(1.0, 0.0, 0.0, λ, μ, ψ, r) == 1.0
        @test pₙ(0, 0.0, 0.0, λ, μ, ψ, r) == 0.0
        @test pₙ(1, 0.0, 0.0, λ, μ, ψ, r) == 1.0
        @test pₙ(2, 0.0, 0.0, λ, μ, ψ, r) == 0.0
        @test γ(1.0, 0.0, 0.0, pars) == 0.0
        @test α(1.0, 0.0, 0.0, pars) == 0.0
        @test β(1.0, 0.0, 0.0, pars) == 1.0
        @test pₙ(1, 0.0, 0.0, pars) == 1.0

        vals = pn_birthdeath([0, 1, 2], 0.0, 1.0, λ, μ, ψ, r)
        vals_struct = pn_birthdeath([0, 1, 2], 0.0, 1.0, pars)
        @test length(vals) == 3
        @test all(isfinite, vals)
        @test all(>=(0), vals)
        @test vals_struct == vals

        @test gamma_bd(1, 0, 1, λ, μ, ψ, r) ≈ γ(1.0, 0.0, 1.0, λ, μ, ψ, r)
        @test alpha_bd(1, 0, 1, λ, μ, ψ, r) ≈ alpha_bd(1, 0, 1, ConstantRateBDParameters(λ, μ, ψ, r))
        @test beta_bd(1, 0, 1, λ, μ, ψ, r) ≈ beta_bd(1, 0, 1, ConstantRateBDParameters(λ, μ, ψ, r))
        @test pn_birthdeath(2, 0, 1, λ, μ, ψ, r) ≈ pn_birthdeath(2, 0, 1, ConstantRateBDParameters(λ, μ, ψ, r))

        @test_throws ArgumentError bd_coefficients(1.0, 0.0, μ, ψ, r)
        @test_throws ArgumentError bd_coefficients(1.0, λ, -μ, ψ, r)
        @test_throws ArgumentError bd_coefficients(1.0, λ, μ, ψ, 1.1)
        @test_throws ArgumentError bd_coefficients(Inf, λ, μ, ψ, r)
    end

    @testset "core invariant: joint NS series, marginals, and tails" begin
        ti = 0.0
        tj = 0.75
        smax = 14
        αs, βs, γs = constant_rate_pgf_series(smax, ti, tj, pars)

        w = 0.2
        powers = w .^ (0:smax)
        @test sum(αs .* powers) ≈ alpha_bd(w, ti, tj, pars) atol=1e-11
        @test sum(βs .* powers) ≈ beta_bd(w, ti, tj, pars) atol=1e-11
        @test sum(γs .* powers) ≈ gamma_bd(w, ti, tj, pars) atol=1e-11

        @test joint_pmf_NS(0, 2, ti, tj, pars) ≈ αs[3]
        @test joint_pmf_NS(1, 2, ti, tj, pars) ≈ βs[3]

        table = joint_pmf_NS_table(8, 6, ti, tj, pars)
        @test size(table) == (9, 7)
        @test table[1, 3] ≈ joint_pmf_NS(0, 2, ti, tj, pars)
        @test table[2, 3] ≈ joint_pmf_NS(1, 2, ti, tj, pars)
        @test table[5, 4] ≈ joint_pmf_NS(4, 3, ti, tj, pars)
        @test all(x -> x >= -1e-14, table)

        @test n_marginal_pmf(0, ti, tj, pars) ≈ alpha_bd(1.0, ti, tj, pars)
        @test n_marginal_pmf(4, ti, tj, pars) ≈ pn_birthdeath(4, ti, tj, pars)

        ncut = n_truncation(ti, tj, pars; atol=1e-10)
        @test n_marginal_tail(ncut, ti, tj, pars) <= 1e-10
        @test sum(n_marginal_pmf(n, ti, tj, pars) for n in 0:ncut) + n_marginal_tail(ncut, ti, tj, pars) ≈ 1.0

        @test s_marginal_pmf(0, ti, tj, pars) ≈ alpha_bd(0.0, ti, tj, pars) + beta_bd(0.0, ti, tj, pars) / (1 - gamma_bd(0.0, ti, tj, pars))
        @test s_marginal_pmf(3, ti, tj, pars) ≈ sum(joint_pmf_NS(n, 3, ti, tj, pars) for n in 0:120) atol=1e-12

        @test s_marginal_tail(0, ti, tj, pars) ≈ 1 - s_marginal_pmf(0, ti, tj, pars)
        @test s_marginal_tail(5, ti, tj, pars) ≈ 1 - sum(s_marginal_pmf(s, ti, tj, pars) for s in 0:5) atol=1e-12
        @test s_marginal_tail(6, ti, tj, pars) <= s_marginal_tail(5, ti, tj, pars)

        scut = s_truncation(ti, tj, pars; atol=1e-9)
        @test s_marginal_tail(scut, ti, tj, pars) <= 1e-9
        if scut > 0
            @test s_marginal_tail(scut - 1, ti, tj, pars) > 1e-9
        end

        no_sampling = ConstantRateBDParameters(λ, μ, 0.0, r)
        @test s_marginal_pmf(0, ti, tj, no_sampling) ≈ 1.0
        @test s_marginal_tail(0, ti, tj, no_sampling) == 0.0
        @test s_truncation(ti, tj, no_sampling; atol=0.0) == 0

        diagnostic = joint_pmf_NS_table(8, 6, ti, tj, pars; diagnostics=true)
        @test diagnostic.table == table
        @test diagnostic.nmax == 8
        @test diagnostic.smax == 6
        @test diagnostic.retained_mass ≈ sum(table)
        @test diagnostic.n_tail_mass ≈ n_marginal_tail(8, ti, tj, pars)
        @test diagnostic.s_tail_mass ≈ s_marginal_tail(6, ti, tj, pars)
        @test diagnostic.missing_mass ≈ 1 - diagnostic.retained_mass
        @test diagnostic.n_only_tail_mass + diagnostic.s_only_tail_mass + diagnostic.joint_tail_overlap_mass + diagnostic.retained_mass ≈ 1.0 atol=1e-12
        @test diagnostic.n_only_tail_mass + diagnostic.joint_tail_overlap_mass ≈ diagnostic.n_tail_mass atol=1e-12
        @test diagnostic.s_only_tail_mass + diagnostic.joint_tail_overlap_mass ≈ diagnostic.s_tail_mass atol=1e-12

        @test_throws ArgumentError constant_rate_pgf_series(-1, ti, tj, pars)
        @test_throws ArgumentError constant_rate_pgf_series(2, tj, ti, pars)
        @test_throws ArgumentError joint_pmf_NS(-1, 0, ti, tj, pars)
        @test_throws ArgumentError joint_pmf_NS(0, -1, ti, tj, pars)
        @test_throws ArgumentError s_marginal_tail(-1, ti, tj, pars)
        @test_throws ArgumentError s_truncation(ti, tj, pars; atol=-1.0)
        @test_throws ArgumentError s_truncation(ti, tj, pars; atol=NaN)
        @test_throws ArgumentError s_truncation(ti, tj, pars; atol=1e-14, max_smax=0)
        @test_throws ArgumentError n_truncation(ti, tj, pars; atol=-1.0)
    end

    @testset "simulation validation: original-process NS distribution" begin
        cases = (
            (name="subcritical_low_sampling", seed=11, pars=ConstantRateBDParameters(0.7, 1.0, 0.12, 0.35), tj=1.1, nsims=8_000),
            (name="near_critical_r_zero", seed=12, pars=ConstantRateBDParameters(1.0, 0.92, 0.35, 0.0), tj=0.9, nsims=8_000),
            (name="supercritical_high_sampling", seed=13, pars=ConstantRateBDParameters(1.45, 0.55, 0.9, 0.9), tj=0.8, nsims=8_000),
            (name="no_sampling_supported", seed=14, pars=ConstantRateBDParameters(1.25, 0.8, 0.0, 0.0), tj=0.9, nsims=8_000),
        )

        for case in cases
            summary = original_process_validation_summary(case.seed, case.pars, case.tj, case.nsims; tail_atol=1e-4)
            assert_original_process_validation(summary;
                joint_tv_atol=0.025,
                marginal_tv_atol=0.022,
                maxerr_atol=0.018,
                tail_slack=0.018,
            )
        end
    end

    @testset "simulation validation: original-process multi-time queries" begin
        multi_pars = ConstantRateBDParameters(1.1, 0.75, 0.35, 0.65)
        times = [0.35, 0.75, 1.1]
        logs = simulate_original_process(21, multi_pars, maximum(times), 7_000)

        for (i, tj) in pairs(times)
            summary = original_process_validation_summary(30 + i, multi_pars, tj, 7_000; tail_atol=1e-4)
            empirical_from_shared_logs = joint_pmf_NS(joint_counts_NS(logs, tj))
            @test total_variation_on_support(empirical_from_shared_logs, summary.analytical_joint, summary.support) <= 0.03
            @test abs(sum(get(empirical_from_shared_logs, key, 0.0) for key in summary.support) -
                      summary.diagnostic.retained_mass) <= 0.02
            assert_original_process_validation(summary;
                joint_tv_atol=0.03,
                marginal_tv_atol=0.026,
                maxerr_atol=0.02,
                tail_slack=0.02,
            )
        end

        count_series = joint_counts_NS(logs, times)
        @test length(count_series) == length(times)
        @test all(counts -> sum(values(counts)) == length(logs), count_series)
    end

    @testset "simulation validation: reconstructed process A and AS distribution" begin
        cases = (
            (name="subcritical_mixed_sampling", seed=41, pars=ConstantRateBDParameters(0.75, 1.0, 0.35, 0.35), tj=0.55, tk=1.3, nsims=10_000),
            (name="near_critical_r_zero", seed=42, pars=ConstantRateBDParameters(1.05, 0.9, 0.55, 0.0), tj=0.5, tk=1.2, nsims=10_000),
            (name="supercritical_high_sampling", seed=43, pars=ConstantRateBDParameters(1.45, 0.55, 0.9, 0.85), tj=0.45, tk=1.0, nsims=10_000),
        )

        for case in cases
            summary = reconstructed_validation_summary(case.seed, case.pars, case.tj, case.tk, case.nsims; tail_atol=1e-4)
            assert_reconstructed_validation(summary;
                joint_tv_atol=0.03,
                marginal_tv_atol=0.026,
                maxerr_atol=0.022,
                tail_slack=0.02,
                retention_atol=0.025,
            )
        end
    end

    @testset "simulation validation: reconstructed multi-time queries" begin
        reconstructed_pars = ConstantRateBDParameters(1.2, 0.7, 0.45, 0.6)
        tk = 1.4
        times = [0.3, 0.7, 1.1]
        rng = MersenneTwister(51)
        logs = [simulate_bd(rng, reconstructed_pars, tk; apply_ρ₀=false) for _ in 1:8_000]

        count_series = reconstructed_counts_A(logs, times)
        joint_series = reconstructed_joint_counts_AS(logs, times)
        @test length(count_series) == length(times)
        @test length(joint_series) == length(times)
        @test all(counts -> sum(values(counts)) == length(logs), count_series)
        @test all(counts -> sum(values(counts)) == length(logs), joint_series)

        for (i, tj) in pairs(times)
            summary = reconstructed_validation_summary(60 + i, reconstructed_pars, tj, tk, 8_000; tail_atol=1e-4)
            empirical_from_shared_logs = reconstructed_joint_pmf_AS(reconstructed_joint_counts_AS(logs, tj))
            @test total_variation_on_support(empirical_from_shared_logs, summary.analytical_joint, summary.support) <= 0.035
            @test abs(sum(get(empirical_from_shared_logs, key, 0.0) for key in summary.support) -
                      summary.diagnostic.retained_mass) <= 0.025
            assert_reconstructed_validation(summary;
                joint_tv_atol=0.035,
                marginal_tv_atol=0.03,
                maxerr_atol=0.025,
                tail_slack=0.025,
                retention_atol=0.03,
            )
        end
    end

    if get(ENV, "BDUTILS_STRESS_TESTS", "false") == "true"
        @testset "stress: simulation validation original-process NS distribution" begin
            stress_cases = (
                (name="subcritical_high_removal", seed=101, pars=ConstantRateBDParameters(0.65, 1.15, 0.7, 0.98), tj=1.5, nsims=30_000),
                (name="near_critical_low_sampling", seed=102, pars=ConstantRateBDParameters(1.02, 0.98, 0.04, 0.4), tj=1.4, nsims=30_000),
                (name="supercritical_r_near_one", seed=103, pars=ConstantRateBDParameters(1.6, 0.4, 0.55, 0.99), tj=1.0, nsims=30_000),
            )

            for case in stress_cases
                summary = original_process_validation_summary(case.seed, case.pars, case.tj, case.nsims; tail_atol=5e-5, max_smax=2_000)
                assert_original_process_validation(summary;
                    joint_tv_atol=0.018,
                    marginal_tv_atol=0.015,
                    maxerr_atol=0.011,
                    tail_slack=0.011,
                )
            end
        end
    end

    # Extended analytical stress checks: same invariants under numerically
    # awkward but valid parameter/time regimes.
    @testset "stress: constant-rate NS numerical regimes" begin
        for regime in STRESS_REGIMES
            name, stress_pars, ti, tj = regime.name, regime.pars, regime.ti, regime.tj
            nmax = n_truncation(ti, tj, stress_pars; atol=1e-8)
            smax = s_truncation(ti, tj, stress_pars; atol=1e-8, max_smax=2_000)
            diagnostic = joint_pmf_NS_table(nmax, smax, ti, tj, stress_pars; diagnostics=true)
            table = diagnostic.table

            @test all(isfinite, table)
            @test minimum(table) >= -1e-10
            @test 0.0 <= diagnostic.retained_mass <= 1.0 + 1e-9
            @test diagnostic.n_tail_mass <= 1e-8 || name == "small_t"
            @test diagnostic.s_tail_mass <= 1e-8 || stress_pars.ψ == 0.0
            @test diagnostic.missing_mass ≈ 1 - diagnostic.retained_mass atol=1e-11
            @test diagnostic.n_only_tail_mass + diagnostic.s_only_tail_mass + diagnostic.joint_tail_overlap_mass + diagnostic.retained_mass ≈ 1.0 atol=2e-8

            for n in 0:min(nmax, 6), s in 0:min(smax, 5)
                @test table[n + 1, s + 1] ≈ joint_pmf_NS(n, s, ti, tj, stress_pars) atol=1e-12 rtol=1e-9
            end

            retained_n = vec(sum(table; dims=2))
            for n in 0:min(nmax, 8)
                @test retained_n[n + 1] <= n_marginal_pmf(n, ti, tj, stress_pars) + max(1e-10, diagnostic.s_tail_mass + 1e-10)
            end

            z = 0.45
            w = 0.35
            approx_pgf = table_pgf_sum(table, z, w)
            scalar_pgf = scalar_joint_pgf(z, w, ti, tj, stress_pars)
            @test isfinite(scalar_pgf)
            @test abs(approx_pgf - scalar_pgf) <= diagnostic.n_tail_mass + diagnostic.s_tail_mass + 5e-8
        end
    end

    @testset "core invariant: NS derivative identities" begin
        ti = 0.0
        tj = 1.1
        derivative_pars = ConstantRateBDParameters(1.7, 0.6, 0.5, 0.65)
        αs, βs, γs = constant_rate_pgf_series(4, ti, tj, derivative_pars)

        for (f, coeffs) in ((w -> alpha_bd(w, ti, tj, derivative_pars), αs),
                            (w -> beta_bd(w, ti, tj, derivative_pars), βs),
                            (w -> gamma_bd(w, ti, tj, derivative_pars), γs))
            @test central_first_derivative(f, 0.0; h=1e-5) ≈ coeffs[2] rtol=2e-5 atol=2e-7
            @test central_second_derivative(f, 0.0; h=2e-4) / 2 ≈ coeffs[3] rtol=2e-4 atol=2e-6
        end

        z = 0.4
        g_coeffs = vec(sum(joint_pmf_NS_table(12, 4, ti, tj, derivative_pars) .* (z .^ (0:12)), dims=1))
        gz = w -> scalar_joint_pgf(z, w, ti, tj, derivative_pars)
        @test central_first_derivative(gz, 0.0; h=1e-5) ≈ g_coeffs[2] rtol=2e-5 atol=2e-7
        @test central_second_derivative(gz, 0.0; h=2e-4) / 2 ≈ g_coeffs[3] rtol=2e-4 atol=2e-6

        γ1 = gamma_bd(1.0, ti, tj, derivative_pars)
        β1 = beta_bd(1.0, ti, tj, derivative_pars)
        expected_n = β1 / (1 - γ1)^2
        expected_n2factorial = 2β1 * γ1 / (1 - γ1)^3
        ncut = n_truncation(ti, tj, derivative_pars; atol=1e-12)
        @test sum(n * n_marginal_pmf(n, ti, tj, derivative_pars) for n in 0:ncut) + β1 * (γ1^ncut) * (ncut + 1 - ncut * γ1) / (1 - γ1)^2 ≈ expected_n rtol=1e-11
        @test central_first_derivative(zval -> scalar_joint_pgf(zval, 1.0, ti, tj, derivative_pars), 1.0; h=1e-5) ≈ expected_n rtol=1e-8
        @test central_second_derivative(zval -> scalar_joint_pgf(zval, 1.0, ti, tj, derivative_pars), 1.0; h=2e-4) ≈ expected_n2factorial rtol=2e-5

        scut = s_truncation(ti, tj, derivative_pars; atol=1e-12, max_smax=2_000)
        expected_s = sum(s * s_marginal_pmf(s, ti, tj, derivative_pars) for s in 0:scut)
        expected_s2factorial = sum(s * (s - 1) * s_marginal_pmf(s, ti, tj, derivative_pars) for s in 0:scut)
        s_tail = s_marginal_tail(scut, ti, tj, derivative_pars)
        gw = w -> scalar_joint_pgf(1.0, w, ti, tj, derivative_pars)
        @test central_first_derivative(gw, 1.0; h=1e-5) ≈ expected_s rtol=5e-6 atol=max(1e-9, 10s_tail)
        @test central_second_derivative(gw, 1.0; h=2e-4) ≈ expected_s2factorial rtol=2e-4 atol=max(1e-8, 100s_tail)

        table = joint_pmf_NS_table(n_truncation(ti, tj, derivative_pars; atol=1e-11),
                                   s_truncation(ti, tj, derivative_pars; atol=1e-11, max_smax=2_000),
                                   ti, tj, derivative_pars)
        z_inside = 0.55
        w_inside = 0.45
        @test table_pgf_sum(table, z_inside, w_inside) ≈ scalar_joint_pgf(z_inside, w_inside, ti, tj, derivative_pars) atol=3e-10
    end

    # Core ODE invariants: the closed forms must satisfy their defining
    # forward and backward triangular systems, plus scalar PGF equations.
    @testset "core invariant: forward triangular ODE residuals" begin
        ti = 0.2

        for ode_pars in ODE_REGIMES
            for w in FORWARD_W_VALUES
                @test alpha_bd(w, ti, ti, ode_pars) ≈ 0.0 atol=1e-14
                @test beta_bd(w, ti, ti, ode_pars) ≈ 1.0 atol=1e-14
                @test gamma_bd(w, ti, ti, ode_pars) ≈ 0.0 atol=1e-14

                a, b, _ = bd_coefficients(w, ode_pars)
                for tj in FORWARD_TJ_VALUES
                    γij = gamma_bd(w, ti, tj, ode_pars)
                    βij = beta_bd(w, ti, tj, ode_pars)
                    dγ = central_first_derivative(x -> gamma_bd(w, ti, x, ode_pars), tj; h=2e-5)
                    dβ = central_first_derivative(x -> beta_bd(w, ti, x, ode_pars), tj; h=2e-5)
                    dα = central_first_derivative(x -> alpha_bd(w, ti, x, ode_pars), tj; h=2e-5)

                    rhs_γ = (1 - γij) * (ode_pars.λ - a * γij) - ode_pars.ψ * (1 - w) * γij
                    rhs_β = (2a * γij + b) * βij
                    rhs_α = a * βij

                    @test dγ ≈ rhs_γ rtol=2e-7 atol=2e-8
                    @test dβ ≈ rhs_β rtol=2e-6 atol=2e-8
                    @test dα ≈ rhs_α rtol=2e-6 atol=2e-8
                end
            end
        end
    end

    @testset "core invariant: backward triangular ODE residuals" begin
        for backward_pars in ODE_REGIMES
            for w in BACKWARD_W_VALUES
                for t in (0.0, 0.8, 1.6)
                    @test alpha_bd(w, t, t, backward_pars) ≈ 0.0 atol=1e-14
                    @test beta_bd(w, t, t, backward_pars) ≈ 1.0 atol=1e-14
                    @test gamma_bd(w, t, t, backward_pars) ≈ 0.0 atol=1e-14
                end

                a = bd_a(w, backward_pars)
                b = bd_b_backward(w, backward_pars)
                for (ti, tj) in BACKWARD_T_PAIRS
                    αij = alpha_bd(w, ti, tj, backward_pars)
                    βij = beta_bd(w, ti, tj, backward_pars)
                    γij = gamma_bd(w, ti, tj, backward_pars)
                    dα = central_first_derivative(x -> alpha_bd(w, x, tj, backward_pars), ti; h=2e-5)
                    dβ = central_first_derivative(x -> beta_bd(w, x, tj, backward_pars), ti; h=2e-5)
                    dγ = central_first_derivative(x -> gamma_bd(w, x, tj, backward_pars), ti; h=2e-5)

                    rhs_α_quadratic = backward_pars.λ * αij^2 + b * αij + a
                    rhs_α_branching = (1 - αij) * (a - backward_pars.λ * αij) - backward_pars.ψ * (1 - w) * αij
                    rhs_β = (2backward_pars.λ * αij + b) * βij
                    rhs_γ = backward_pars.λ * βij

                    @test rhs_α_quadratic ≈ rhs_α_branching rtol=2e-13 atol=2e-14
                    @test -dα ≈ rhs_α_quadratic rtol=2e-6 atol=2e-8
                    @test -dβ ≈ rhs_β rtol=2e-6 atol=2e-8
                    @test -dγ ≈ rhs_γ rtol=2e-7 atol=2e-8
                end
            end
        end
    end

    @testset "core invariant: backward Kolmogorov PGF residuals" begin
        z_values = (0.0, 0.35, 0.8)
        w_values = (0.0, 0.4, 1.0)

        for backward_pars in ODE_REGIMES
            for z in z_values, w in w_values, (ti, tj) in PGF_T_PAIRS
                F = scalar_joint_pgf(z, w, ti, tj, backward_pars)
                dF = central_first_derivative(x -> scalar_joint_pgf(z, w, x, tj, backward_pars), ti; h=2e-5)
                rhs = backward_generator(F, w, backward_pars)
                @test -dF ≈ rhs rtol=4e-6 atol=3e-8
            end
        end
    end

    @testset "core invariant: backward Kendall specialization" begin
        for backward_pars in KENDALL_REGIMES
            δ = backward_pars.μ + backward_pars.r * backward_pars.ψ
            for (ti, tj) in BACKWARD_T_PAIRS
                ξ = alpha_bd(1.0, ti, tj, backward_pars)
                η = gamma_bd(1.0, ti, tj, backward_pars)
                β1 = beta_bd(1.0, ti, tj, backward_pars)
                dξ = central_first_derivative(x -> alpha_bd(1.0, x, tj, backward_pars), ti; h=2e-5)
                dη = central_first_derivative(x -> gamma_bd(1.0, x, tj, backward_pars), ti; h=2e-5)
                dβ = central_first_derivative(x -> beta_bd(1.0, x, tj, backward_pars), ti; h=2e-5)

                @test β1 ≈ (1 - ξ) * (1 - η) rtol=2e-12 atol=2e-14
                @test -dξ ≈ backward_pars.λ * ξ^2 - (backward_pars.λ + δ) * ξ + δ rtol=2e-6 atol=2e-8
                @test -dη ≈ backward_pars.λ * β1 rtol=2e-7 atol=2e-8
                @test -dβ ≈ (2backward_pars.λ * ξ - (backward_pars.λ + δ)) * β1 rtol=2e-6 atol=2e-8
            end
        end
    end

    @testset "core invariant: forward Kendall identities at w = 1" begin
        ti = 0.0

        for kendall_pars in KENDALL_REGIMES
            @test alpha_bd(1.0, ti, ti, kendall_pars) ≈ 0.0 atol=1e-14
            @test gamma_bd(1.0, ti, ti, kendall_pars) ≈ 0.0 atol=1e-14
            @test beta_bd(1.0, ti, ti, kendall_pars) ≈ 1.0 atol=1e-14

            δ = kendall_pars.μ + kendall_pars.r * kendall_pars.ψ
            for tj in (0.35, 1.0, 1.7)
                ξ = alpha_bd(1.0, ti, tj, kendall_pars)
                η = gamma_bd(1.0, ti, tj, kendall_pars)
                β1 = beta_bd(1.0, ti, tj, kendall_pars)
                @test β1 ≈ (1 - ξ) * (1 - η) rtol=2e-12 atol=2e-14

                dη = central_first_derivative(x -> gamma_bd(1.0, ti, x, kendall_pars), tj; h=2e-5)
                dξ = central_first_derivative(x -> alpha_bd(1.0, ti, x, kendall_pars), tj; h=2e-5)
                @test dη ≈ (kendall_pars.λ - δ * η) * (1 - η) rtol=2e-7 atol=2e-8
                @test dξ ≈ δ * (1 - ξ) * (1 - η) rtol=2e-6 atol=2e-8
            end
        end
    end

    @testset "core invariant: Riccati residual for p" begin
        for riccati_pars in ODE_REGIMES
            tk = 2.5
            @test p_unsampled(tk, tk, riccati_pars) ≈ 1.0 atol=1e-14
            for tj in (0.2, 1.0, 1.9)
                p = p_unsampled(tj, tk, riccati_pars)
                dp = central_first_derivative(x -> p_unsampled(x, tk, riccati_pars), tj; h=2e-5)
                rhs = riccati_pars.μ - (riccati_pars.λ + riccati_pars.μ + riccati_pars.ψ) * p + riccati_pars.λ * p^2
                @test -dp ≈ rhs rtol=2e-6 atol=2e-8
            end
        end
    end

    @testset "public API: reconstructed scalar helpers" begin
        ti = 0.0
        tk = 2.4

        for reconstructed_pars in KENDALL_REGIMES
            for w in (0.0, 0.4, 1.0), tj in (0.35, 1.0, 1.7)
                p = unsampled_probability(tj, tk, reconstructed_pars)
                expected = transformed_alpha_beta_gamma(w, ti, tj, tk, reconstructed_pars)

                @test p ≈ p_unsampled(tj, tk, reconstructed_pars)
                @test transformed_birth_rate(tj, tk, reconstructed_pars) ≈ reconstructed_pars.λ * (1 - p)
                @test transformed_death_rate(tj, tk, reconstructed_pars) ≈ reconstructed_pars.ψ * (reconstructed_pars.r + (1 - reconstructed_pars.r) * p) / (1 - p)
                @test transformed_sampling_rate(tj, tk, reconstructed_pars) ≈ reconstructed_pars.ψ / (1 - p)

                @test reconstructed_alpha_bd(w, ti, tj, tk, reconstructed_pars) ≈ expected.α
                @test reconstructed_beta_bd(w, ti, tj, tk, reconstructed_pars) ≈ expected.β
                @test reconstructed_gamma_bd(w, ti, tj, tk, reconstructed_pars) ≈ expected.γ

                z = 0.55
                @test reconstructed_pgf(z, w, ti, tj, tk, reconstructed_pars) ≈ expected.α + expected.β * z / (1 - expected.γ * z)
            end

            for tj in (0.35, 1.0, 1.7)
                ξ = reconstructed_xi(ti, tj, tk, reconstructed_pars)
                η = reconstructed_eta(ti, tj, tk, reconstructed_pars)
                β1 = reconstructed_beta_bd(1.0, ti, tj, tk, reconstructed_pars)
                @test β1 ≈ (1 - ξ) * (1 - η) rtol=2e-11 atol=2e-13
                @test reconstructed_count_pmf(0, ti, tj, tk, reconstructed_pars) ≈ ξ
                @test reconstructed_count_pmf(1, ti, tj, tk, reconstructed_pars) ≈ (1 - ξ) * (1 - η)
                @test reconstructed_count_pmf(4, ti, tj, tk, reconstructed_pars) ≈ (1 - ξ) * (1 - η) * η^3
                @test sum(reconstructed_count_pmf(a, ti, tj, tk, reconstructed_pars) for a in 0:200) ≈ 1.0 atol=1e-12
            end
        end

        @test unsampled_probability(1.0, 1.0, pars) ≈ 1.0
        @test_throws ArgumentError unsampled_probability(2.0, 1.0, pars)
        @test_throws ArgumentError reconstructed_alpha_bd(1.0, 1.1, 1.0, 2.0, pars)
        @test_throws ArgumentError reconstructed_count_pmf(-1, 0.0, 1.0, 2.0, pars)
        @test_throws ArgumentError transformed_birth_rate(0.0, 1.0, ConstantRateBDParameters(1.0, 0.5, 0.0, 0.0))
    end

    @testset "public API: reconstructed series, PMF, marginals, and truncation" begin
        ti = 0.0
        tj = 0.85
        tk = 2.4
        reconstructed_pars = ConstantRateBDParameters(1.8, 0.5, 0.7, 0.4)
        smax = 10
        αs, βs, γs = reconstructed_pgf_series(smax, ti, tj, tk, reconstructed_pars)

        w = 0.25
        powers = w .^ (0:smax)
        @test sum(αs .* powers) ≈ reconstructed_alpha_bd(w, ti, tj, tk, reconstructed_pars) atol=1e-11
        @test sum(βs .* powers) ≈ reconstructed_beta_bd(w, ti, tj, tk, reconstructed_pars) atol=1e-11
        @test sum(γs .* powers) ≈ reconstructed_gamma_bd(w, ti, tj, tk, reconstructed_pars) atol=1e-11

        table = reconstructed_joint_pmf_table(9, 7, ti, tj, tk, reconstructed_pars)
        @test size(table) == (10, 8)
        @test table[1, 3] ≈ reconstructed_joint_pmf(0, 2, ti, tj, tk, reconstructed_pars)
        @test table[2, 3] ≈ reconstructed_joint_pmf(1, 2, ti, tj, tk, reconstructed_pars)
        @test table[5, 4] ≈ reconstructed_joint_pmf(4, 3, ti, tj, tk, reconstructed_pars)
        @test all(x -> x >= -1e-13, table)

        z = 0.45
        w_inside = 0.35
        count_cut = reconstructed_count_truncation(ti, tj, tk, reconstructed_pars; atol=1e-11)
        sampling_cut = reconstructed_sampling_truncation(ti, tj, tk, reconstructed_pars; atol=1e-11, max_smax=2_000)
        pgf_table = reconstructed_joint_pmf_table(count_cut, sampling_cut, ti, tj, tk, reconstructed_pars)
        @test table_pgf_sum(pgf_table, z, w_inside) ≈ reconstructed_pgf(z, w_inside, ti, tj, tk, reconstructed_pars) atol=3e-10

        @test reconstructed_count_pmf(0, ti, tj, tk, reconstructed_pars) ≈ reconstructed_xi(ti, tj, tk, reconstructed_pars)
        @test reconstructed_count_pmf(4, ti, tj, tk, reconstructed_pars) ≈ sum(reconstructed_joint_pmf(4, s, ti, tj, tk, reconstructed_pars) for s in 0:120) atol=1e-12
        @test reconstructed_sampling_marginal_pmf(3, ti, tj, tk, reconstructed_pars) ≈ sum(reconstructed_joint_pmf(a, 3, ti, tj, tk, reconstructed_pars) for a in 0:120) atol=1e-12

        @test reconstructed_count_tail(count_cut, ti, tj, tk, reconstructed_pars) <= 1e-11
        if count_cut > 0
            @test reconstructed_count_tail(count_cut - 1, ti, tj, tk, reconstructed_pars) > 1e-11
        end
        @test sum(reconstructed_count_pmf(a, ti, tj, tk, reconstructed_pars) for a in 0:count_cut) + reconstructed_count_tail(count_cut, ti, tj, tk, reconstructed_pars) ≈ 1.0

        @test reconstructed_sampling_tail(5, ti, tj, tk, reconstructed_pars) ≈ 1 - sum(reconstructed_sampling_marginal_pmf(s, ti, tj, tk, reconstructed_pars) for s in 0:5) atol=1e-12
        @test reconstructed_sampling_tail(sampling_cut, ti, tj, tk, reconstructed_pars) <= 1e-11
        if sampling_cut > 0
            @test reconstructed_sampling_tail(sampling_cut - 1, ti, tj, tk, reconstructed_pars) > 1e-11
        end

        diagnostic = reconstructed_joint_pmf_table(9, 7, ti, tj, tk, reconstructed_pars; diagnostics=true)
        @test diagnostic.table == table
        @test diagnostic.amax == 9
        @test diagnostic.smax == 7
        @test diagnostic.retained_mass ≈ sum(table)
        @test diagnostic.count_tail_mass ≈ reconstructed_count_tail(9, ti, tj, tk, reconstructed_pars)
        @test diagnostic.sampling_tail_mass ≈ reconstructed_sampling_tail(7, ti, tj, tk, reconstructed_pars)
        @test diagnostic.missing_mass ≈ 1 - diagnostic.retained_mass
        @test diagnostic.count_only_tail_mass + diagnostic.sampling_only_tail_mass + diagnostic.joint_tail_overlap_mass + diagnostic.retained_mass ≈ 1.0 atol=1e-11
        @test diagnostic.count_only_tail_mass + diagnostic.joint_tail_overlap_mass ≈ diagnostic.count_tail_mass atol=1e-11
        @test diagnostic.sampling_only_tail_mass + diagnostic.joint_tail_overlap_mass ≈ diagnostic.sampling_tail_mass atol=1e-11

        audit_amax = reconstructed_count_truncation(ti, tj, tk, reconstructed_pars; atol=1e-10)
        audit_smax = reconstructed_sampling_truncation(ti, tj, tk, reconstructed_pars; atol=1e-10, max_smax=2_000)
        audit_table = reconstructed_joint_pmf_table(audit_amax, audit_smax, ti, tj, tk, reconstructed_pars)
        for a in 0:audit_amax
            @test sum(audit_table[a + 1, :]) ≈ reconstructed_count_pmf(a, ti, tj, tk, reconstructed_pars) atol=2e-10
        end
        for s in 0:audit_smax
            @test sum(audit_table[:, s + 1]) ≈ reconstructed_sampling_marginal_pmf(s, ti, tj, tk, reconstructed_pars) atol=2e-10
        end

        @test_throws ArgumentError reconstructed_pgf_series(-1, ti, tj, tk, reconstructed_pars)
        @test_throws ArgumentError reconstructed_pgf_series(2, tj, ti, tk, reconstructed_pars)
        @test_throws ArgumentError reconstructed_joint_pmf(-1, 0, ti, tj, tk, reconstructed_pars)
        @test_throws ArgumentError reconstructed_joint_pmf(0, -1, ti, tj, tk, reconstructed_pars)
        @test_throws ArgumentError reconstructed_count_truncation(ti, tj, tk, reconstructed_pars; atol=-1.0)
        @test_throws ArgumentError reconstructed_sampling_truncation(ti, tj, tk, reconstructed_pars; atol=-1.0)
        @test_throws ArgumentError reconstructed_sampling_truncation(ti, tj, tk, reconstructed_pars; atol=1e-14, max_smax=0)
        degenerate_series = reconstructed_pgf_series(2, 0.0, 0.5, 1.0, ConstantRateBDParameters(1.0, 0.5, 0.0, 0.0))
        @test all(v -> all(isfinite, v), degenerate_series)
    end

    # Extended reconstructed/conditioned residuals now exercise the public API.
    @testset "extended invariant: transformed-system residuals" begin
        ti = 0.0
        tk = 2.4

        for transformed_pars in KENDALL_REGIMES
            for w in (0.0, 0.4, 1.0), tj in (0.35, 1.0, 1.7)
                α = reconstructed_alpha_bd(w, ti, tj, tk, transformed_pars)
                β = reconstructed_beta_bd(w, ti, tj, tk, transformed_pars)
                γ = reconstructed_gamma_bd(w, ti, tj, tk, transformed_pars)
                λ̃ = transformed_birth_rate(tj, tk, transformed_pars)
                μ̃ = transformed_death_rate(tj, tk, transformed_pars)
                ψ̃ = transformed_sampling_rate(tj, tk, transformed_pars)
                dγ = central_first_derivative(x -> reconstructed_gamma_bd(w, ti, x, tk, transformed_pars), tj; h=2e-5)
                dβ = central_first_derivative(x -> reconstructed_beta_bd(w, ti, x, tk, transformed_pars), tj; h=2e-5)
                dα = central_first_derivative(x -> reconstructed_alpha_bd(w, ti, x, tk, transformed_pars), tj; h=2e-5)

                rate_sum = λ̃ + μ̃ * w + ψ̃ * (1 - w)
                rhs_γ = λ̃ - rate_sum * γ + μ̃ * w * γ^2
                rhs_β = (2μ̃ * w * γ - rate_sum) * β
                rhs_α = μ̃ * w * β

                @test dγ ≈ rhs_γ rtol=5e-6 atol=2e-7
                @test dβ ≈ rhs_β rtol=8e-6 atol=2e-7
                @test dα ≈ rhs_α rtol=8e-6 atol=2e-7
            end
        end
    end

    @testset "extended invariant: transformed Kendall identities" begin
        ti = 0.0
        tk = 2.4

        for transformed_pars in KENDALL_REGIMES
            for tj in (0.35, 1.0, 1.7)
                ξ = reconstructed_xi(ti, tj, tk, transformed_pars)
                η = reconstructed_eta(ti, tj, tk, transformed_pars)
                β1 = reconstructed_beta_bd(1.0, ti, tj, tk, transformed_pars)
                λ̃ = transformed_birth_rate(tj, tk, transformed_pars)
                μ̃ = transformed_death_rate(tj, tk, transformed_pars)
                @test β1 ≈ (1 - ξ) * (1 - η) rtol=2e-11 atol=2e-13

                dη = central_first_derivative(x -> reconstructed_eta(ti, x, tk, transformed_pars), tj; h=2e-5)
                dξ = central_first_derivative(x -> reconstructed_xi(ti, x, tk, transformed_pars), tj; h=2e-5)
                @test dη ≈ (λ̃ - μ̃ * η) * (1 - η) rtol=5e-6 atol=2e-7
                @test dξ ≈ μ̃ * (1 - ξ) * (1 - η) rtol=8e-6 atol=2e-7
            end
        end
    end

    @testset "extinction and survival helpers" begin
        @test E_constant(0.0, λ, μ, ψ) == 1.0
        @test E_constant(0.0, λ, μ, ψ; ρ₀=0.25) == 0.75
        @test E_constant(0.0, pars) == 0.75
        @test g_constant(0.0, λ, μ, ψ) == 0.0
        @test g_constant(0.0, pars) == 0.0

        E1 = E_constant(1.0, λ, μ, ψ)
        g1 = g_constant(1.0, λ, μ, ψ)
        @test E_constant(1.0, pars) ≈ E_constant(1.0, λ, μ, ψ; ρ₀=0.25)
        @test g_constant(1.0, pars) ≈ g_constant(1.0, λ, μ, ψ; ρ₀=0.25)
        @test isfinite(E1)
        @test isfinite(g1)
        @test 0.0 <= E1 <= 1.0
        @test logaddexp(log(0.25), log(0.75)) ≈ 0.0 atol=eps(Float64)

        @test_throws ArgumentError E_constant(1.0, 0.0, μ, ψ)
        @test_throws ArgumentError E_constant(1.0, λ, μ, ψ; ρ₀=-0.1)
        @test_throws ArgumentError g_constant(NaN, λ, μ, ψ)
    end

    @testset "TreeSim likelihood benchmark and admissibility" begin
        tree = tiny_tree()
        @test validate_tree(tree; require_single_root=true, require_reachable=true)

        ll = bd_loglikelihood_constant(tree, λ, μ, ψ, r)
        @test isfinite(ll)
        @test ll ≈ -1.6722221934689507
        @test bd_loglikelihood_constant(tree, ConstantRateBDParameters(λ, μ, ψ, r)) ≈ ll
        @test bd_loglikelihood_constant(tree, pars) ≈ bd_loglikelihood_constant(tree, λ, μ, ψ, r; ρ₀=0.25)

        ll_ascii = bd_loglikelihood_constant(tree, 2, 0.5, 0.4, 0.7)
        @test ll_ascii ≈ ll

        @test_throws ArgumentError bd_loglikelihood_constant(tree, 0.0, μ, ψ, r)
        @test_throws ArgumentError bd_loglikelihood_constant(tree, λ, μ, 0.0, r)
        @test_throws ArgumentError bd_loglikelihood_constant(tree, λ, μ, ψ, -0.01)
        @test_throws ArgumentError bd_loglikelihood_constant(tree, λ, μ, ψ, r; ρ₀=1.01)

        analytically_invalid = unsampled_unary_tree()
        @test validate_tree(analytically_invalid; require_single_root=true, require_reachable=true)
        @test_throws ArgumentError bd_loglikelihood_constant(analytically_invalid, λ, μ, ψ, r)

        @test_throws ArgumentError bd_loglikelihood_constant(Tree(), λ, μ, ψ, r)
        @test_throws ArgumentError bd_loglikelihood_constant(root_only_tree(), λ, μ, ψ, r)
    end

    @testset "constant-rate fitting output" begin
        fit = fit_bd_full(
            tiny_tree();
            param=RateParameterization(BDFixedSpec(:λ, λ)),
            r=r,
            θ_init=log.([μ, ψ]),
        )
        @test fit.constant_rates isa ConstantRateBDParameters
        @test fit.constant_rates.λ == fit.rates.λ
        @test fit.constant_rates.μ == fit.rates.μ
        @test fit.constant_rates.ψ == fit.rates.ψ
        @test fit.constant_rates.r == r
        @test fit.constant_rates.ρ₀ == 0.0
    end

    @testset "reconstructed tree statistics analytics" begin
        stats_pars = ConstantRateBDParameters(1.35, 0.45, 0.75, 0.65)
        t0 = 0.0
        T = 3.0

        @test reconstructed_y(t0, T, stats_pars) ≈ 1 - unsampled_probability(t0, T, stats_pars)
        rates = reconstructed_effective_rates(0.5, T, stats_pars)
        @test rates.b ≈ transformed_birth_rate(0.5, T, stats_pars)
        @test rates.d ≈ transformed_death_rate(0.5, T, stats_pars)
        @test rates.R ≈ rates.b + rates.d

        @test reconstructed_mean_lineages(t0, t0, T, stats_pars) ≈ 1.0
        @test reconstructed_one_tip_probability(T, T, stats_pars) ≈ 1.0
        @test isfinite(expected_reconstructed_cherries(t0, T, stats_pars))
        @test expected_reconstructed_cherries(t0, T, stats_pars) > 0

        q_from_internal = 1 - BDUtils._quad_simpson(
            ℓ -> reconstructed_internal_branch_density(ℓ, t0, T, stats_pars),
            t0,
            T;
            n=1024,
        )
        @test q_from_internal ≈ reconstructed_one_tip_probability(t0, T, stats_pars) atol=1e-8

        tree_counts = reconstructed_tree_stat_counts(tiny_tree())
        @test tree_counts.node_count == 1
        @test tree_counts.cherries == 1
        @test tree_counts.internal_branches == 1
        @test tree_counts.external_branches == 2
    end

    @testset "constant-rate numerical regression regimes" begin
        tree = tiny_tree()

        @test bd_loglikelihood_constant(tree, 1e-9, 1e-12, 1e-12, 1e-6) ≈ -103.27978391012593
        @test bd_loglikelihood_constant(tree, 50.0, 10.0, 5.0, 0.99) ≈ -11.941695345332093
        @test bd_loglikelihood_constant(tree, 2.0, 0.5, 1e-9, 0.0) ≈ -59.312744030768314

        close_times = Tree(
            [0.0, 1e-10, 2e-10],
            [2, 0, 0],
            [3, 0, 0],
            [0, 1, 1],
            [Root, SampledLeaf, SampledLeaf],
            [0, 0, 0],
            [0, 101, 102],
        )
        @test isfinite(bd_loglikelihood_constant(close_times, λ, μ, ψ, r))
    end
end
