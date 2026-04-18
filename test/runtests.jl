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

    @testset "multitype simulation foundation" begin
        pars_mt = MultitypeBDParameters(
            [0.0 1.0; 0.5 0.0],
            [0.1, 0.2],
            [0.3, 0.4],
            [0.0, 1.0],
            [0.0 0.7; 0.2 0.0],
            [0.0, 0.0],
        )
        @test length(pars_mt) == 2
        @test sprint(show, pars_mt) == "MultitypeBDParameters(K=2)"

        handcrafted_mt = MultitypeBDEventLog(
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [2, 1, 2, 2, 1],
            [1, 0, 0, 0, 0],
            [MultitypeBirth, MultitypeTransition, MultitypeFossilizedSampling,
             MultitypeSerialSampling, MultitypeDeath],
            [1, 1, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [1],
            1.0,
        )
        @test length(handcrafted_mt) == 5
        @test handcrafted_mt[1] == MultitypeBDEventRecord(0.2, 2, 1, MultitypeBirth, 1, 2)
        @test collect(record.kind for record in handcrafted_mt) == [
            MultitypeBirth,
            MultitypeTransition,
            MultitypeFossilizedSampling,
            MultitypeSerialSampling,
            MultitypeDeath,
        ]
        @test sprint(show, handcrafted_mt) == "MultitypeBDEventLog(5 events, K=2, initial_lineages=1, tmax=1.0)"
        @test validate_multitype_eventlog(handcrafted_mt)

        @test multitype_NS_at(handcrafted_mt, 0.0) == (N=[1, 0], S=[0, 0])
        @test multitype_NS_at(handcrafted_mt, 0.2) == (N=[1, 1], S=[0, 0])
        @test multitype_NS_at(handcrafted_mt, 0.35) == (N=[0, 2], S=[0, 0])
        @test multitype_NS_at(handcrafted_mt, 0.45) == (N=[0, 2], S=[0, 1])
        @test multitype_NS_at(handcrafted_mt, 1.0) == (N=[0, 0], S=[0, 2])
        @test multitype_N_at(handcrafted_mt, 0.35) == [0, 2]
        @test multitype_S_at(handcrafted_mt, 1.0) == [0, 2]
        @test multitype_N_over_time(handcrafted_mt, [0.0, 0.35, 1.0]) == [[1, 0], [0, 2], [0, 0]]
        @test multitype_S_over_time(handcrafted_mt, [0.0, 0.45, 1.0]) == [[0, 0], [0, 1], [0, 2]]
        @test multitype_mean_N([handcrafted_mt, handcrafted_mt], 0.35) == [0.0, 2.0]

        zero_mt = simulate_multitype_bd(
            MersenneTwister(11),
            MultitypeBDParameters(zeros(2, 2), zeros(2), zeros(2), zeros(2), zeros(2, 2), zeros(2)),
            1.0;
            initial_types=[1, 2],
        )
        @test isempty(zero_mt)
        @test multitype_N_at(zero_mt, 1.0) == [1, 1]
        @test multitype_S_at(zero_mt, 1.0) == [0, 0]
        @test validate_multitype_eventlog(zero_mt)

        transition_only = simulate_multitype_bd(
            MersenneTwister(12),
            MultitypeBDParameters(zeros(2, 2), zeros(2), zeros(2), zeros(2), [0.0 20.0; 0.0 0.0], zeros(2)),
            0.5;
            initial_types=[1],
        )
        @test all(==(MultitypeTransition), transition_only.kind)
        @test validate_multitype_eventlog(transition_only)
        @test sum(multitype_N_at(transition_only, 0.5)) == 1

        k1_pars = MultitypeBDParameters([1.0;;], [0.0], [0.0], [0.0], [0.0;;], [0.0])
        k1_log = simulate_multitype_bd(MersenneTwister(1), k1_pars, 0.5)
        @test all(==(MultitypeBirth), k1_log.kind)
        @test validate_multitype_eventlog(k1_log)
        @test multitype_N_at(k1_log, 0.5) == [1 + length(k1_log)]
        @test multitype_S_at(k1_log, 0.5) == [0]

        sampled_k1 = simulate_multitype_bd(
            MersenneTwister(3),
            MultitypeBDParameters([0.0;;], [0.0], [0.0], [1.0], [0.0;;], [1.0]),
            0.0,
        )
        @test sampled_k1.kind == [MultitypeSerialSampling]
        @test multitype_NS_at(sampled_k1, 0.0) == (N=[0], S=[1])

        bad_mt = MultitypeBDEventLog([0.1], [1], [0], [MultitypeDeath], [2], [2], [1], 1.0)
        @test_throws ArgumentError validate_multitype_eventlog(bad_mt)
        @test_throws ArgumentError MultitypeBDParameters(zeros(2, 2), zeros(1), zeros(1), zeros(1), zeros(2, 2))
        @test_throws ArgumentError MultitypeBDParameters([-1.0;;], [0.0], [0.0], [0.0], [0.0;;])
        @test_throws ArgumentError simulate_multitype_bd(k1_pars, 1.0; initial_types=[2])
        @test_throws ArgumentError multitype_NS_at(handcrafted_mt, -0.1)
        @test_throws ArgumentError multitype_mean_N(MultitypeBDEventLog[], 0.0)
    end

    @testset "multitype analytical backbone" begin
        analytical_mt = MultitypeBDParameters(
            [0.6 0.2; 0.1 0.4],
            [0.3, 0.5],
            [0.2, 0.1],
            [0.0, 0.8],
            [0.0 0.15; 0.05 0.0],
            [0.1, 0.25],
        )

        E0 = multitype_E(0.0, analytical_mt)
        @test E0 ≈ [0.9, 0.75]
        @test all(0 .<= multitype_E(1.0, analytical_mt) .<= 1)
        @test all(0 .<= multitype_E(2.0, analytical_mt) .<= 1)
        @test multitype_E_over_time([0.0, 0.5, 1.0], analytical_mt) == [
            multitype_E(0.0, analytical_mt),
            multitype_E(0.5, analytical_mt),
            multitype_E(1.0, analytical_mt),
        ]

        k1_analytical = MultitypeBDParameters([1.4;;], [0.6], [0.3], [0.7], [0.0;;], [0.2])
        @test only(multitype_E(0.0, k1_analytical)) ≈ E_constant(0.0, ConstantRateBDParameters(1.4, 0.6, 0.3, 0.0, 0.2))
        @test only(multitype_E(1.2, k1_analytical; steps_per_unit=1024)) ≈
              E_constant(1.2, ConstantRateBDParameters(1.4, 0.6, 0.3, 0.0, 0.2)) atol=2e-6

        no_observation = MultitypeBDParameters(
            [0.5 0.1; 0.2 0.4],
            [0.3, 0.7],
            zeros(2),
            zeros(2),
            [0.0 0.2; 0.1 0.0],
            zeros(2),
        )
        @test multitype_E(3.0, no_observation) ≈ [1.0, 1.0]

        no_birth = MultitypeBDParameters(zeros(2, 2), [0.4, 0.2], [0.6, 0.8], [0.0, 1.0], zeros(2, 2), [0.0, 0.0])
        E_no_birth = multitype_E(1.5, no_birth)
        @test all(multitype_E(2.0, no_birth) .<= multitype_E(1.0, no_birth) .+ 1e-8)
        @test E_no_birth[1] ≈ 0.4 / 1.0 + (1 - 0.4 / 1.0) * exp(-1.0 * 1.5) atol=1e-8
        @test E_no_birth[2] ≈ 0.2 / 1.0 + (1 - 0.2 / 1.0) * exp(-1.0 * 1.5) atol=1e-8

        death_only_with_present_sampling = MultitypeBDParameters(zeros(1, 1), [0.7], [0.0], [0.0], zeros(1, 1), [0.3])
        @test only(multitype_E(2.0, death_only_with_present_sampling)) ≈ 1 - 0.3 * exp(-0.7 * 2.0) atol=1e-8

        flow_no_birth = multitype_log_flow(1.25, no_birth)
        @test flow_no_birth ≈ [-1.0 * 1.25, -1.0 * 1.25] atol=1e-8
        @test multitype_flow(1.25, no_birth) ≈ exp.(flow_no_birth)
        @test multitype_log_flow(0.0, analytical_mt) == [0.0, 0.0]

        rng = MersenneTwister(20260417)
        sim_pars = MultitypeBDParameters(zeros(1, 1), [0.4], [0.6], [1.0], zeros(1, 1), [0.0])
        nsims = 2_000
        no_sample_count = 0
        for _ in 1:nsims
            log = simulate_multitype_bd(rng, sim_pars, 1.0; apply_ρ₀=true)
            no_sample_count += multitype_S_at(log, 1.0)[1] == 0
        end
        @test no_sample_count / nsims ≈ only(multitype_E(1.0, sim_pars)) atol=0.04

        @test_throws ArgumentError multitype_E(-0.1, analytical_mt)
        @test_throws ArgumentError multitype_E(1.0, analytical_mt; steps_per_unit=0)
        @test_throws ArgumentError multitype_E_over_time([0.0, Inf], analytical_mt)
        @test_throws ArgumentError multitype_log_flow(NaN, analytical_mt)
    end

    @testset "multitype fully colored likelihood" begin
        one_type_pars = MultitypeBDParameters([1.2;;], [0.4], [0.3], [0.5], [0.0;;], [0.8])
        one_type_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.0, 1.0)],
            present_samples=[1],
        )
        @test validate_multitype_colored_tree(one_type_tree, one_type_pars)
        expected_one_type = log(0.8) + only(multitype_log_flow(1.0, one_type_pars))
        @test multitype_colored_loglikelihood(one_type_tree, one_type_pars) ≈ expected_one_type
        @test multitype_colored_likelihood(one_type_tree, one_type_pars) ≈ exp(expected_one_type)

        two_type_pars = MultitypeBDParameters(
            [0.5 0.7; 0.2 0.4],
            [0.2, 0.3],
            [0.6, 0.5],
            [0.25, 0.8],
            [0.0 0.9; 0.1 0.0],
            [0.4, 0.6],
        )
        birth_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.5, 1.0),
                MultitypeColoredSegment(1, 0.0, 0.5),
                MultitypeColoredSegment(2, 0.0, 0.5),
            ],
            births=[MultitypeColoredBirth(0.5, 1, 2)],
            present_samples=[1, 2],
        )
        logflow_05 = multitype_log_flow(0.5, two_type_pars)
        logflow_10 = multitype_log_flow(1.0, two_type_pars)
        expected_birth = (logflow_10[1] - logflow_05[1]) + logflow_05[1] + logflow_05[2] +
                         log(two_type_pars.birth[1, 2]) + log(two_type_pars.ρ₀[1]) + log(two_type_pars.ρ₀[2])
        @test multitype_colored_loglikelihood(birth_tree, two_type_pars) ≈ expected_birth

        transition_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.4, 1.0),
                MultitypeColoredSegment(2, 0.0, 0.4),
            ],
            transitions=[MultitypeColoredTransition(0.4, 1, 2)],
            present_samples=[2],
        )
        expected_transition = (logflow_10[1] - multitype_log_flow(0.4, two_type_pars)[1]) +
                              multitype_log_flow(0.4, two_type_pars)[2] +
                              log(two_type_pars.transition[1, 2]) + log(two_type_pars.ρ₀[2])
        @test multitype_colored_loglikelihood(transition_tree, two_type_pars) ≈ expected_transition

        hidden_birth_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.4, 1.0),
                MultitypeColoredSegment(2, 0.0, 0.4),
            ],
            hidden_births=[MultitypeColoredHiddenBirth(0.4, 1, 2)],
            present_samples=[2],
        )
        E04 = multitype_E(0.4, two_type_pars)[1]
        expected_hidden_birth = (logflow_10[1] - multitype_log_flow(0.4, two_type_pars)[1]) +
                                multitype_log_flow(0.4, two_type_pars)[2] +
                                log(two_type_pars.birth[1, 2] * E04) + log(two_type_pars.ρ₀[2])
        @test multitype_colored_loglikelihood(hidden_birth_tree, two_type_pars) ≈ expected_hidden_birth
        @test multitype_colored_loglikelihood(hidden_birth_tree, two_type_pars) !=
              multitype_colored_loglikelihood(transition_tree, two_type_pars)

        sampling_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.7, 1.0),
                MultitypeColoredSegment(1, 0.2, 0.7),
            ],
            terminal_samples=[MultitypeColoredSampling(0.2, 1)],
            ancestral_samples=[MultitypeColoredSampling(0.7, 1)],
        )
        E02 = multitype_E(0.2, two_type_pars)[1]
        lf02 = multitype_log_flow(0.2, two_type_pars)
        lf07 = multitype_log_flow(0.7, two_type_pars)
        expected_sampling = (logflow_10[1] - lf07[1]) + (lf07[1] - lf02[1]) +
                            log(two_type_pars.sampling[1] * (1 - two_type_pars.removal_probability[1])) +
                            log(two_type_pars.sampling[1] * (two_type_pars.removal_probability[1] +
                                (1 - two_type_pars.removal_probability[1]) * E02))
        @test multitype_colored_loglikelihood(sampling_tree, two_type_pars) ≈ expected_sampling

        higher_birth = MultitypeBDParameters(
            [0.5 1.4; 0.2 0.4],
            two_type_pars.death,
            two_type_pars.sampling,
            two_type_pars.removal_probability,
            two_type_pars.transition,
            two_type_pars.ρ₀,
        )
        @test multitype_colored_loglikelihood(birth_tree, higher_birth) >
              multitype_colored_loglikelihood(birth_tree, two_type_pars)

        bad_root = MultitypeColoredTree(1.0, 1; segments=[MultitypeColoredSegment(2, 0.0, 1.0)])
        @test_throws ArgumentError validate_multitype_colored_tree(bad_root, two_type_pars)
        @test_throws ArgumentError MultitypeColoredTree(-1.0, 1; segments=[MultitypeColoredSegment(1, 0.0, 1.0)]) |>
                                   tree -> validate_multitype_colored_tree(tree, one_type_pars)
        bad_birth = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.0, 1.0)],
            births=[MultitypeColoredBirth(0.5, 1, 2)],
        )
        @test_throws ArgumentError multitype_colored_loglikelihood(bad_birth, one_type_pars)
        bad_hidden_birth = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.0, 1.0)],
            hidden_births=[MultitypeColoredHiddenBirth(0.5, 1, 2)],
        )
        @test_throws ArgumentError multitype_colored_loglikelihood(bad_hidden_birth, one_type_pars)
        impossible_present = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.0, 1.0)],
            present_samples=[1],
        )
        no_present_sampling = MultitypeBDParameters([0.0;;], [0.1], [0.0], [0.0], [0.0;;], [0.0])
        @test_throws ArgumentError multitype_colored_loglikelihood(impossible_present, no_present_sampling)
    end

    @testset "uncoloured MTBD-2 likelihood" begin
        mtbd2_pars = UncolouredMTBD2ConstantParameters(
            [0.8 0.35; 0.25 0.7],
            [0.2, 0.3],
            [0.45, 0.55],
            [0.7, 0.4],
            [0.0 0.12; 0.08 0.0],
            [0.2, 0.3],
        )
        tree = tiny_tree()
        tip_states = Dict(3 => 1, 4 => 2)

        ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, tip_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8)
        @test isfinite(ll)
        @test likelihood_uncoloured_mtbd2(tree, mtbd2_pars, tip_states;
                                                     root_prior=[0.6, 0.4],
                                                     steps_per_unit=64,
                                                     min_steps=8) ≈ exp(ll)
        @test ll ≈ -4.404102999291537
        @test loglikelihood_uncoloured_mtbd2_known_tips(tree, mtbd2_pars, tip_states;
                                                        root_prior=[0.6, 0.4],
                                                        steps_per_unit=64,
                                                        min_steps=8) ≈ ll
        @test likelihood_uncoloured_mtbd2_known_tips(tree, mtbd2_pars, tip_states;
                                                     root_prior=[0.6, 0.4],
                                                     steps_per_unit=64,
                                                     min_steps=8) ≈ exp(ll)

        vector_tip_states = [0, 0, 1, 2]
        @test loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, vector_tip_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ ll
        @test loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => (1,), 4 => [2]);
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ ll

        unknown_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => missing, 4 => nothing);
                                                              root_prior=[0.6, 0.4],
                                                              steps_per_unit=64,
                                                              min_steps=8)
        @test isfinite(unknown_ll)
        @test unknown_ll >= ll
        @test loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => (1, 2), 4 => [true, true]);
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ unknown_ll

        mixed_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => 1, 4 => :);
                                                            root_prior=[0.6, 0.4],
                                                            steps_per_unit=64,
                                                            min_steps=8)
        @test isfinite(mixed_ll)
        @test ll <= mixed_ll <= unknown_ll

        swapped = Tree(
            [0.0, 0.6, 1.0, 1.4],
            [2, 4, 0, 0],
            [0, 3, 0, 0],
            [0, 1, 2, 2],
            [Root, Binary, SampledLeaf, SampledLeaf],
            [0, 0, 0, 0],
            [0, 0, 101, 102],
        )
        @test loglikelihood_uncoloured_mtbd2(swapped, mtbd2_pars, tip_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ ll
        @test loglikelihood_uncoloured_mtbd2(swapped, mtbd2_pars, Dict(3 => missing, 4 => nothing);
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ unknown_ll

        present_tip_tree = Tree(
            [0.0, 1.0, 1.0],
            [2, 0, 0],
            [3, 0, 0],
            [0, 1, 1],
            [Root, SampledLeaf, SampledLeaf],
            [0, 0, 0],
            [0, 101, 102],
        )
        present_pars = UncolouredMTBD2ConstantParameters(
            [0.9 0.1; 0.2 0.8],
            [0.1, 0.1],
            [0.3, 0.3],
            [1.0, 1.0],
            zeros(2, 2),
            [0.25, 0.75],
        )
        root1_ll = loglikelihood_uncoloured_mtbd2(present_tip_tree, present_pars, Dict(2 => 1, 3 => 2);
                                                             root_prior=[1.0, 0.0])
        root2_ll = loglikelihood_uncoloured_mtbd2(present_tip_tree, present_pars, Dict(2 => 1, 3 => 2);
                                                             root_prior=[0.0, 1.0])
        richer_present_pars = UncolouredMTBD2ConstantParameters(
            present_pars.birth,
            present_pars.death,
            present_pars.sampling,
            present_pars.removal_probability,
            present_pars.transition,
            [0.5, 0.9],
        )
        richer_present_ll = loglikelihood_uncoloured_mtbd2(present_tip_tree, richer_present_pars, Dict(2 => 1, 3 => 2);
                                                                      root_prior=[1.0, 0.0])
        @test isfinite(root1_ll)
        @test isfinite(root2_ll)
        @test root1_ll != root2_ll
        @test richer_present_ll > root1_ll

        sampled_unary_tree = Tree(
            [0.0, 0.4, 0.8, 1.0, 1.2],
            [2, 3, 4, 0, 0],
            [0, 5, 0, 0, 0],
            [0, 1, 2, 3, 2],
            [Root, Binary, SampledUnary, SampledLeaf, SampledLeaf],
            [0, 0, 0, 0, 0],
            [0, 0, 201, 202, 203],
        )
        sampled_unary_states = Dict(3 => 1, 4 => 2, 5 => 1)
        sampled_unary_ll = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars, sampled_unary_states;
                                                          root_prior=[0.6, 0.4],
                                                          steps_per_unit=64,
                                                          min_steps=8)
        @test isfinite(sampled_unary_ll)

        sampled_unary_unknown_ll = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars,
                                                                  Dict(3 => missing, 4 => nothing, 5 => :);
                                                                  root_prior=[0.6, 0.4],
                                                                  steps_per_unit=64,
                                                                  min_steps=8)
        sampled_unary_mixed_ll = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars,
                                                                Dict(3 => 1, 4 => missing, 5 => 1);
                                                                root_prior=[0.6, 0.4],
                                                                steps_per_unit=64,
                                                                min_steps=8)
        @test isfinite(sampled_unary_unknown_ll)
        @test isfinite(sampled_unary_mixed_ll)
        @test sampled_unary_ll <= sampled_unary_mixed_ll <= sampled_unary_unknown_ll

        sampled_unary_swapped = Tree(
            [0.0, 0.4, 0.8, 1.0, 1.2],
            [2, 5, 4, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 1, 2, 3, 2],
            [Root, Binary, SampledUnary, SampledLeaf, SampledLeaf],
            [0, 0, 0, 0, 0],
            [0, 0, 201, 202, 203],
        )
        @test loglikelihood_uncoloured_mtbd2(sampled_unary_swapped, mtbd2_pars, sampled_unary_states;
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8) ≈ sampled_unary_ll
        @test loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars, Dict(3 => [1], 4 => [false, true], 5 => (1,));
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8) ≈ sampled_unary_ll

        batch_trees = [tree, swapped, sampled_unary_tree]
        batch_observations = [
            tip_states,
            Dict(3 => missing, 4 => nothing),
            Dict(3 => 1, 4 => missing, 5 => 1),
        ]
        repeated_batch_ll = [
            loglikelihood_uncoloured_mtbd2(batch_trees[i], mtbd2_pars, batch_observations[i];
                                           root_prior=[0.6, 0.4],
                                           steps_per_unit=64,
                                           min_steps=8)
            for i in eachindex(batch_trees)
        ]
        batch_ll = loglikelihoods_uncoloured_mtbd2(batch_trees, mtbd2_pars, batch_observations;
                                                   root_prior=[0.6, 0.4],
                                                   steps_per_unit=64,
                                                   min_steps=8)
        @test batch_ll ≈ repeated_batch_ll
        @test likelihoods_uncoloured_mtbd2(batch_trees, mtbd2_pars, batch_observations;
                                           root_prior=[0.6, 0.4],
                                           steps_per_unit=64,
                                           min_steps=8) ≈ exp.(batch_ll)
        @test total_loglikelihood_uncoloured_mtbd2(batch_trees, mtbd2_pars, batch_observations;
                                                   root_prior=[0.6, 0.4],
                                                   steps_per_unit=64,
                                                   min_steps=8) ≈ sum(batch_ll)
        batch_score = score_uncoloured_mtbd2(batch_trees, mtbd2_pars, batch_observations;
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8)
        @test batch_score.per_tree_loglikelihood ≈ batch_ll
        @test batch_score.total_loglikelihood ≈ sum(batch_ll)
        @test batch_score.mean_loglikelihood ≈ sum(batch_ll) / length(batch_ll)
        @test batch_score.n_scored == 3
        @test_throws ArgumentError loglikelihoods_uncoloured_mtbd2(batch_trees, mtbd2_pars, batch_observations[1:2])

        θ = uncoloured_mtbd2_parameter_vector(mtbd2_pars)
        @test length(θ) == length(UNCOLOURED_MTBD2_PARAMETER_ORDER) == 16
        @test θ == [
            mtbd2_pars.birth[1, 1], mtbd2_pars.birth[1, 2], mtbd2_pars.birth[2, 1], mtbd2_pars.birth[2, 2],
            mtbd2_pars.death[1], mtbd2_pars.death[2],
            mtbd2_pars.sampling[1], mtbd2_pars.sampling[2],
            mtbd2_pars.removal_probability[1], mtbd2_pars.removal_probability[2],
            mtbd2_pars.transition[1, 1], mtbd2_pars.transition[1, 2], mtbd2_pars.transition[2, 1], mtbd2_pars.transition[2, 2],
            mtbd2_pars.ρ₀[1], mtbd2_pars.ρ₀[2],
        ]
        roundtrip_pars = uncoloured_mtbd2_parameters_from_vector(θ)
        @test roundtrip_pars.birth == mtbd2_pars.birth
        @test roundtrip_pars.death == mtbd2_pars.death
        @test roundtrip_pars.sampling == mtbd2_pars.sampling
        @test roundtrip_pars.removal_probability == mtbd2_pars.removal_probability
        @test roundtrip_pars.transition == mtbd2_pars.transition
        @test roundtrip_pars.ρ₀ == mtbd2_pars.ρ₀

        all_free_spec = UncolouredMTBD2ParameterSpec(mtbd2_pars)
        @test free_parameter_vector(all_free_spec) == θ
        @test free_parameter_vector(mtbd2_pars, all_free_spec) == θ
        all_free_rebuilt = uncoloured_mtbd2_parameters_from_free_vector(θ, all_free_spec)
        @test uncoloured_mtbd2_parameter_vector(all_free_rebuilt) == θ

        partial_spec = UncolouredMTBD2ParameterSpec(
            mtbd2_pars;
            birth=[true false; false true],
            death=[false, true],
            sampling=[true, false],
            removal_probability=[false, false],
            transition=[false true; true false],
            ρ₀=[true, false],
        )
        partial_free = free_parameter_vector(mtbd2_pars, partial_spec)
        @test partial_free == [
            mtbd2_pars.birth[1, 1],
            mtbd2_pars.birth[2, 2],
            mtbd2_pars.death[2],
            mtbd2_pars.sampling[1],
            mtbd2_pars.transition[1, 2],
            mtbd2_pars.transition[2, 1],
            mtbd2_pars.ρ₀[1],
        ]
        changed_free = partial_free .+ [0.1, 0.2, 0.03, 0.04, 0.01, 0.02, -0.05]
        changed_pars = uncoloured_mtbd2_parameters_from_free_vector(changed_free, partial_spec)
        @test changed_pars.birth[1, 1] == changed_free[1]
        @test changed_pars.birth[1, 2] == mtbd2_pars.birth[1, 2]
        @test changed_pars.birth[2, 1] == mtbd2_pars.birth[2, 1]
        @test changed_pars.birth[2, 2] == changed_free[2]
        @test changed_pars.death[1] == mtbd2_pars.death[1]
        @test changed_pars.death[2] == changed_free[3]
        @test changed_pars.sampling[1] == changed_free[4]
        @test changed_pars.sampling[2] == mtbd2_pars.sampling[2]
        @test changed_pars.transition[1, 2] == changed_free[5]
        @test changed_pars.transition[2, 1] == changed_free[6]
        @test changed_pars.ρ₀[1] == changed_free[7]
        @test changed_pars.ρ₀[2] == mtbd2_pars.ρ₀[2]

        fixed_spec = UncolouredMTBD2ParameterSpec(mtbd2_pars; birth=falses(2, 2), death=falses(2),
                                                  sampling=falses(2), removal_probability=falses(2),
                                                  transition=falses(2, 2), ρ₀=falses(2))
        @test isempty(free_parameter_vector(fixed_spec))
        @test uncoloured_mtbd2_parameter_vector(uncoloured_mtbd2_parameters_from_free_vector(Float64[], fixed_spec)) == θ

        @test loglikelihood_uncoloured_mtbd2_from_free(θ, tree, all_free_spec, tip_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ ll
        @test loglikelihood_uncoloured_mtbd2_from_free(θ, sampled_unary_tree, all_free_spec, sampled_unary_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ sampled_unary_ll
        @test total_loglikelihood_uncoloured_mtbd2_from_free(θ, batch_trees, all_free_spec, batch_observations;
                                                             root_prior=[0.6, 0.4],
                                                             steps_per_unit=64,
                                                             min_steps=8) ≈ sum(batch_ll)
        @test loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, tip_states;
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8) ≈ ll

        @test_throws ArgumentError uncoloured_mtbd2_parameters_from_vector(θ[1:15])
        @test_throws ArgumentError uncoloured_mtbd2_parameters_from_vector(vcat(-0.1, θ[2:end]))
        @test_throws ArgumentError UncolouredMTBD2ParameterSpec(mtbd2_pars; birth=trues(3, 2))
        @test_throws ArgumentError UncolouredMTBD2ParameterSpec(mtbd2_pars; death=trues(3))
        @test_throws ArgumentError uncoloured_mtbd2_parameters_from_free_vector(partial_free[1:6], partial_spec)

        mle_spec = UncolouredMTBD2ParameterSpec(
            mtbd2_pars;
            birth=falses(2, 2),
            death=falses(2),
            sampling=[true, false],
            removal_probability=falses(2),
            transition=falses(2, 2),
            ρ₀=falses(2),
        )
        θ0_mle = free_parameter_vector(mtbd2_pars, mle_spec)
        initial_single_nll = -loglikelihood_uncoloured_mtbd2_from_free(θ0_mle, tree, mle_spec, tip_states;
                                                                       root_prior=[0.6, 0.4],
                                                                       steps_per_unit=64,
                                                                       min_steps=8)
        single_fit = fit_uncoloured_mtbd2_mle(tree, θ0_mle, mle_spec, tip_states;
                                              root_prior=[0.6, 0.4],
                                              steps_per_unit=64,
                                              min_steps=8,
                                              lower=[1e-6],
                                              upper=[2.0],
                                              initial_step=0.05,
                                              tolerance=1e-4,
                                              maxiter=100)
        @test length(single_fit.θ_free_hat) == 1
        @test single_fit.params_hat isa UncolouredMTBD2ConstantParameters
        @test isfinite(single_fit.maximum_loglikelihood)
        @test isfinite(single_fit.minimum_negloglikelihood)
        @test single_fit.minimum_negloglikelihood ≈ -single_fit.maximum_loglikelihood
        @test single_fit.initial_negloglikelihood ≈ initial_single_nll
        @test single_fit.minimum_negloglikelihood <= initial_single_nll + 1e-8
        @test single_fit.spec === mle_spec
        @test haskey(single_fit.optimizer_summary, :minimizer)

        single_fit_from_params = fit_uncoloured_mtbd2_mle(tree, mtbd2_pars, mle_spec, tip_states;
                                                         root_prior=[0.6, 0.4],
                                                         steps_per_unit=64,
                                                         min_steps=8,
                                                         lower=[1e-6],
                                                         upper=[2.0],
                                                         initial_step=0.05,
                                                         tolerance=1e-4,
                                                         maxiter=100)
        @test single_fit_from_params.initial_negloglikelihood ≈ single_fit.initial_negloglikelihood
        @test single_fit_from_params.minimum_negloglikelihood ≈ single_fit.minimum_negloglikelihood

        batch_fit = fit_uncoloured_mtbd2_mle(batch_trees, mtbd2_pars, mle_spec, batch_observations;
                                            root_prior=[0.6, 0.4],
                                            steps_per_unit=64,
                                            min_steps=8,
                                            lower=[1e-6],
                                            upper=[2.0],
                                            initial_step=0.05,
                                            tolerance=1e-4,
                                            maxiter=100)
        initial_batch_nll = -total_loglikelihood_uncoloured_mtbd2_from_free(θ0_mle, batch_trees, mle_spec, batch_observations;
                                                                            root_prior=[0.6, 0.4],
                                                                            steps_per_unit=64,
                                                                            min_steps=8)
        @test isfinite(batch_fit.maximum_loglikelihood)
        @test batch_fit.minimum_negloglikelihood ≈ -batch_fit.maximum_loglikelihood
        @test batch_fit.initial_negloglikelihood ≈ initial_batch_nll
        @test batch_fit.minimum_negloglikelihood <= initial_batch_nll + 1e-8

        @test_throws ArgumentError fit_uncoloured_mtbd2_mle(tree, Float64[], fixed_spec, tip_states)
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle(Tree[], θ0_mle, mle_spec, Dict[])
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle(batch_trees, θ0_mle, mle_spec, batch_observations[1:2])
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle(tree, Float64[], mle_spec, tip_states)
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle(tree, θ0_mle, mle_spec, tip_states; lower=[0.0, 0.0])

        transform_spec = UncolouredMTBD2ParameterSpec(
            mtbd2_pars;
            birth=[true false; false false],
            death=[false, true],
            sampling=[true, false],
            removal_probability=[true, false],
            transition=[false true; false false],
            ρ₀=[true, false],
        )
        θ_transform = free_parameter_vector(mtbd2_pars, transform_spec)
        η_transform = uncoloured_mtbd2_unconstrained_from_free(θ_transform, transform_spec)
        @test uncoloured_mtbd2_free_from_unconstrained(η_transform, transform_spec) ≈ θ_transform
        @test uncoloured_mtbd2_unconstrained_from_free(free_parameter_vector(mtbd2_pars, transform_spec), transform_spec) ≈ η_transform
        prob_spec = UncolouredMTBD2ParameterSpec(
            mtbd2_pars;
            birth=falses(2, 2),
            death=falses(2),
            sampling=falses(2),
            removal_probability=[true, false],
            transition=falses(2, 2),
            ρ₀=[true, false],
        )
        prob_free = free_parameter_vector(mtbd2_pars, prob_spec)
        prob_eta = uncoloured_mtbd2_unconstrained_from_free(prob_free, prob_spec)
        @test prob_eta ≈ log.(prob_free ./ (1 .- prob_free))
        @test uncoloured_mtbd2_free_from_unconstrained(prob_eta, prob_spec) ≈ prob_free

        @test loglikelihood_uncoloured_mtbd2_from_unconstrained(η_transform, tree, transform_spec, tip_states;
                                                                root_prior=[0.6, 0.4],
                                                                steps_per_unit=64,
                                                                min_steps=8) ≈
              loglikelihood_uncoloured_mtbd2_from_free(θ_transform, tree, transform_spec, tip_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8)
        @test loglikelihood_uncoloured_mtbd2_from_unconstrained(η_transform, sampled_unary_tree, transform_spec, sampled_unary_states;
                                                                root_prior=[0.6, 0.4],
                                                                steps_per_unit=64,
                                                                min_steps=8) ≈
              loglikelihood_uncoloured_mtbd2_from_free(θ_transform, sampled_unary_tree, transform_spec, sampled_unary_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8)
        @test total_loglikelihood_uncoloured_mtbd2_from_unconstrained(η_transform, batch_trees, transform_spec, batch_observations;
                                                                      root_prior=[0.6, 0.4],
                                                                      steps_per_unit=64,
                                                                      min_steps=8) ≈
              total_loglikelihood_uncoloured_mtbd2_from_free(θ_transform, batch_trees, transform_spec, batch_observations;
                                                             root_prior=[0.6, 0.4],
                                                             steps_per_unit=64,
                                                             min_steps=8)

        transformed_fit = fit_uncoloured_mtbd2_mle_transformed(tree, θ0_mle, mle_spec, tip_states;
                                                              root_prior=[0.6, 0.4],
                                                              steps_per_unit=64,
                                                              min_steps=8,
                                                              initial_step=0.05,
                                                              tolerance=1e-4,
                                                              maxiter=100)
        @test length(transformed_fit.η_free_hat) == length(transformed_fit.θ_free_hat) == 1
        @test transformed_fit.params_hat isa UncolouredMTBD2ConstantParameters
        @test transformed_fit.θ_free_hat ≈ free_parameter_vector(transformed_fit.params_hat, mle_spec)
        @test transformed_fit.minimum_negloglikelihood ≈ -transformed_fit.maximum_loglikelihood
        @test transformed_fit.minimum_negloglikelihood <= transformed_fit.initial_negloglikelihood + 1e-8
        @test transformed_fit.spec === mle_spec

        transformed_fit_from_eta = fit_uncoloured_mtbd2_mle_transformed(tree, log.(θ0_mle), mle_spec, tip_states;
                                                                        init_is_unconstrained=true,
                                                                        root_prior=[0.6, 0.4],
                                                                        steps_per_unit=64,
                                                                        min_steps=8,
                                                                        initial_step=0.05,
                                                                        tolerance=1e-4,
                                                                        maxiter=100)
        @test transformed_fit_from_eta.initial_negloglikelihood ≈ transformed_fit.initial_negloglikelihood

        transformed_batch_fit = fit_uncoloured_mtbd2_mle_transformed(batch_trees, mtbd2_pars, mle_spec, batch_observations;
                                                                     root_prior=[0.6, 0.4],
                                                                     steps_per_unit=64,
                                                                     min_steps=8,
                                                                     initial_step=0.05,
                                                                     tolerance=1e-4,
                                                                     maxiter=100)
        @test isfinite(transformed_batch_fit.maximum_loglikelihood)
        @test transformed_batch_fit.minimum_negloglikelihood <= transformed_batch_fit.initial_negloglikelihood + 1e-8

        @test_throws ArgumentError uncoloured_mtbd2_unconstrained_from_free([0.0], mle_spec)
        @test_throws ArgumentError uncoloured_mtbd2_unconstrained_from_free([1.0], prob_spec)
        @test_throws ArgumentError uncoloured_mtbd2_free_from_unconstrained(Float64[], mle_spec)
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle_transformed(tree, Float64[], fixed_spec, tip_states)
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle_transformed(tree, mtbd2_pars, mle_spec, tip_states; init_is_unconstrained=true)
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle_transformed(batch_trees, θ0_mle, mle_spec, batch_observations[1:2])

        superspreader_pars = UncolouredMTBD2SuperspreaderParameters(
            1.8,
            0.2,
            6.0,
            [0.15, 0.25],
            [0.5, 0.7],
            [0.6, 0.4],
            [0.1, 0.2],
        )
        native_from_superspreader = uncoloured_mtbd2_native_parameters(superspreader_pars)
        q = superspreader_pars.superspreader_fraction
        ptypes = [1 - q, q]
        τ = [1.0, superspreader_pars.relative_transmissibility]
        δ1 = superspreader_pars.death[1] + superspreader_pars.sampling[1] * superspreader_pars.removal_probability[1]
        δ2 = superspreader_pars.death[2] + superspreader_pars.sampling[2] * superspreader_pars.removal_probability[2]
        c = superspreader_pars.total_R0 / (ptypes[1] * τ[1] + ptypes[2] * τ[2])
        λ1 = c * τ[1] * δ1
        λ2 = c * τ[2] * δ2
        @test native_from_superspreader.birth ≈ [λ1 * ptypes[1] λ1 * ptypes[2];
                                                 λ2 * ptypes[1] λ2 * ptypes[2]]
        @test native_from_superspreader.death == superspreader_pars.death
        @test native_from_superspreader.sampling == superspreader_pars.sampling
        @test native_from_superspreader.removal_probability == superspreader_pars.removal_probability
        @test native_from_superspreader.transition == zeros(2, 2)
        @test native_from_superspreader.ρ₀ == superspreader_pars.ρ₀
        @test sum(ptypes[i] * (sum(native_from_superspreader.birth[i, :]) / (i == 1 ? δ1 : δ2)) for i in 1:2) ≈
              superspreader_pars.total_R0
        @test sum(native_from_superspreader.birth[2, :]) / δ2 ≈
              superspreader_pars.relative_transmissibility * sum(native_from_superspreader.birth[1, :]) / δ1

        superspreader_vector = uncoloured_mtbd2_superspreader_parameter_vector(superspreader_pars)
        @test length(superspreader_vector) == length(UNCOLOURED_MTBD2_SUPERSPREADER_PARAMETER_ORDER) == 11
        superspreader_roundtrip = uncoloured_mtbd2_superspreader_parameters_from_vector(superspreader_vector)
        @test uncoloured_mtbd2_superspreader_parameter_vector(superspreader_roundtrip) == superspreader_vector

        @test loglikelihood_uncoloured_mtbd2_superspreader(tree, superspreader_pars, tip_states;
                                                           root_prior=[0.6, 0.4],
                                                           steps_per_unit=64,
                                                           min_steps=8) ≈
              loglikelihood_uncoloured_mtbd2(tree, native_from_superspreader, tip_states;
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8)
        @test likelihood_uncoloured_mtbd2_superspreader(tree, superspreader_pars, tip_states;
                                                        root_prior=[0.6, 0.4],
                                                        steps_per_unit=64,
                                                        min_steps=8) ≈
              exp(loglikelihood_uncoloured_mtbd2(tree, native_from_superspreader, tip_states;
                                                 root_prior=[0.6, 0.4],
                                                 steps_per_unit=64,
                                                 min_steps=8))
        @test total_loglikelihood_uncoloured_mtbd2_superspreader(batch_trees, superspreader_pars, batch_observations;
                                                                 root_prior=[0.6, 0.4],
                                                                 steps_per_unit=64,
                                                                 min_steps=8) ≈
              total_loglikelihood_uncoloured_mtbd2(batch_trees, native_from_superspreader, batch_observations;
                                                   root_prior=[0.6, 0.4],
                                                   steps_per_unit=64,
                                                   min_steps=8)

        @test_throws ArgumentError UncolouredMTBD2SuperspreaderParameters(0.0, 0.2, 6.0, [0.15, 0.25], [0.5, 0.7], [0.6, 0.4], [0.1, 0.2])
        @test_throws ArgumentError UncolouredMTBD2SuperspreaderParameters(1.8, 0.0, 6.0, [0.15, 0.25], [0.5, 0.7], [0.6, 0.4], [0.1, 0.2])
        @test_throws ArgumentError UncolouredMTBD2SuperspreaderParameters(1.8, 0.2, 0.0, [0.15, 0.25], [0.5, 0.7], [0.6, 0.4], [0.1, 0.2])
        @test_throws ArgumentError UncolouredMTBD2SuperspreaderParameters(1.8, 0.2, 6.0, [0.0, 0.25], [0.0, 0.7], [0.0, 0.4], [0.1, 0.2])
        @test_throws ArgumentError uncoloured_mtbd2_superspreader_parameters_from_vector(superspreader_vector[1:10])

        superspreader_roundtrip2 = uncoloured_mtbd2_superspreader_parameters_from_vector(
            uncoloured_mtbd2_superspreader_parameter_vector(superspreader_pars),
        )
        @test superspreader_roundtrip2.total_R0 == superspreader_pars.total_R0
        @test superspreader_roundtrip2.superspreader_fraction == superspreader_pars.superspreader_fraction
        @test superspreader_roundtrip2.relative_transmissibility == superspreader_pars.relative_transmissibility
        @test superspreader_roundtrip2.death == superspreader_pars.death
        @test superspreader_roundtrip2.sampling == superspreader_pars.sampling
        @test superspreader_roundtrip2.removal_probability == superspreader_pars.removal_probability
        @test superspreader_roundtrip2.ρ₀ == superspreader_pars.ρ₀

        superspreader_spec = UncolouredMTBD2SuperspreaderSpec(
            superspreader_pars;
            total_R0=true,
            superspreader_fraction=false,
            relative_transmissibility=true,
            death=[true, false],
            sampling=[false, true],
            removal_probability=[true, false],
            ρ₀=[false, true],
        )
        θ_super_free = free_parameter_vector(superspreader_pars, superspreader_spec)
        @test θ_super_free == [
            superspreader_pars.total_R0,
            superspreader_pars.relative_transmissibility,
            superspreader_pars.death[1],
            superspreader_pars.sampling[2],
            superspreader_pars.removal_probability[1],
            superspreader_pars.ρ₀[2],
        ]
        superspreader_rebuilt = uncoloured_mtbd2_superspreader_parameters_from_free_vector(
            θ_super_free, superspreader_spec,
        )
        @test uncoloured_mtbd2_superspreader_parameter_vector(superspreader_rebuilt) == superspreader_vector
        θ_super_changed = copy(θ_super_free)
        θ_super_changed[1] = 2.1
        θ_super_changed[2] = 4.0
        superspreader_changed = uncoloured_mtbd2_superspreader_parameters_from_free_vector(
            θ_super_changed, superspreader_spec,
        )
        @test superspreader_changed.total_R0 == 2.1
        @test superspreader_changed.superspreader_fraction == superspreader_pars.superspreader_fraction
        @test superspreader_changed.relative_transmissibility == 4.0
        @test superspreader_changed.death[2] == superspreader_pars.death[2]
        @test_throws ArgumentError UncolouredMTBD2SuperspreaderSpec(superspreader_pars; death=[true])
        @test_throws ArgumentError uncoloured_mtbd2_superspreader_parameters_from_free_vector(θ_super_free[1:5], superspreader_spec)

        η_super_free = uncoloured_mtbd2_superspreader_unconstrained_from_free(θ_super_free, superspreader_spec)
        @test uncoloured_mtbd2_superspreader_free_from_unconstrained(η_super_free, superspreader_spec) ≈ θ_super_free
        @test_throws ArgumentError uncoloured_mtbd2_superspreader_unconstrained_from_free([0.0], UncolouredMTBD2SuperspreaderSpec(superspreader_pars; total_R0=true, superspreader_fraction=false, relative_transmissibility=false, death=falses(2), sampling=falses(2), removal_probability=falses(2), ρ₀=falses(2)))

        @test loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_super_free, tree, superspreader_spec, tip_states;
                                                                     root_prior=[0.6, 0.4],
                                                                     steps_per_unit=64,
                                                                     min_steps=8) ≈
              loglikelihood_uncoloured_mtbd2(tree, native_from_superspreader, tip_states;
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8)
        @test total_loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_super_free, batch_trees, superspreader_spec, batch_observations;
                                                                           root_prior=[0.6, 0.4],
                                                                           steps_per_unit=64,
                                                                           min_steps=8) ≈
              total_loglikelihood_uncoloured_mtbd2(batch_trees, native_from_superspreader, batch_observations;
                                                   root_prior=[0.6, 0.4],
                                                   steps_per_unit=64,
                                                   min_steps=8)
        @test loglikelihood_uncoloured_mtbd2_superspreader_from_unconstrained(η_super_free, tree, superspreader_spec, tip_states;
                                                                              root_prior=[0.6, 0.4],
                                                                              steps_per_unit=64,
                                                                              min_steps=8) ≈
              loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_super_free, tree, superspreader_spec, tip_states;
                                                                     root_prior=[0.6, 0.4],
                                                                     steps_per_unit=64,
                                                                     min_steps=8)
        @test total_loglikelihood_uncoloured_mtbd2_superspreader_from_unconstrained(η_super_free, batch_trees, superspreader_spec, batch_observations;
                                                                                    root_prior=[0.6, 0.4],
                                                                                    steps_per_unit=64,
                                                                                    min_steps=8) ≈
              total_loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_super_free, batch_trees, superspreader_spec, batch_observations;
                                                                           root_prior=[0.6, 0.4],
                                                                           steps_per_unit=64,
                                                                           min_steps=8)

        superspreader_mle_spec = UncolouredMTBD2SuperspreaderSpec(
            superspreader_pars;
            total_R0=true,
            superspreader_fraction=false,
            relative_transmissibility=true,
            death=falses(2),
            sampling=falses(2),
            removal_probability=falses(2),
            ρ₀=falses(2),
        )
        θ_super_mle0 = free_parameter_vector(superspreader_pars, superspreader_mle_spec)
        η_super_mle0 = uncoloured_mtbd2_superspreader_unconstrained_from_free(θ_super_mle0, superspreader_mle_spec)
        initial_super_nll = -total_loglikelihood_uncoloured_mtbd2_superspreader_from_unconstrained(
            η_super_mle0, batch_trees, superspreader_mle_spec, batch_observations;
            root_prior=[0.6, 0.4], steps_per_unit=64, min_steps=8,
        )
        superspreader_fit = fit_uncoloured_mtbd2_superspreader_mle(
            batch_trees, superspreader_pars, superspreader_mle_spec, batch_observations;
            root_prior=[0.6, 0.4], steps_per_unit=64, min_steps=8,
            initial_step=0.02, tolerance=1e-4, maxiter=20,
        )
        @test isfinite(superspreader_fit.maximum_loglikelihood)
        @test superspreader_fit.minimum_negloglikelihood <= initial_super_nll + 1e-8
        @test superspreader_fit.initial_negloglikelihood ≈ initial_super_nll
        @test superspreader_fit.θ_free_hat ≈ free_parameter_vector(superspreader_fit.superspreader_params_hat, superspreader_mle_spec)
        @test uncoloured_mtbd2_parameter_vector(superspreader_fit.params_hat) ≈
              uncoloured_mtbd2_parameter_vector(uncoloured_mtbd2_native_parameters(superspreader_fit.superspreader_params_hat))
        superspreader_fit_from_η = fit_uncoloured_mtbd2_superspreader_mle(
            tree, η_super_mle0, superspreader_mle_spec, tip_states;
            init_is_unconstrained=true, root_prior=[0.6, 0.4],
            steps_per_unit=64, min_steps=8, initial_step=0.02,
            tolerance=1e-4, maxiter=5,
        )
        @test isfinite(superspreader_fit_from_η.minimum_negloglikelihood)
        no_free_superspreader_spec = UncolouredMTBD2SuperspreaderSpec(
            superspreader_pars;
            total_R0=false,
            superspreader_fraction=false,
            relative_transmissibility=false,
            death=falses(2),
            sampling=falses(2),
            removal_probability=falses(2),
            ρ₀=falses(2),
        )
        @test_throws ArgumentError fit_uncoloured_mtbd2_superspreader_mle(tree, superspreader_pars, no_free_superspreader_spec, tip_states)

        # Packet-12 diagnostic caveat: in the zero-transition superspreader map,
        # unknown sampled-node states can make superspreader coordinates weakly
        # identified. These checks are qualitative slice diagnostics, not formal
        # identifiability claims.
        function superspreader_with(base; total_R0=base.total_R0,
                                    superspreader_fraction=base.superspreader_fraction,
                                    relative_transmissibility=base.relative_transmissibility)
            return UncolouredMTBD2SuperspreaderParameters(
                total_R0,
                superspreader_fraction,
                relative_transmissibility,
                base.death,
                base.sampling,
                base.removal_probability,
                base.ρ₀,
            )
        end
        function superspreader_slice(grid, coordinate::Symbol, observations)
            return [
                total_loglikelihood_uncoloured_mtbd2_superspreader(
                    batch_trees,
                    coordinate === :total_R0 ? superspreader_with(superspreader_pars; total_R0=x) :
                    coordinate === :superspreader_fraction ? superspreader_with(superspreader_pars; superspreader_fraction=x) :
                    coordinate === :relative_transmissibility ? superspreader_with(superspreader_pars; relative_transmissibility=x) :
                    error("unexpected coordinate"),
                    observations;
                    root_prior=[0.6, 0.4],
                    steps_per_unit=64,
                    min_steps=8,
                )
                for x in grid
            ]
        end
        diagnostic_known = [
            Dict(3 => 1, 4 => 2),
            Dict(3 => 2, 4 => 1),
            Dict(3 => 1, 4 => 2, 5 => 1),
        ]
        diagnostic_unknown = [
            Dict(3 => missing, 4 => missing),
            Dict(3 => missing, 4 => missing),
            Dict(3 => missing, 4 => missing, 5 => missing),
        ]
        diagnostic_mixed = [
            Dict(3 => 1, 4 => missing),
            Dict(3 => missing, 4 => 1),
            Dict(3 => 1, 4 => missing, 5 => 1),
        ]
        R0_grid = collect(0.8:0.25:2.8)
        q_grid = collect(0.05:0.05:0.55)
        rel_grid = collect(1.0:1.0:10.0)
        for (grid, coordinate) in ((R0_grid, :total_R0),
                                   (q_grid, :superspreader_fraction),
                                   (rel_grid, :relative_transmissibility))
            known_values = superspreader_slice(grid, coordinate, diagnostic_known)
            unknown_values = superspreader_slice(grid, coordinate, diagnostic_unknown)
            mixed_values = superspreader_slice(grid, coordinate, diagnostic_mixed)
            @test all(isfinite, known_values)
            @test all(isfinite, unknown_values)
            @test all(isfinite, mixed_values)
            @test maximum(known_values) - minimum(known_values) >=
                  maximum(unknown_values) - minimum(unknown_values) - 1e-8
        end
        unknown_R0_values = superspreader_slice(R0_grid, :total_R0, diagnostic_unknown)
        @test 1 < argmax(unknown_R0_values) < length(unknown_R0_values)
        @test maximum(unknown_R0_values) - minimum(unknown_R0_values) > 0.25

        swap_mtbd2_pars(pars) = UncolouredMTBD2ConstantParameters(
            pars.birth[[2, 1], [2, 1]],
            pars.death[[2, 1]],
            pars.sampling[[2, 1]],
            pars.removal_probability[[2, 1]],
            pars.transition[[2, 1], [2, 1]],
            pars.ρ₀[[2, 1]],
        )
        swap_mtbd2_obs(obs::Integer) = 3 - obs
        swap_mtbd2_obs(::Missing) = missing
        swap_mtbd2_obs(::Nothing) = nothing
        swap_mtbd2_obs(::Colon) = (:)
        swap_mtbd2_obs(obs::Tuple) = tuple((swap_mtbd2_obs(x) for x in obs)...)
        swap_mtbd2_obs(obs::AbstractVector{Bool}) = reverse(obs)
        swap_mtbd2_obs(obs::AbstractVector{<:Integer}) = [swap_mtbd2_obs(x) for x in obs]
        swap_mtbd2_observations(obs::AbstractDict) = Dict(k => swap_mtbd2_obs(v) for (k, v) in obs)

        swapped_pars = swap_mtbd2_pars(mtbd2_pars)
        symmetry_obs = Dict(3 => 1, 4 => (1, 2))
        symmetry_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, symmetry_obs;
                                                     root_prior=[0.65, 0.35],
                                                     steps_per_unit=64,
                                                     min_steps=8)
        swapped_symmetry_ll = loglikelihood_uncoloured_mtbd2(tree, swapped_pars, swap_mtbd2_observations(symmetry_obs);
                                                             root_prior=[0.35, 0.65],
                                                             steps_per_unit=64,
                                                             min_steps=8)
        @test swapped_symmetry_ll ≈ symmetry_ll

        sampled_unary_symmetry_ll = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars,
                                                                   Dict(3 => 1, 4 => [true, true], 5 => 2);
                                                                   root_prior=[0.65, 0.35],
                                                                   steps_per_unit=64,
                                                                   min_steps=8)
        swapped_sampled_unary_symmetry_ll = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, swapped_pars,
                                                                           swap_mtbd2_observations(Dict(3 => 1, 4 => [true, true], 5 => 2));
                                                                           root_prior=[0.35, 0.65],
                                                                           steps_per_unit=64,
                                                                           min_steps=8)
        @test swapped_sampled_unary_symmetry_ll ≈ sampled_unary_symmetry_ll

        one_type_pars = UncolouredMTBD2ConstantParameters(
            [1.05 0.0; 0.0 1.05],
            [0.25, 0.25],
            [0.5, 0.5],
            [0.65, 0.65],
            zeros(2, 2),
            [0.2, 0.2],
        )
        one_type_binary_unknown = loglikelihood_uncoloured_mtbd2(tree, one_type_pars, Dict(3 => missing, 4 => missing);
                                                                 root_prior=[0.5, 0.5],
                                                                 steps_per_unit=64,
                                                                 min_steps=8)
        one_type_binary_known1 = loglikelihood_uncoloured_mtbd2(tree, one_type_pars, Dict(3 => 1, 4 => 1);
                                                                root_prior=[1.0, 0.0],
                                                                steps_per_unit=64,
                                                                min_steps=8)
        one_type_binary_known2 = loglikelihood_uncoloured_mtbd2(tree, one_type_pars, Dict(3 => 2, 4 => 2);
                                                                root_prior=[0.0, 1.0],
                                                                steps_per_unit=64,
                                                                min_steps=8)
        @test one_type_binary_unknown ≈ one_type_binary_known1
        @test one_type_binary_known1 ≈ one_type_binary_known2

        one_type_sampled_unary_unknown = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, one_type_pars,
                                                                        Dict(3 => missing, 4 => missing, 5 => missing);
                                                                        root_prior=[0.5, 0.5],
                                                                        steps_per_unit=64,
                                                                        min_steps=8)
        one_type_sampled_unary_known1 = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, one_type_pars,
                                                                       Dict(3 => 1, 4 => 1, 5 => 1);
                                                                       root_prior=[1.0, 0.0],
                                                                       steps_per_unit=64,
                                                                       min_steps=8)
        @test one_type_sampled_unary_unknown ≈ one_type_sampled_unary_known1

        unknown_missing_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => missing, 4 => missing);
                                                           root_prior=[0.6, 0.4],
                                                           steps_per_unit=64,
                                                           min_steps=8)
        unknown_nothing_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => nothing, 4 => nothing);
                                                           root_prior=[0.6, 0.4],
                                                           steps_per_unit=64,
                                                           min_steps=8)
        unknown_colon_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => :, 4 => :);
                                                         root_prior=[0.6, 0.4],
                                                         steps_per_unit=64,
                                                         min_steps=8)
        unknown_mask_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => [true, true], 4 => [true, true]);
                                                        root_prior=[0.6, 0.4],
                                                        steps_per_unit=64,
                                                        min_steps=8)
        singleton_mask_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => [true, false], 4 => [false, true]);
                                                          root_prior=[0.6, 0.4],
                                                          steps_per_unit=64,
                                                          min_steps=8)
        @test unknown_missing_ll ≈ unknown_nothing_ll
        @test unknown_missing_ll ≈ unknown_colon_ll
        @test unknown_missing_ll ≈ unknown_mask_ll
        @test singleton_mask_ll ≈ ll

        function scaled_tree(base_tree, scale)
            return Tree(
                base_tree.time .* scale,
                copy(base_tree.left),
                copy(base_tree.right),
                copy(base_tree.parent),
                copy(base_tree.kind),
                copy(base_tree.host),
                copy(base_tree.label),
            )
        end
        stability_parameters = (
            mtbd2_pars,
            UncolouredMTBD2ConstantParameters(
                [0.15 0.03; 0.02 0.12],
                [0.01, 0.015],
                [0.02, 0.025],
                [0.2, 0.8],
                [0.0 0.01; 0.015 0.0],
                [0.05, 0.08],
            ),
            UncolouredMTBD2ConstantParameters(
                [2.5 0.4; 0.35 2.0],
                [0.7, 0.5],
                [1.1, 0.9],
                [0.95, 0.15],
                [0.0 0.3; 0.25 0.0],
                [0.4, 0.3],
            ),
        )
        for pars_i in stability_parameters, scale in (0.25, 1.0, 3.0)
            binary_scaled = scaled_tree(tree, scale)
            unary_scaled = scaled_tree(sampled_unary_tree, scale)
            coarse_binary = loglikelihood_uncoloured_mtbd2(binary_scaled, pars_i, Dict(3 => missing, 4 => 2);
                                                           root_prior=[0.55, 0.45],
                                                           steps_per_unit=48,
                                                           min_steps=8)
            fine_binary = loglikelihood_uncoloured_mtbd2(binary_scaled, pars_i, Dict(3 => missing, 4 => 2);
                                                         root_prior=[0.55, 0.45],
                                                         steps_per_unit=96,
                                                         min_steps=16)
            coarse_unary = loglikelihood_uncoloured_mtbd2(unary_scaled, pars_i, Dict(3 => 1, 4 => missing, 5 => 2);
                                                          root_prior=[0.55, 0.45],
                                                          steps_per_unit=48,
                                                          min_steps=8)
            fine_unary = loglikelihood_uncoloured_mtbd2(unary_scaled, pars_i, Dict(3 => 1, 4 => missing, 5 => 2);
                                                        root_prior=[0.55, 0.45],
                                                        steps_per_unit=96,
                                                        min_steps=16)
            @test isfinite(coarse_binary)
            @test isfinite(fine_binary)
            @test isfinite(coarse_unary)
            @test isfinite(fine_unary)
            @test coarse_binary ≈ fine_binary atol=0.05
            @test coarse_unary ≈ fine_unary atol=0.05
        end

        sim_true = UncolouredMTBD2ConstantParameters(
            [1.25 0.25; 0.12 0.85],
            [0.18, 0.28],
            [0.75, 0.55],
            [0.55, 0.35],
            [0.0 0.08; 0.05 0.0],
            [0.25, 0.20],
        )
        sim_perturbed = UncolouredMTBD2ConstantParameters(
            [0.65 0.08; 0.30 1.45],
            [0.35, 0.12],
            [0.35, 0.95],
            [0.20, 0.80],
            [0.0 0.22; 0.01 0.0],
            [0.08, 0.35],
        )
        multitype_for_validation(pars) = MultitypeBDParameters(
            pars.birth,
            pars.death,
            pars.sampling,
            pars.removal_probability,
            pars.transition,
            pars.ρ₀,
        )
        function plain_validation_eventlog(log::MultitypeBDEventLog)
            times = Float64[]
            lineages = Int[]
            parents = Int[]
            kinds = BDEventKind[]
            for i in 1:length(log)
                if log.kind[i] == MultitypeBirth
                    push!(times, log.time[i]); push!(lineages, log.lineage[i]); push!(parents, log.parent[i]); push!(kinds, Birth)
                elseif log.kind[i] == MultitypeDeath
                    push!(times, log.time[i]); push!(lineages, log.lineage[i]); push!(parents, log.parent[i]); push!(kinds, Death)
                elseif log.kind[i] == MultitypeFossilizedSampling
                    push!(times, log.time[i]); push!(lineages, log.lineage[i]); push!(parents, log.parent[i]); push!(kinds, FossilizedSampling)
                elseif log.kind[i] == MultitypeSerialSampling
                    push!(times, log.time[i]); push!(lineages, log.lineage[i]); push!(parents, log.parent[i]); push!(kinds, SerialSampling)
                end
            end
            return BDEventLog(times, lineages, parents, kinds, length(log.initial_types), log.tmax)
        end
        function sampled_validation_observations(tree::TreeSim.Tree, log::MultitypeBDEventLog)
            observations = Dict{Int,Int}()
            for node in eachindex(tree)
                tree.kind[node] in (TreeSim.SampledLeaf, TreeSim.SampledUnary) || continue
                event = findfirst(i -> log.lineage[i] == tree.host[node] &&
                                       isapprox(log.time[i], tree.time[node]; atol=1e-10, rtol=0.0) &&
                                       log.kind[i] in (MultitypeFossilizedSampling, MultitypeSerialSampling),
                                  1:length(log))
                event === nothing && error("missing sampled state for validation node $node")
                observations[node] = log.type_before[event]
            end
            return observations
        end
        validation_rng = MersenneTwister(20260418)
        validation_rows = []
        validation_attempts = 0
        while validation_attempts < 40 && length(validation_rows) < 8
            validation_attempts += 1
            log = simulate_multitype_bd(validation_rng, multitype_for_validation(sim_true), 2.0;
                                        initial_types=[1], apply_ρ₀=true)
            forest = forest_from_eventlog(plain_validation_eventlog(log); tj=0.0, tk=log.tmax)
            length(forest) == 1 || continue
            simulated_tree = only(forest)
            all(kind -> kind in (TreeSim.Root, TreeSim.Binary, TreeSim.SampledLeaf, TreeSim.SampledUnary), simulated_tree.kind) || continue
            known_obs = sampled_validation_observations(simulated_tree, log)
            unknown_obs = Dict(node => missing for node in keys(known_obs))
            mixed_obs = Dict(node => (isodd(node) ? state : missing) for (node, state) in known_obs)
            true_known = loglikelihood_uncoloured_mtbd2(simulated_tree, sim_true, known_obs)
            perturbed_known = loglikelihood_uncoloured_mtbd2(simulated_tree, sim_perturbed, known_obs)
            true_unknown = loglikelihood_uncoloured_mtbd2(simulated_tree, sim_true, unknown_obs)
            true_mixed = loglikelihood_uncoloured_mtbd2(simulated_tree, sim_true, mixed_obs)
            push!(validation_rows, (
                tree=simulated_tree,
                true_known=true_known,
                perturbed_known=perturbed_known,
                true_unknown=true_unknown,
                true_mixed=true_mixed,
            ))
        end
        @test length(validation_rows) >= 6
        @test any(row -> any(==(TreeSim.SampledUnary), row.tree.kind), validation_rows)
        @test all(row -> isfinite(row.true_known) &&
                         isfinite(row.perturbed_known) &&
                         isfinite(row.true_unknown) &&
                         isfinite(row.true_mixed),
                  validation_rows)
        @test sum(row.true_known for row in validation_rows) >
              sum(row.perturbed_known for row in validation_rows)

        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => 1))
        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => 1, 4 => 3))
        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => 1, 4 => Int[]))
        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => 1, 4 => [true, false, true]))
        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars, Dict(4 => 2, 5 => 1))
        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(unsampled_unary_tree(), mtbd2_pars, Dict(3 => 1))
        @test_throws ArgumentError UncolouredMTBD2ConstantParameters([0.1 0.2], [0.1, 0.1], [0.1, 0.1], [1.0, 1.0], zeros(2, 2))
    end

    @testset "multitype event-log to colored-tree bridge" begin
        bridge_log = MultitypeBDEventLog(
            [0.2, 0.4, 0.5, 0.7, 1.0],
            [2, 1, 2, 1, 2],
            [1, 0, 0, 0, 0],
            [MultitypeBirth, MultitypeTransition, MultitypeFossilizedSampling,
             MultitypeSerialSampling, MultitypeSerialSampling],
            [1, 1, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [1],
            1.0,
        )
        bridge_tree = multitype_colored_tree_from_eventlog(bridge_log)
        @test bridge_tree.origin_time == 1.0
        @test bridge_tree.root_type == 1
        @test [segment.type for segment in bridge_tree.segments] == [1, 1, 2, 2, 2]
        @test [segment.start_time for segment in bridge_tree.segments] ≈ [0.8, 0.6, 0.5, 0.3, 0.0]
        @test [segment.end_time for segment in bridge_tree.segments] ≈ [1.0, 0.8, 0.8, 0.6, 0.5]
        @test bridge_tree.births == [MultitypeColoredBirth(0.8, 1, 2)]
        @test bridge_tree.transitions == [MultitypeColoredTransition(0.6, 1, 2)]
        @test bridge_tree.ancestral_samples == [MultitypeColoredSampling(0.5, 2)]
        @test [sample.time for sample in bridge_tree.terminal_samples] ≈ [0.3]
        @test [sample.type for sample in bridge_tree.terminal_samples] == [2]
        @test bridge_tree.present_samples == [2]

        bridge_pars = MultitypeBDParameters(
            [0.4 0.8; 0.2 0.5],
            [0.1, 0.2],
            [0.3, 0.6],
            [0.2, 0.7],
            [0.0 0.9; 0.1 0.0],
            [0.4, 0.5],
        )
        @test validate_multitype_colored_tree(bridge_tree, bridge_pars)
        checked_bridge_tree = validate_multitype_colored_tree_from_eventlog(bridge_log, bridge_pars)
        @test checked_bridge_tree.origin_time == bridge_tree.origin_time
        @test checked_bridge_tree.root_type == bridge_tree.root_type
        @test length(checked_bridge_tree.segments) == length(bridge_tree.segments)
        @test isfinite(multitype_colored_loglikelihood(bridge_tree, bridge_pars))
        @test multitype_colored_likelihood(bridge_tree, bridge_pars) > 0

        terminal_at_tmax = multitype_colored_tree_from_eventlog(bridge_log; serial_at_tmax=:terminal)
        @test terminal_at_tmax.present_samples == Int[]
        @test terminal_at_tmax.terminal_samples[end] == MultitypeColoredSampling(0.0, 2)

        k1_bridge_log = MultitypeBDEventLog(
            [1.0],
            [1],
            [0],
            [MultitypeSerialSampling],
            [1],
            [1],
            [1],
            1.0,
        )
        k1_bridge_tree = multitype_colored_tree_from_eventlog(k1_bridge_log)
        @test k1_bridge_tree.segments == [MultitypeColoredSegment(1, 0.0, 1.0)]
        @test k1_bridge_tree.present_samples == [1]
        k1_bridge_pars = MultitypeBDParameters([0.2;;], [0.1], [0.0], [0.0], [0.0;;], [0.8])
        @test multitype_colored_loglikelihood(k1_bridge_tree, k1_bridge_pars) ≈
              log(0.8) + only(multitype_log_flow(1.0, k1_bridge_pars))

        death_log = MultitypeBDEventLog([0.5], [1], [0], [MultitypeDeath], [1], [1], [1], 1.0)
        @test_throws ArgumentError multitype_colored_tree_from_eventlog(death_log)
        survivor_log = MultitypeBDEventLog(Float64[], Int[], Int[], MultitypeBDEventKind[], Int[], Int[], [1], 1.0)
        @test_throws ArgumentError multitype_colored_tree_from_eventlog(survivor_log)
        multi_root_log = MultitypeBDEventLog([1.0, 1.0], [1, 2], [0, 0],
                                             [MultitypeSerialSampling, MultitypeSerialSampling],
                                             [1, 1], [1, 1], [1, 1], 1.0)
        @test_throws ArgumentError multitype_colored_tree_from_eventlog(multi_root_log)
        @test_throws ArgumentError multitype_colored_tree_from_eventlog(bridge_log; serial_at_tmax=:unknown)
    end

    @testset "multitype pruned event-log extraction" begin
        pruned_log = MultitypeBDEventLog(
            [0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0],
            [2, 3, 3, 1, 1, 1, 2],
            [1, 1, 0, 0, 0, 0, 0],
            [MultitypeBirth, MultitypeBirth, MultitypeDeath, MultitypeTransition,
             MultitypeFossilizedSampling, MultitypeSerialSampling, MultitypeSerialSampling],
            [1, 1, 2, 1, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [1],
            1.0,
        )
        pruned_tree = pruned_multitype_colored_tree_from_eventlog(pruned_log)
        @test pruned_tree.origin_time == 1.0
        @test pruned_tree.root_type == 1
        @test [segment.type for segment in pruned_tree.segments] == [1, 1, 2, 2, 2]
        @test [segment.start_time for segment in pruned_tree.segments] ≈ [0.8, 0.5, 0.35, 0.2, 0.0]
        @test [segment.end_time for segment in pruned_tree.segments] ≈ [1.0, 0.8, 0.5, 0.35, 0.8]
        @test pruned_tree.births == [MultitypeColoredBirth(0.8, 1, 2)]
        @test pruned_tree.hidden_births == MultitypeColoredHiddenBirth[]
        @test pruned_tree.transitions == [MultitypeColoredTransition(0.5, 1, 2)]
        @test pruned_tree.ancestral_samples == [MultitypeColoredSampling(0.35, 2)]
        @test [sample.time for sample in pruned_tree.terminal_samples] ≈ [0.2]
        @test pruned_tree.present_samples == [2]

        pruned_pars = MultitypeBDParameters(
            [0.4 0.8; 0.2 0.5],
            [0.1, 0.2],
            [0.3, 0.6],
            [0.2, 0.7],
            [0.0 0.9; 0.1 0.0],
            [0.4, 0.5],
        )
        @test validate_multitype_colored_tree(pruned_tree, pruned_pars)
        @test validate_pruned_multitype_colored_tree_from_eventlog(pruned_log, pruned_pars).root_type == 1
        @test isfinite(multitype_colored_loglikelihood(pruned_tree, pruned_pars))

        terminal_fossil_log = MultitypeBDEventLog(
            [0.4, 0.9],
            [1, 1],
            [0, 0],
            [MultitypeFossilizedSampling, MultitypeDeath],
            [1, 1],
            [1, 1],
            [1],
            1.0,
        )
        terminal_fossil_tree = pruned_multitype_colored_tree_from_eventlog(terminal_fossil_log)
        @test terminal_fossil_tree.ancestral_samples == MultitypeColoredSampling[]
        @test terminal_fossil_tree.terminal_samples == [MultitypeColoredSampling(0.6, 1)]
        @test terminal_fossil_tree.segments == [MultitypeColoredSegment(1, 0.6, 1.0)]

        k1_pruned_log = MultitypeBDEventLog(
            [0.2, 0.4, 0.6],
            [2, 2, 1],
            [1, 0, 0],
            [MultitypeBirth, MultitypeDeath, MultitypeSerialSampling],
            [1, 1, 1],
            [1, 1, 1],
            [1],
            1.0,
        )
        k1_pruned_tree = pruned_multitype_colored_tree_from_eventlog(k1_pruned_log)
        @test k1_pruned_tree.births == MultitypeColoredBirth[]
        @test k1_pruned_tree.hidden_births == MultitypeColoredHiddenBirth[]
        @test k1_pruned_tree.terminal_samples == [MultitypeColoredSampling(0.4, 1)]
        @test k1_pruned_tree.segments == [MultitypeColoredSegment(1, 0.4, 1.0)]

        child_only_log = MultitypeBDEventLog(
            [0.2, 0.5, 0.8],
            [2, 1, 2],
            [1, 0, 0],
            [MultitypeBirth, MultitypeDeath, MultitypeSerialSampling],
            [1, 1, 2],
            [2, 1, 2],
            [1],
            1.0,
        )
        child_only_tree = pruned_multitype_colored_tree_from_eventlog(child_only_log)
        @test child_only_tree.births == MultitypeColoredBirth[]
        @test child_only_tree.hidden_births == [MultitypeColoredHiddenBirth(0.8, 1, 2)]
        @test [segment.type for segment in child_only_tree.segments] == [1, 2]
        @test [segment.start_time for segment in child_only_tree.segments] ≈ [0.8, 0.2]
        @test [segment.end_time for segment in child_only_tree.segments] ≈ [1.0, 0.8]
        @test [sample.time for sample in child_only_tree.terminal_samples] ≈ [0.2]
        @test [sample.type for sample in child_only_tree.terminal_samples] == [2]
        child_only_pars = MultitypeBDParameters([0.2 0.7; 0.1 0.3], [0.2, 0.2], [0.4, 0.5],
                                                [0.5, 0.5], [0.0 0.1; 0.1 0.0], [0.0, 0.0])
        @test isfinite(multitype_colored_loglikelihood(child_only_tree, child_only_pars))

        unsampled_log = MultitypeBDEventLog([0.5], [1], [0], [MultitypeDeath], [1], [1], [1], 1.0)
        @test_throws ArgumentError pruned_multitype_colored_tree_from_eventlog(unsampled_log)
        @test_throws ArgumentError pruned_multitype_colored_tree_from_eventlog(pruned_log; serial_at_tmax=:unknown)
    end

    @testset "multitype scoring and fitting wrappers" begin
        score_pars = MultitypeBDParameters(
            [0.4 0.8; 0.2 0.5],
            [0.1, 0.2],
            [0.3, 0.6],
            [0.2, 0.7],
            [0.0 0.9; 0.1 0.0],
            [0.4, 0.5],
        )
        score_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.8, 1.0),
                MultitypeColoredSegment(1, 0.5, 0.8),
                MultitypeColoredSegment(2, 0.0, 0.8),
            ],
            births=[MultitypeColoredBirth(0.8, 1, 2)],
            transitions=[MultitypeColoredTransition(0.5, 1, 2)],
            present_samples=[2],
        )
        raw_ll = multitype_colored_loglikelihood(score_tree, score_pars)
        @test multitype_loglikelihood(score_tree, score_pars) ≈ raw_ll
        @test multitype_loglikelihood([score_tree, score_tree], score_pars) ≈ 2raw_ll

        spec = MultitypeMLESpec(
            score_pars;
            fit_birth=[false true; false false],
            fit_death=[false, false],
            fit_sampling=[false, false],
            fit_transition=[false false; false false],
        )
        θ = multitype_pack_parameters(score_pars, spec)
        @test length(θ) == 1
        unpacked = multitype_unpack_parameters(θ, spec)
        @test unpacked.birth == score_pars.birth
        @test unpacked.death == score_pars.death
        @test unpacked.sampling == score_pars.sampling
        @test unpacked.transition == score_pars.transition
        @test multitype_negloglikelihood(θ, score_tree, spec) ≈ -raw_ll
        @test multitype_negloglikelihood(θ, [score_tree, score_tree], spec) ≈ -2raw_ll

        lower_birth = MultitypeBDParameters(
            [0.4 0.2; 0.2 0.5],
            score_pars.death,
            score_pars.sampling,
            score_pars.removal_probability,
            score_pars.transition,
            score_pars.ρ₀,
        )
        fit_spec = MultitypeMLESpec(
            lower_birth;
            fit_birth=[false true; false false],
            fit_death=[false, false],
            fit_sampling=[false, false],
            fit_transition=[false false; false false],
        )
        before = multitype_loglikelihood(score_tree, lower_birth)
        fit = fit_multitype_mle(
            score_tree;
            spec=fit_spec,
            lower=[log(0.05)],
            upper=[log(5.0)],
            initial_step=0.5,
            tolerance=1e-4,
            maxiter=200,
        )
        @test fit.loglikelihood >= before
        @test fit.parameters.birth[1, 2] > lower_birth.birth[1, 2]
        @test fit.result.converged || fit.result.iterations == 200

        k1_score_pars = MultitypeBDParameters([0.2;;], [0.1], [0.4], [0.5], [0.0;;], [0.8])
        k1_score_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.0, 1.0)],
            present_samples=[1],
        )
        @test multitype_loglikelihood(k1_score_tree, k1_score_pars) ≈
              multitype_colored_loglikelihood(k1_score_tree, k1_score_pars)

        @test_throws ArgumentError multitype_loglikelihood(MultitypeColoredTree[], score_pars)
        @test_throws ArgumentError multitype_unpack_parameters(Float64[], spec)
        @test_throws ArgumentError MultitypeMLESpec(score_pars; fit_birth=[true false])
        zero_support_pars = MultitypeBDParameters([0.0 0.8; 0.2 0.5], score_pars.death,
                                                  score_pars.sampling, score_pars.removal_probability,
                                                  score_pars.transition, score_pars.ρ₀)
        @test_throws ArgumentError MultitypeMLESpec(zero_support_pars; fit_birth=[true false; false false],
                                                         fit_death=[false, false],
                                                         fit_sampling=[false, false],
                                                         fit_transition=[false false; false false])
        @test_throws ArgumentError fit_multitype_mle(MultitypeColoredTree[]; spec=spec)
        no_fit_spec = MultitypeMLESpec(
            score_pars;
            fit_birth=falses(2, 2),
            fit_death=falses(2),
            fit_sampling=falses(2),
            fit_transition=falses(2, 2),
        )
        @test_throws ArgumentError fit_multitype_mle(score_tree; spec=no_fit_spec)

        rng_fit = MersenneTwister(20260418)
        sim_fit_pars = MultitypeBDParameters([0.0 0.9; 0.0 0.0], [0.15, 0.25], [0.05, 0.8],
                                             [1.0, 1.0], zeros(2, 2), [0.0, 0.0])
        extracted = MultitypeColoredTree[]
        attempts = 0
        while length(extracted) < 2 && attempts < 200
            attempts += 1
            log = simulate_multitype_bd(rng_fit, sim_fit_pars, 1.0; initial_types=[1], apply_ρ₀=false)
            try
                push!(extracted, pruned_multitype_colored_tree_from_eventlog(log))
            catch err
                err isa ArgumentError || rethrow()
            end
        end
        @test length(extracted) == 2
        @test multitype_loglikelihood(extracted, sim_fit_pars) ≈
              sum(tree -> multitype_loglikelihood(tree, sim_fit_pars), extracted)

        perturbed_fit_pars = MultitypeBDParameters([0.0 0.25; 0.0 0.0], sim_fit_pars.death,
                                                   sim_fit_pars.sampling, sim_fit_pars.removal_probability,
                                                   sim_fit_pars.transition, sim_fit_pars.ρ₀)
        extracted_spec = MultitypeMLESpec(perturbed_fit_pars;
                                          fit_birth=[false true; false false],
                                          fit_death=falses(2),
                                          fit_sampling=falses(2),
                                          fit_transition=falses(2, 2))
        extracted_before = multitype_loglikelihood(extracted, perturbed_fit_pars)
        extracted_fit = fit_multitype_mle(extracted; spec=extracted_spec, lower=[log(0.05)], upper=[log(5.0)],
                                          initial_step=0.5, tolerance=1e-4, maxiter=200)
        @test extracted_fit.loglikelihood >= extracted_before
    end

    @testset "multitype validation comparisons" begin
        validation_pars = MultitypeBDParameters(
            [0.15 0.08; 0.04 0.12],
            [0.35, 0.25],
            [0.20, 0.30],
            [0.70, 0.40],
            [0.0 0.10; 0.06 0.0],
            [0.15, 0.25],
        )
        t_validation = 0.8
        analytical_E = multitype_E(t_validation, validation_pars; steps_per_unit=512)
        rng = MersenneTwister(90417)
        nsims = 2_500
        for initial_type in 1:2
            no_observed = 0
            for _ in 1:nsims
                log = simulate_multitype_bd(rng, validation_pars, t_validation; initial_types=[initial_type])
                no_observed += sum(multitype_S_at(log, t_validation)) == 0
            end
            @test no_observed / nsims ≈ analytical_E[initial_type] atol=0.04
        end

        k1_validation = MultitypeBDParameters([0.45;;], [0.30], [0.25], [0.60], [0.0;;], [0.20])
        k1_single = ConstantRateBDParameters(0.45, 0.30, 0.25, 0.60, 0.20)
        @test only(multitype_E(1.1, k1_validation; steps_per_unit=1024)) ≈ E_constant(1.1, k1_single) atol=2e-6
        rng_k1 = MersenneTwister(90418)
        k1_no_observed = 0
        for _ in 1:nsims
            log = simulate_multitype_bd(rng_k1, k1_validation, 1.1)
            k1_no_observed += sum(multitype_S_at(log, 1.1)) == 0
        end
        @test k1_no_observed / nsims ≈ E_constant(1.1, k1_single) atol=0.04

        flow_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.25, 1.0)],
            terminal_samples=[MultitypeColoredSampling(0.25, 1)],
        )
        logflow_025 = multitype_log_flow(0.25, validation_pars)
        logflow_100 = multitype_log_flow(1.0, validation_pars)
        E025 = multitype_E(0.25, validation_pars)[1]
        expected_flow_tree = (logflow_100[1] - logflow_025[1]) +
                             log(validation_pars.sampling[1] *
                                 (validation_pars.removal_probability[1] +
                                  (1 - validation_pars.removal_probability[1]) * E025))
        @test multitype_colored_loglikelihood(flow_tree, validation_pars) ≈ expected_flow_tree
        @test multitype_colored_likelihood(flow_tree, validation_pars) ≈ exp(expected_flow_tree)

        transition_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.5, 1.0),
                MultitypeColoredSegment(2, 0.0, 0.5),
            ],
            transitions=[MultitypeColoredTransition(0.5, 1, 2)],
            present_samples=[2],
        )
        higher_transition = MultitypeBDParameters(
            validation_pars.birth,
            validation_pars.death,
            validation_pars.sampling,
            validation_pars.removal_probability,
            [0.0 0.25; 0.06 0.0],
            validation_pars.ρ₀,
        )
        @test multitype_colored_loglikelihood(transition_tree, higher_transition) >
              multitype_colored_loglikelihood(transition_tree, validation_pars)
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
