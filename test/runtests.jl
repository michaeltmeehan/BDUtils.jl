using Test
using BDUtils
using TreeSim

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

    @testset "PGF/probability helpers" begin
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
