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

    @testset "derived helpers" begin
        @test compute_R0(λ, μ, ψ, r) == λ / (μ + ψ * r)
        @test compute_delta(λ, μ, ψ, r) == λ - (μ + ψ * r)
        @test compute_R0([λ], [μ], [ψ], r) == [λ / (μ + ψ * r)]
    end

    @testset "PGF/probability helpers" begin
        a, b, Δ = bd_coefficients(1.0, λ, μ, ψ, r)
        @test isfinite(a)
        @test isfinite(b)
        @test isfinite(Δ)
        @test Δ >= 0

        @test γ(1.0, 0.0, 0.0, λ, μ, ψ, r) == 0.0
        @test α(1.0, 0.0, 0.0, λ, μ, ψ, r) == 0.0
        @test β(1.0, 0.0, 0.0, λ, μ, ψ, r) == 1.0
        @test pₙ(0, 0.0, 0.0, λ, μ, ψ, r) == 0.0
        @test pₙ(1, 0.0, 0.0, λ, μ, ψ, r) == 1.0
        @test pₙ(2, 0.0, 0.0, λ, μ, ψ, r) == 0.0

        vals = pn_birthdeath([0, 1, 2], 0.0, 1.0, λ, μ, ψ, r)
        @test length(vals) == 3
        @test all(isfinite, vals)
        @test all(>=(0), vals)

        @test gamma_bd(1, 0, 1, λ, μ, ψ, r) ≈ γ(1.0, 0.0, 1.0, λ, μ, ψ, r)

        @test_throws ArgumentError bd_coefficients(1.0, 0.0, μ, ψ, r)
        @test_throws ArgumentError bd_coefficients(1.0, λ, -μ, ψ, r)
        @test_throws ArgumentError bd_coefficients(1.0, λ, μ, ψ, 1.1)
        @test_throws ArgumentError bd_coefficients(Inf, λ, μ, ψ, r)
    end

    @testset "extinction and survival helpers" begin
        @test E_constant(0.0, λ, μ, ψ) == 1.0
        @test E_constant(0.0, λ, μ, ψ; ρ₀=0.25) == 0.75
        @test g_constant(0.0, λ, μ, ψ) == 0.0

        E1 = E_constant(1.0, λ, μ, ψ)
        g1 = g_constant(1.0, λ, μ, ψ)
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
