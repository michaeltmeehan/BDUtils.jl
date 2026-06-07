    @testset "extinction and survival helpers" begin
        λ0, μ0, ψ0, r0 = pars.λ, pars.μ, pars.ψ, pars.r

        @test E_constant(0.0, λ0, μ0, ψ0) == 1.0
        @test E_constant(0.0, λ0, μ0, ψ0; ρ₀=pars.ρ₀) == 1 - pars.ρ₀
        @test E_constant(0.0, pars) == 0.75
        @test g_constant(0.0, λ0, μ0, ψ0) == 0.0
        @test g_constant(0.0, pars) == 0.0

        E1 = E_constant(1.0, λ0, μ0, ψ0)
        g1 = g_constant(1.0, λ0, μ0, ψ0)
        @test E_constant(1.0, pars) ≈ E_constant(1.0, λ0, μ0, ψ0; ρ₀=pars.ρ₀)
        @test g_constant(1.0, pars) ≈ g_constant(1.0, λ0, μ0, ψ0; ρ₀=pars.ρ₀)
        @test isfinite(E1)
        @test isfinite(g1)
        @test 0.0 <= E1 <= 1.0
        @test logaddexp(log(0.25), log(0.75)) ≈ 0.0 atol=eps(Float64)

        @test_throws ArgumentError E_constant(1.0, 0.0, μ0, ψ0)
        @test_throws ArgumentError E_constant(1.0, λ0, μ0, ψ0; ρ₀=-0.1)
        @test_throws ArgumentError g_constant(NaN, λ0, μ0, ψ0)
    end

    @testset "TreeSim likelihood benchmark and admissibility" begin
        λ0, μ0, ψ0, r0 = pars.λ, pars.μ, pars.ψ, pars.r
        tree = tiny_tree()
        @test validate_tree(tree; require_single_root=true, require_reachable=true)

        ll = bd_loglikelihood_constant(tree, λ0, μ0, ψ0, r0)
        @test isfinite(ll)
        @test ll ≈ -1.6722221934689507
        @test bd_loglikelihood_constant(tree, ConstantRateBDParameters(λ0, μ0, ψ0, r0)) ≈ ll
        @test bd_loglikelihood_constant(tree, pars) ≈ bd_loglikelihood_constant(tree, λ0, μ0, ψ0, r0; ρ₀=pars.ρ₀)

        ll_ascii = bd_loglikelihood_constant(tree, 2, 0.5, 0.4, 0.7)
        @test ll_ascii ≈ ll

        @test_throws ArgumentError bd_loglikelihood_constant(tree, 0.0, μ0, ψ0, r0)
        @test_throws ArgumentError bd_loglikelihood_constant(tree, λ0, μ0, 0.0, r0)
        @test_throws ArgumentError bd_loglikelihood_constant(tree, λ0, μ0, ψ0, -0.01)
        @test_throws ArgumentError bd_loglikelihood_constant(tree, λ0, μ0, ψ0, r0; ρ₀=1.01)

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
