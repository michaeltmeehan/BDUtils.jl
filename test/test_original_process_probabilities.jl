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
