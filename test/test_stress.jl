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
