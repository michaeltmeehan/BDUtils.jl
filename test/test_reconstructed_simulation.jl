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

        terminal_cases = (
            (name="subcritical_terminal_sampling", seed=44, pars=ConstantRateBDParameters(0.8, 1.0, 0.3, 0.4, 0.35), tj=0.5, tk=1.25, nsims=10_000),
            (name="supercritical_terminal_sampling", seed=45, pars=ConstantRateBDParameters(1.35, 0.55, 0.65, 0.7, 0.25), tj=0.45, tk=1.15, nsims=10_000),
        )

        for case in terminal_cases
            summary = reconstructed_validation_summary(case.seed, case.pars, case.tj, case.tk, case.nsims; tail_atol=1e-4, apply_ρ₀=true)
            assert_reconstructed_validation(summary;
                joint_tv_atol=0.035,
                marginal_tv_atol=0.03,
                maxerr_atol=0.025,
                tail_slack=0.025,
                retention_atol=0.03,
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
