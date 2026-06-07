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
