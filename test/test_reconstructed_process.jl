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

        @test unsampled_probability(1.0, 1.0, pars) ≈ 1 - pars.ρ₀
        @test_throws ArgumentError unsampled_probability(2.0, 1.0, pars)
        @test_throws ArgumentError reconstructed_alpha_bd(1.0, 1.1, 1.0, 2.0, pars)
        @test_throws ArgumentError reconstructed_count_pmf(-1, 0.0, 1.0, 2.0, pars)
        @test_throws ArgumentError transformed_birth_rate(0.0, 1.0, ConstantRateBDParameters(1.0, 0.5, 0.0, 0.0))
    end

    @testset "public API: conditioned reconstructed scalar helpers" begin
        ti = 0.0
        tk = 2.4

        for reconstructed_pars in KENDALL_REGIMES
            for w in (0.0, 0.4, 1.0), tj in (0.35, 1.0, 1.7)
                pᵢ = unsampled_probability(ti, tk, reconstructed_pars)
                qᵢ = 1 - pᵢ
                raw_α = reconstructed_alpha_bd(w, ti, tj, tk, reconstructed_pars)
                raw_β = reconstructed_beta_bd(w, ti, tj, tk, reconstructed_pars)
                raw_γ = reconstructed_gamma_bd(w, ti, tj, tk, reconstructed_pars)

                @test conditioned_reconstructed_alpha_bd(w, ti, tj, tk, reconstructed_pars) ≈ (raw_α - pᵢ) / qᵢ
                @test conditioned_reconstructed_beta_bd(w, ti, tj, tk, reconstructed_pars) ≈ raw_β / qᵢ
                @test conditioned_reconstructed_gamma_bd(w, ti, tj, tk, reconstructed_pars) ≈ raw_γ

                z = 0.55
                raw_pgf = reconstructed_pgf(z, w, ti, tj, tk, reconstructed_pars)
                @test conditioned_reconstructed_pgf(z, w, ti, tj, tk, reconstructed_pars) ≈ (raw_pgf - pᵢ) / qᵢ
            end

            for tj in (0.35, 1.0, 1.7)
                ξ = conditioned_reconstructed_xi(ti, tj, tk, reconstructed_pars)
                η = conditioned_reconstructed_eta(ti, tj, tk, reconstructed_pars)
                β1 = conditioned_reconstructed_beta_bd(1.0, ti, tj, tk, reconstructed_pars)
                @test β1 ≈ (1 - ξ) * (1 - η) rtol=2e-11 atol=2e-13
                @test conditioned_reconstructed_count_pmf(0, ti, tj, tk, reconstructed_pars) ≈ ξ
                @test conditioned_reconstructed_count_pmf(1, ti, tj, tk, reconstructed_pars) ≈ (1 - ξ) * (1 - η)
                @test conditioned_reconstructed_count_pmf(4, ti, tj, tk, reconstructed_pars) ≈ (1 - ξ) * (1 - η) * η^3
                @test sum(conditioned_reconstructed_count_pmf(a, ti, tj, tk, reconstructed_pars) for a in 0:200) ≈ 1.0 atol=1e-12
            end
        end

        @test_throws ArgumentError conditioned_reconstructed_alpha_bd(1.0, 1.1, 1.0, 2.0, pars)
        @test_throws ArgumentError conditioned_reconstructed_count_pmf(-1, 0.0, 1.0, 2.0, pars)
        @test_throws ArgumentError conditioned_reconstructed_alpha_bd(1.0, 1.0, 1.0, 1.0, ConstantRateBDParameters(λ, μ, ψ, r))
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
