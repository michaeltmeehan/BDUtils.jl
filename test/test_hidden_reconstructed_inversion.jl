@testset "hidden reconstructed inversion" begin
    local_logbinomial(n, k) = sum(log(n - min(k, n - k) + i) - log(i) for i in 1:min(k, n - k); init=0.0)
    ti = 0.0
    tj = 1.0
    tl = 2.0
    inv_pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 0.8)

    @test hidden_reconstructed_unsampled_probability(tj, tl, inv_pars) ≈
          unsampled_probability(tj, tl, inv_pars)

    @testset "a >= 1 shifted negative binomial" begin
        a = 2
        q = hidden_reconstructed_unsampled_probability(tj, tl, inv_pars)
        θ = gamma_bd(1.0, ti, tj, inv_pars) * q
        success = 1 - θ

        probs = [hidden_count_given_reconstructed_count_pmf(n, a, ti, tj, tl, inv_pars) for n in 0:120]
        @test sum(probs) ≈ 1.0 atol=1e-12
        @test all(isfinite, probs)
        @test all(p -> p >= -1e-14, probs)

        for n in a:(a + 12)
            expected = exp(local_logbinomial(n, a) + (a + 1) * log(success) + (n - a) * log(θ))
            @test hidden_count_given_reconstructed_count_pmf(n, a, ti, tj, tl, inv_pars) ≈ expected atol=1e-14
        end
    end

    @testset "a = 0 normalization and coefficients" begin
        probs = [hidden_count_given_reconstructed_count_pmf(n, 0, ti, tj, tl, inv_pars) for n in 0:160]
        @test sum(probs) ≈ 1.0 atol=1e-12
        @test probs[1] > 0
        @test all(isfinite, probs)
        @test all(p -> p >= -1e-14, probs)

        q = hidden_reconstructed_unsampled_probability(tj, tl, inv_pars)
        α1 = alpha_bd(1.0, ti, tj, inv_pars)
        β1 = beta_bd(1.0, ti, tj, inv_pars)
        γ1 = gamma_bd(1.0, ti, tj, inv_pars)
        α0 = alpha_bd(0.0, ti, tj, inv_pars)
        β0 = beta_bd(0.0, ti, tj, inv_pars)
        γ0 = gamma_bd(0.0, ti, tj, inv_pars)
        H0 = α1 - α0 + β1 * q / (1 - γ1 * q) - β0 * q / (1 - γ0 * q)

        @test hidden_count_given_reconstructed_count_pmf(0, 0, ti, tj, tl, inv_pars) ≈ (α1 - α0) / H0
        for n in 1:12
            h = q^n * (β1 * γ1^(n - 1) - β0 * γ0^(n - 1))
            @test hidden_count_given_reconstructed_count_pmf(n, 0, ti, tj, tl, inv_pars) ≈ h / H0 atol=1e-13
        end
    end

    @testset "impossible states and validation" begin
        @test hidden_count_given_reconstructed_count_pmf(1, 2, ti, tj, tl, inv_pars) == 0.0
        @test hidden_count_given_reconstructed_count_logpmf(1, 2, ti, tj, tl, inv_pars) == -Inf
        @test_throws ArgumentError hidden_count_given_reconstructed_count_pmf(-1, 0, ti, tj, tl, inv_pars)
        @test_throws ArgumentError hidden_count_given_reconstructed_count_pmf(0, -1, ti, tj, tl, inv_pars)
        @test_throws ArgumentError hidden_count_given_reconstructed_count_pmf(0, 0, tj, ti, tl, inv_pars)
    end

    @testset "table output" begin
        rows = hidden_count_given_reconstructed_count_pmf_table(1, ti, tj, tl, inv_pars; nmax=8)
        @test length(rows) == 9
        @test rows[1].n == 0
        @test rows[1].a == 1
        @test rows[1].case == "a_ge_1"
        @test rows[1].probability == 0.0
        @test rows[3].probability ≈ hidden_count_given_reconstructed_count_pmf(2, 1, ti, tj, tl, inv_pars)

        rows0 = hidden_count_given_reconstructed_count_pmf_table(0, ti, tj, tl, inv_pars; nmax=2)
        @test rows0[1].case == "a0_n0"
        @test rows0[2].case == "a0_nge1"
    end

    @testset "edge regimes" begin
        q0_pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 1.0)
        @test hidden_reconstructed_unsampled_probability(tl, tl, q0_pars) ≈ 0.0
        @test hidden_count_given_reconstructed_count_pmf(3, 3, ti, tl, tl, q0_pars) ≈ 1.0
        @test hidden_count_given_reconstructed_count_pmf(4, 3, ti, tl, tl, q0_pars) ≈ 0.0

        low_sample_pars = ConstantRateBDParameters(1.4, 0.3, 0.02, 0.7, 0.01)
        low_q = hidden_reconstructed_unsampled_probability(tj, tl, low_sample_pars)
        @test low_q > 0.8
        low_probs = [hidden_count_given_reconstructed_count_pmf(n, 1, ti, tj, tl, low_sample_pars) for n in 0:400]
        @test sum(low_probs) ≈ 1.0 atol=1e-9
        @test sum(n * low_probs[n + 1] for n in 0:400) > 1.2

        a0_pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 0.4, 0.8)
        @test hidden_count_given_reconstructed_count_pmf(0, 0, ti, tj, tl, a0_pars) > 0
        @test sum(hidden_count_given_reconstructed_count_pmf(n, 0, ti, tj, tl, a0_pars) for n in 0:160) ≈ 1.0 atol=1e-12
    end
end
