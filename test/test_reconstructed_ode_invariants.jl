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
