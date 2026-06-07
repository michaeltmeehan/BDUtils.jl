    # Core analytical invariants: scalar closed forms, coefficients, PMFs,
    # marginals, and tails. These are the first line of defense for refactors.
    @testset "core invariant: PGF/probability helpers" begin
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
