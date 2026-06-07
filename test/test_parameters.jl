    @testset "derived helpers" begin
        @test compute_R0(λ, μ, ψ, r) == λ / (μ + ψ * r)
        @test compute_delta(λ, μ, ψ, r) == μ + ψ * r
        @test compute_sampled_removal_rate(pars) == r * ψ
        @test compute_sampling_fraction(pars) == (r * ψ) / (μ + ψ * r)
        @test compute_R0([λ], [μ], [ψ], r) == [λ / (μ + ψ * r)]
        @test compute_R0(pars) == compute_R0(λ, μ, ψ, r)
        @test compute_delta(pars) == compute_delta(λ, μ, ψ, r)

        alt = reparameterize_R0_delta_s(pars)
        @test keys(alt) == (:R0, :δ, :s, :r, :ρ₀)
        @test alt.R0 == compute_R0(pars)
        @test alt.δ == compute_delta(pars)
        @test alt.s == compute_sampling_fraction(pars)
        @test alt.r == pars.r
        @test alt.ρ₀ == pars.ρ₀

        roundtrip = parameters_from_R0_delta_s_r(alt.R0, alt.δ, alt.s, alt.r, alt.ρ₀)
        @test roundtrip.λ ≈ pars.λ
        @test roundtrip.μ ≈ pars.μ
        @test roundtrip.ψ ≈ pars.ψ
        @test roundtrip.r == pars.r
        @test roundtrip.ρ₀ == pars.ρ₀

        hand = parameters_from_R0_delta_s_r(2.0, 1.5, 0.2, 0.5, 0.1)
        @test hand.λ == 3.0
        @test hand.μ ≈ 1.2
        @test hand.ψ ≈ 0.6
        @test hand.r == 0.5
        @test hand.ρ₀ == 0.1

        zero_r = parameters_from_R0_delta_s_r(2.0, 1.5, 0.0, 0.0, 0.25)
        @test zero_r.λ == 3.0
        @test zero_r.μ == 1.5
        @test zero_r.ψ == 0.0
        @test zero_r.r == 0.0
        @test zero_r.ρ₀ == 0.25

        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 1.5, 0.1, 0.0, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(0.0, 1.5, 0.0, 0.0, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 0.0, 0.0, 0.0, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 1.5, -0.1, 0.5, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 1.5, 1.1, 0.5, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 1.5, 0.1, -0.1, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, 1.5, 0.1, 0.5, 1.1)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(Inf, 1.5, 0.1, 0.5, 0.0)
        @test_throws ArgumentError parameters_from_R0_delta_s_r(2.0, NaN, 0.1, 0.5, 0.0)
    end

    @testset "constant-rate parameter object" begin
        @test pars.λ == λ
        @test pars.μ == μ
        @test pars.ψ == ψ
        @test pars.r == r
        @test pars.ρ₀ == 0.25
        @test ConstantRateBDParameters(λ, μ, ψ, r).ρ₀ == 0.0
        @test ConstantRateBDParameters(2, 0, 0, 0, 1) isa ConstantRateBDParameters{Float64}
        @test sprint(show, pars) == "ConstantRateBDParameters(λ=2.0, μ=0.5, ψ=0.4, r=0.7, ρ₀=0.25)"

        @test ConstantRateBDParameters(λ, μ, ψ, 0.0, 0.0).r == 0.0
        @test ConstantRateBDParameters(λ, μ, ψ, 1.0, 1.0).ρ₀ == 1.0

        @test_throws ArgumentError ConstantRateBDParameters(0.0, μ, ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, -μ, ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, -ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, ψ, 1.1)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, ψ, r, -0.1)
        @test_throws ArgumentError ConstantRateBDParameters(Inf, μ, ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(NaN, μ, ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, Inf, ψ, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, NaN, r)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, ψ, NaN)
        @test_throws ArgumentError ConstantRateBDParameters(λ, μ, ψ, r, Inf)
    end
