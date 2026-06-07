    @testset "core invariant: NS derivative identities" begin
        ti = 0.0
        tj = 1.1
        derivative_pars = ConstantRateBDParameters(1.7, 0.6, 0.5, 0.65)
        αs, βs, γs = constant_rate_pgf_series(4, ti, tj, derivative_pars)

        for (f, coeffs) in ((w -> alpha_bd(w, ti, tj, derivative_pars), αs),
                            (w -> beta_bd(w, ti, tj, derivative_pars), βs),
                            (w -> gamma_bd(w, ti, tj, derivative_pars), γs))
            @test central_first_derivative(f, 0.0; h=1e-5) ≈ coeffs[2] rtol=2e-5 atol=2e-7
            @test central_second_derivative(f, 0.0; h=2e-4) / 2 ≈ coeffs[3] rtol=2e-4 atol=2e-6
        end

        z = 0.4
        g_coeffs = vec(sum(joint_pmf_NS_table(12, 4, ti, tj, derivative_pars) .* (z .^ (0:12)), dims=1))
        gz = w -> scalar_joint_pgf(z, w, ti, tj, derivative_pars)
        @test central_first_derivative(gz, 0.0; h=1e-5) ≈ g_coeffs[2] rtol=2e-5 atol=2e-7
        @test central_second_derivative(gz, 0.0; h=2e-4) / 2 ≈ g_coeffs[3] rtol=2e-4 atol=2e-6

        γ1 = gamma_bd(1.0, ti, tj, derivative_pars)
        β1 = beta_bd(1.0, ti, tj, derivative_pars)
        expected_n = β1 / (1 - γ1)^2
        expected_n2factorial = 2β1 * γ1 / (1 - γ1)^3
        ncut = n_truncation(ti, tj, derivative_pars; atol=1e-12)
        @test sum(n * n_marginal_pmf(n, ti, tj, derivative_pars) for n in 0:ncut) + β1 * (γ1^ncut) * (ncut + 1 - ncut * γ1) / (1 - γ1)^2 ≈ expected_n rtol=1e-11
        @test central_first_derivative(zval -> scalar_joint_pgf(zval, 1.0, ti, tj, derivative_pars), 1.0; h=1e-5) ≈ expected_n rtol=1e-8
        @test central_second_derivative(zval -> scalar_joint_pgf(zval, 1.0, ti, tj, derivative_pars), 1.0; h=2e-4) ≈ expected_n2factorial rtol=2e-5

        scut = s_truncation(ti, tj, derivative_pars; atol=1e-12, max_smax=2_000)
        expected_s = sum(s * s_marginal_pmf(s, ti, tj, derivative_pars) for s in 0:scut)
        expected_s2factorial = sum(s * (s - 1) * s_marginal_pmf(s, ti, tj, derivative_pars) for s in 0:scut)
        s_tail = s_marginal_tail(scut, ti, tj, derivative_pars)
        gw = w -> scalar_joint_pgf(1.0, w, ti, tj, derivative_pars)
        @test central_first_derivative(gw, 1.0; h=1e-5) ≈ expected_s rtol=5e-6 atol=max(1e-9, 10s_tail)
        @test central_second_derivative(gw, 1.0; h=2e-4) ≈ expected_s2factorial rtol=2e-4 atol=max(1e-8, 100s_tail)

        table = joint_pmf_NS_table(n_truncation(ti, tj, derivative_pars; atol=1e-11),
                                   s_truncation(ti, tj, derivative_pars; atol=1e-11, max_smax=2_000),
                                   ti, tj, derivative_pars)
        z_inside = 0.55
        w_inside = 0.45
        @test table_pgf_sum(table, z_inside, w_inside) ≈ scalar_joint_pgf(z_inside, w_inside, ti, tj, derivative_pars) atol=3e-10
    end

    # Core ODE invariants: the closed forms must satisfy their defining
    # forward and backward triangular systems, plus scalar PGF equations.
    @testset "core invariant: forward triangular ODE residuals" begin
        ti = 0.2

        for ode_pars in ODE_REGIMES
            for w in FORWARD_W_VALUES
                @test alpha_bd(w, ti, ti, ode_pars) ≈ 0.0 atol=1e-14
                @test beta_bd(w, ti, ti, ode_pars) ≈ 1.0 atol=1e-14
                @test gamma_bd(w, ti, ti, ode_pars) ≈ 0.0 atol=1e-14

                a, b, _ = bd_coefficients(w, ode_pars)
                for tj in FORWARD_TJ_VALUES
                    γij = gamma_bd(w, ti, tj, ode_pars)
                    βij = beta_bd(w, ti, tj, ode_pars)
                    dγ = central_first_derivative(x -> gamma_bd(w, ti, x, ode_pars), tj; h=2e-5)
                    dβ = central_first_derivative(x -> beta_bd(w, ti, x, ode_pars), tj; h=2e-5)
                    dα = central_first_derivative(x -> alpha_bd(w, ti, x, ode_pars), tj; h=2e-5)

                    rhs_γ = (1 - γij) * (ode_pars.λ - a * γij) - ode_pars.ψ * (1 - w) * γij
                    rhs_β = (2a * γij + b) * βij
                    rhs_α = a * βij

                    @test dγ ≈ rhs_γ rtol=2e-7 atol=2e-8
                    @test dβ ≈ rhs_β rtol=2e-6 atol=2e-8
                    @test dα ≈ rhs_α rtol=2e-6 atol=2e-8
                end
            end
        end
    end

    @testset "core invariant: backward triangular ODE residuals" begin
        for backward_pars in ODE_REGIMES
            for w in BACKWARD_W_VALUES
                for t in (0.0, 0.8, 1.6)
                    @test alpha_bd(w, t, t, backward_pars) ≈ 0.0 atol=1e-14
                    @test beta_bd(w, t, t, backward_pars) ≈ 1.0 atol=1e-14
                    @test gamma_bd(w, t, t, backward_pars) ≈ 0.0 atol=1e-14
                end

                a = bd_a(w, backward_pars)
                b = bd_b_backward(w, backward_pars)
                for (ti, tj) in BACKWARD_T_PAIRS
                    αij = alpha_bd(w, ti, tj, backward_pars)
                    βij = beta_bd(w, ti, tj, backward_pars)
                    γij = gamma_bd(w, ti, tj, backward_pars)
                    dα = central_first_derivative(x -> alpha_bd(w, x, tj, backward_pars), ti; h=2e-5)
                    dβ = central_first_derivative(x -> beta_bd(w, x, tj, backward_pars), ti; h=2e-5)
                    dγ = central_first_derivative(x -> gamma_bd(w, x, tj, backward_pars), ti; h=2e-5)

                    rhs_α_quadratic = backward_pars.λ * αij^2 + b * αij + a
                    rhs_α_branching = (1 - αij) * (a - backward_pars.λ * αij) - backward_pars.ψ * (1 - w) * αij
                    rhs_β = (2backward_pars.λ * αij + b) * βij
                    rhs_γ = backward_pars.λ * βij

                    @test rhs_α_quadratic ≈ rhs_α_branching rtol=2e-13 atol=2e-14
                    @test -dα ≈ rhs_α_quadratic rtol=2e-6 atol=2e-8
                    @test -dβ ≈ rhs_β rtol=2e-6 atol=2e-8
                    @test -dγ ≈ rhs_γ rtol=2e-7 atol=2e-8
                end
            end
        end
    end

    @testset "core invariant: backward Kolmogorov PGF residuals" begin
        z_values = (0.0, 0.35, 0.8)
        w_values = (0.0, 0.4, 1.0)

        for backward_pars in ODE_REGIMES
            for z in z_values, w in w_values, (ti, tj) in PGF_T_PAIRS
                F = scalar_joint_pgf(z, w, ti, tj, backward_pars)
                dF = central_first_derivative(x -> scalar_joint_pgf(z, w, x, tj, backward_pars), ti; h=2e-5)
                rhs = backward_generator(F, w, backward_pars)
                @test -dF ≈ rhs rtol=4e-6 atol=3e-8
            end
        end
    end

    @testset "core invariant: backward Kendall specialization" begin
        for backward_pars in KENDALL_REGIMES
            δ = backward_pars.μ + backward_pars.r * backward_pars.ψ
            for (ti, tj) in BACKWARD_T_PAIRS
                ξ = alpha_bd(1.0, ti, tj, backward_pars)
                η = gamma_bd(1.0, ti, tj, backward_pars)
                β1 = beta_bd(1.0, ti, tj, backward_pars)
                dξ = central_first_derivative(x -> alpha_bd(1.0, x, tj, backward_pars), ti; h=2e-5)
                dη = central_first_derivative(x -> gamma_bd(1.0, x, tj, backward_pars), ti; h=2e-5)
                dβ = central_first_derivative(x -> beta_bd(1.0, x, tj, backward_pars), ti; h=2e-5)

                @test β1 ≈ (1 - ξ) * (1 - η) rtol=2e-12 atol=2e-14
                @test -dξ ≈ backward_pars.λ * ξ^2 - (backward_pars.λ + δ) * ξ + δ rtol=2e-6 atol=2e-8
                @test -dη ≈ backward_pars.λ * β1 rtol=2e-7 atol=2e-8
                @test -dβ ≈ (2backward_pars.λ * ξ - (backward_pars.λ + δ)) * β1 rtol=2e-6 atol=2e-8
            end
        end
    end

    @testset "core invariant: forward Kendall identities at w = 1" begin
        ti = 0.0

        for kendall_pars in KENDALL_REGIMES
            @test alpha_bd(1.0, ti, ti, kendall_pars) ≈ 0.0 atol=1e-14
            @test gamma_bd(1.0, ti, ti, kendall_pars) ≈ 0.0 atol=1e-14
            @test beta_bd(1.0, ti, ti, kendall_pars) ≈ 1.0 atol=1e-14

            δ = kendall_pars.μ + kendall_pars.r * kendall_pars.ψ
            for tj in (0.35, 1.0, 1.7)
                ξ = alpha_bd(1.0, ti, tj, kendall_pars)
                η = gamma_bd(1.0, ti, tj, kendall_pars)
                β1 = beta_bd(1.0, ti, tj, kendall_pars)
                @test β1 ≈ (1 - ξ) * (1 - η) rtol=2e-12 atol=2e-14

                dη = central_first_derivative(x -> gamma_bd(1.0, ti, x, kendall_pars), tj; h=2e-5)
                dξ = central_first_derivative(x -> alpha_bd(1.0, ti, x, kendall_pars), tj; h=2e-5)
                @test dη ≈ (kendall_pars.λ - δ * η) * (1 - η) rtol=2e-7 atol=2e-8
                @test dξ ≈ δ * (1 - ξ) * (1 - η) rtol=2e-6 atol=2e-8
            end
        end
    end

    @testset "core invariant: Riccati residual for p" begin
        for riccati_pars in ODE_REGIMES
            tk = 2.5
            @test p_unsampled(tk, tk, riccati_pars) ≈ 1.0 atol=1e-14
            for tj in (0.2, 1.0, 1.9)
                p = p_unsampled(tj, tk, riccati_pars)
                dp = central_first_derivative(x -> p_unsampled(x, tk, riccati_pars), tj; h=2e-5)
                rhs = riccati_pars.μ - (riccati_pars.λ + riccati_pars.μ + riccati_pars.ψ) * p + riccati_pars.λ * p^2
                @test -dp ≈ rhs rtol=2e-6 atol=2e-8
            end
        end
    end
