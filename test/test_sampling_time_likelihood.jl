@testset "sampling_time_likelihood independent validation" begin
    function conditioned_coeff(n, ti, tj, tl, pars)
        n < 0 && return 0.0
        α = conditioned_reconstructed_alpha_bd(0.0, ti, tj, tl, pars)
        β = conditioned_reconstructed_beta_bd(0.0, ti, tj, tl, pars)
        γ = conditioned_reconstructed_gamma_bd(0.0, ti, tj, tl, pars)
        n == 0 && return α
        return β * γ^(n - 1)
    end

    function coeff_power(a, b, ti, tj, tl, pars)
        a < 0 && return 0.0
        a == 0 && return b == 0 ? 1.0 : 0.0
        masses = zeros(Float64, b + 1)
        masses[1] = 1.0
        for _ in 1:a
            next = zeros(Float64, b + 1)
            for current in 0:b
                iszero(masses[current + 1]) && continue
                for added in 0:(b - current)
                    next[current + added + 1] += masses[current + 1] *
                        conditioned_coeff(added, ti, tj, tl, pars)
                end
            end
            masses = next
        end
        return masses[b + 1]
    end

    function removal_jump(b, c, ψ̃; labelled_samples=false)
        c > b && return 0.0
        coefficient = labelled_samples ? prod((b - k for k in 0:(c - 1)); init=1) : binomial(b, c)
        return coefficient * ψ̃^c
    end

    function independent_sampling_time_likelihood(
        t0,
        sampling_times,
        sample_counts,
        terminal_count,
        pars;
        tℓ,
        max_count,
        labelled_samples=false,
    )
        f = Dict{Int,Float64}(1 => 1.0)
        u = t0
        for i in eachindex(sampling_times)
            ti = sampling_times[i]
            c = sample_counts[i]
            g = Dict{Int,Float64}()
            for (a, mass) in f
                for b in 0:max_count
                    k = coeff_power(a, b, u, ti, tℓ, pars)
                    iszero(k) && continue
                    g[b] = get(g, b, 0.0) + mass * k
                end
            end

            ψ̃ = transformed_sampling_rate(ti, tℓ, pars)
            f_new = Dict{Int,Float64}()
            for (b, mass) in g
                d = b - c
                d >= 0 || continue
                jump = removal_jump(b, c, ψ̃; labelled_samples=labelled_samples)
                f_new[d] = get(f_new, d, 0.0) + mass * jump
            end
            f = f_new
            u = ti
        end

        return sum(
            mass * coeff_power(a, terminal_count, u, tℓ, tℓ, pars)
            for (a, mass) in f
        )
    end

    function independent_retained_mass(t0, sampling_times, sample_counts, pars; tℓ, max_count)
        f = Dict{Int,Float64}(1 => 1.0)
        u = t0
        for i in eachindex(sampling_times)
            ti = sampling_times[i]
            c = sample_counts[i]
            g = Dict{Int,Float64}()
            for (a, mass) in f
                for b in 0:max_count
                    g[b] = get(g, b, 0.0) + mass * coeff_power(a, b, u, ti, tℓ, pars)
                end
            end
            ψ̃ = transformed_sampling_rate(ti, tℓ, pars)
            f = Dict{Int,Float64}()
            for (b, mass) in g
                d = b - c
                d >= 0 || continue
                f[d] = get(f, d, 0.0) + mass * removal_jump(b, c, ψ̃)
            end
            u = ti
        end
        return sum(values(f))
    end

    @testset "input validation" begin
        removal_pars = ConstantRateBDParameters(1.8, 0.5, 0.7, 1.0, 0.25)
        nonremoval_pars = ConstantRateBDParameters(1.8, 0.5, 0.7, 0.0, 0.25)
        t0, s, tl = 0.0, 0.6, 1.4

        @test_throws ArgumentError sampling_time_likelihood(t0, [s, 0.9], [1], 1, removal_pars; tℓ=tl)
        @test_throws ArgumentError sampling_time_likelihood(t0, [s, s], [1, 1], 1, removal_pars; tℓ=tl)
        @test_throws ArgumentError sampling_time_likelihood(t0, [0.9, s], [1, 1], 1, removal_pars; tℓ=tl)
        @test_throws ArgumentError sampling_time_likelihood(t0, [s], [1], 1, removal_pars; tℓ=s)
        @test_throws ArgumentError sampling_time_likelihood(t0, [s], [1], 1, removal_pars; tℓ=0.5)
        @test_throws ArgumentError sampling_time_likelihood(t0, [s], [1], 1, nonremoval_pars; tℓ=tl)
        @test_throws ArgumentError sampling_time_likelihood(t0, [s], [1], 1, removal_pars; tℓ=tl, max_count=0)
        @test_throws ArgumentError sampling_time_likelihood(t0, [s], [1], 1, removal_pars; tℓ=tl, terminal_sampling=false)
        @test_throws ArgumentError sampling_time_likelihood(
            t0, [s], [1], 0, removal_pars;
            tℓ=tl,
            terminal_sampling=false,
            terminal_condition=:observed,
        )
    end

    @testset "deterministic: one serial time against independent coefficient composition" begin
        pars1 = ConstantRateBDParameters(1.45, 0.35, 0.55, 1.0, 0.4)
        t0, s, tl = 0.0, 0.45, 1.2
        direct = independent_sampling_time_likelihood(t0, [s], [2], 1, pars1; tℓ=tl, max_count=8)
        observed = sampling_time_likelihood(t0, [s], [2], 1, pars1; tℓ=tl, max_count=8)
        @test observed ≈ direct rtol=5e-12 atol=5e-14
    end

    @testset "deterministic: two serial times against brute-force latent-count summation" begin
        pars2 = ConstantRateBDParameters(1.25, 0.25, 0.5, 1.0, 0.35)
        t0, tl = 0.0, 1.1
        times = [0.3, 0.75]
        counts = [1, 1]
        max_count = 9

        retained = independent_retained_mass(t0, times, counts, pars2; tℓ=tl, max_count=max_count)
        retained_plus = independent_retained_mass(t0, times, counts, pars2; tℓ=tl, max_count=max_count + 3)
        @test retained / retained_plus > 0.999

        direct = independent_sampling_time_likelihood(t0, times, counts, 1, pars2; tℓ=tl, max_count=max_count)
        observed = sampling_time_likelihood(t0, times, counts, 1, pars2; tℓ=tl, max_count=max_count)
        @test observed ≈ direct rtol=5e-10 atol=5e-13
    end

    @testset "edge cases" begin
        removal_pars = ConstantRateBDParameters(1.8, 0.5, 0.7, 1.0, 0.25)
        no_sampling_pars = ConstantRateBDParameters(1.8, 0.5, 0.0, 1.0, 0.25)
        no_rho_pars = ConstantRateBDParameters(1.8, 0.5, 0.7, 1.0, 0.0)
        full_rho_pars = ConstantRateBDParameters(1.8, 0.5, 0.7, 1.0, 1.0)
        t0, s, tl = 0.0, 0.6, 1.4

        @test sampling_time_likelihood(t0, [s], [1], 0, no_sampling_pars; tℓ=tl) == 0.0
        @test sampling_time_likelihood(t0, [s], [1], 1, no_rho_pars; tℓ=tl) >= 0.0
        @test sampling_time_likelihood(t0, [s], [1], 1, full_rho_pars; tℓ=tl) >= 0.0
        @test sampling_time_likelihood(t0, [s], [1], 1, removal_pars; tℓ=tl) >= 0.0

        zero_checkpoint = sampling_time_likelihood(t0, [s], [0], 1, removal_pars; tℓ=tl)
        no_checkpoint = sampling_time_likelihood(t0, Float64[], Int[], 1, removal_pars; tℓ=tl)
        @test 0.0 <= zero_checkpoint <= no_checkpoint

        @test sampling_time_likelihood(
            t0, Float64[], Int[], 0, removal_pars;
            tℓ=tl,
            terminal_sampling=false,
            terminal_condition=:any,
        ) ≈ 1.0
    end

    @testset "terminal endpoint identities" begin
        pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 0.5)
        no_rho_pars = ConstantRateBDParameters(1.2, 0.4, 0.5, 1.0, 0.0)
        t0, s, tl = 0.0, 0.75, 1.5

        terminal_cap = conditioned_reconstructed_count_truncation(t0, tl, tl, pars; atol=1e-12)
        terminal_sum = sum(
            sampling_time_likelihood(t0, Float64[], Int[], m, pars; tℓ=tl)
            for m in 0:terminal_cap
        )
        terminal_tail = conditioned_reconstructed_count_tail(terminal_cap, t0, tl, tl, pars)
        @test terminal_sum + terminal_tail ≈ 1.0 rtol=1e-10 atol=2e-12

        @test sampling_time_likelihood(t0, Float64[], Int[], 0, no_rho_pars; tℓ=tl) ≈ 1.0
        @test sampling_time_likelihood(t0, Float64[], Int[], 1, no_rho_pars; tℓ=tl) ≈ 0.0 atol=1e-12

        endpoint_values = [
            sampling_time_likelihood(t0, [s], [1], m, pars; tℓ=tl)
            for m in 0:5
        ]
        @test all(isfinite, endpoint_values)
        @test all(x -> x >= 0.0, endpoint_values)
        @test all(x -> x <= 1.0, endpoint_values)

        grouped = sampling_time_likelihood(t0, [s], [2], 1, pars; tℓ=tl)
        @test isfinite(grouped)
        @test 0.0 <= grouped <= 1.0

        @test_throws ArgumentError sampling_time_likelihood(t0, Float64[], Int[], -1, pars; tℓ=tl)
    end

    @testset "diagnostics" begin
        pars = ConstantRateBDParameters(1.25, 0.25, 0.5, 1.0, 0.35)
        t0, tl = 0.0, 1.1
        times = [0.3, 0.75]
        counts = [1, 1]
        terminal_count = 1

        likelihood = sampling_time_likelihood(t0, times, counts, terminal_count, pars; tℓ=tl)
        diagnostic = sampling_time_likelihood(
            t0, times, counts, terminal_count, pars;
            tℓ=tl,
            diagnostics=true,
        )

        @test diagnostic.likelihood ≈ likelihood
        @test length(diagnostic.forward_vectors) == length(times) + 1
        @test diagnostic.forward_vectors[1][2] == 1.0
        @test length(diagnostic.serial_contributions) == length(times)
        @test diagnostic.effective_max_counts == [sum(counts) + terminal_count, counts[end] + terminal_count]
        @test diagnostic.max_count === nothing
        @test diagnostic.retained_mass ≈ sum(last(diagnostic.forward_vectors))
        @test diagnostic.terminal_contribution ≈ diagnostic.likelihood
        @test diagnostic.tail_mass === nothing

        no_terminal = sampling_time_likelihood(
            t0, times, counts, 0, pars;
            tℓ=tl,
            terminal_sampling=false,
            terminal_condition=:any,
            diagnostics=true,
        )
        @test no_terminal.likelihood ≈ no_terminal.retained_mass
        @test no_terminal.terminal_contribution == 1.0
    end

    @testset "cached origin-time evaluator" begin
        pars = ConstantRateBDParameters(1.35, 0.3, 0.55, 1.0, 0.4)
        no_sampling_pars = ConstantRateBDParameters(1.35, 0.3, 0.0, 1.0, 0.4)
        tl = 1.4

        @test BDUtils._binomial_coefficient(Float64, 100, 50) ≈ Float64(binomial(big(100), 50))
        @test isfinite(BDUtils._no_sample_reconstructed_kernel(0.0, 0.5, 1.0, 31, 68, pars))
        @test isfinite(BDUtils._grouped_removal_sampling_jump(100, 50, 0.2))

        one_time = [0.45]
        one_counts = [1]
        one_cache = cache_sampling_time_likelihood(one_time, one_counts, 1, pars; tℓ=tl)
        @test one_cache.first_sampling_time == first(one_time)
        for t0 in [-0.2, 0.0, 0.3]
            cached = sampling_time_likelihood(one_cache, t0)
            full = sampling_time_likelihood(t0, one_time, one_counts, 1, pars; tℓ=tl)
            @test cached ≈ full rtol=5e-12 atol=5e-14
            @test sampling_time_loglikelihood(one_cache, t0) ≈ log(full) rtol=5e-12 atol=5e-14
        end

        times = [0.35, 0.8]
        counts = [1, 2]
        multi_cache = cache_sampling_time_likelihood(times, counts, 1, pars; tℓ=tl, labelled_samples=true)
        for t0 in [-0.1, 0.05, 0.25]
            cached = sampling_time_likelihood(multi_cache, t0)
            full = sampling_time_likelihood(t0, times, counts, 1, pars; tℓ=tl, labelled_samples=true)
            @test cached ≈ full rtol=5e-12 atol=5e-14
        end

        terminal_cache = cache_sampling_time_likelihood(
            [0.55],
            [1],
            0,
            pars;
            tℓ=tl,
            terminal_sampling=false,
            terminal_condition=:censored,
        )
        for t0 in [-0.05, 0.2, 0.5]
            cached = sampling_time_likelihood(terminal_cache, t0)
            full = sampling_time_likelihood(
                t0,
                [0.55],
                [1],
                0,
                pars;
                tℓ=tl,
                terminal_sampling=false,
                terminal_condition=:censored,
            )
            @test cached ≈ full rtol=5e-12 atol=5e-14
        end

        grouped_cache = cache_sampling_time_likelihood(
            [0.4, 0.9],
            [1, 1],
            pars;
            tℓ=0.9,
            terminal_condition=:terminated,
        )
        for t0 in [-0.1, 0.0, 0.3]
            cached = sampling_time_likelihood(grouped_cache, t0)
            full = grouped_sampling_time_likelihood(
                t0,
                [0.4, 0.9],
                [1, 1],
                pars;
                tℓ=0.9,
                terminal_condition=:terminated,
            )
            @test cached ≈ full rtol=5e-12 atol=5e-14
        end

        clustered_times = [-0.2 + 3.24 * (1 - (1 - i / 199)^4) for i in 0:199]
        clustered_pars = ConstantRateBDParameters(2.0, 0.2, 0.7, 1.0, 0.0)
        clustered_cache = cache_sampling_time_likelihood(
            clustered_times,
            ones(Int, length(clustered_times)),
            clustered_pars;
            tℓ=last(clustered_times),
            terminal_condition=:any,
        )
        @test all(isfinite, clustered_cache.downstream)
        @test isfinite(clustered_cache.downstream_log_scale)
        @test sampling_time_likelihood(clustered_cache, -1.0) == Inf
        @test isfinite(sampling_time_loglikelihood(clustered_cache, -1.0))
        clustered_profile = origin_time_loglikelihood_profile(clustered_cache, [-3.0, -2.0, -1.0, -0.5])
        @test all(isfinite, clustered_profile.loglikelihood)
        clustered_fit = origin_time_mle(clustered_cache; lower=-3.0, upper=prevfloat(first(clustered_times)))
        @test clustered_fit.status in (:converged, :lower_bound, :upper_bound, :maxiter)
        @test isfinite(clustered_fit.loglikelihood)

        @test sampling_time_likelihood(one_cache, first(one_time)) == 0.0
        @test sampling_time_likelihood(one_cache, 0.6) == 0.0
        @test sampling_time_loglikelihood(one_cache, first(one_time)) == -Inf

        profile = origin_time_loglikelihood_profile(one_cache, [-0.2, 0.0, 0.45, 0.6])
        @test profile.t0 == [-0.2, 0.0, 0.45, 0.6]
        @test length(profile.loglikelihood) == 4
        @test length(profile.delta_loglikelihood) == 4
        @test maximum(filter(isfinite, profile.delta_loglikelihood)) ≈ 0.0
        @test profile.loglikelihood[3] == -Inf
        @test profile.delta_loglikelihood[4] == -Inf

        fit = origin_time_mle(multi_cache; lower=-0.5, upper=0.3, tolerance=1e-8)
        @test fit isa OriginTimeMLEResult
        @test -0.5 <= fit.t0_hat <= 0.3
        @test fit.lower == -0.5
        @test fit.upper == 0.3
        @test fit.iterations >= 0
        @test fit.n_evaluations >= fit.iterations
        @test fit.status in (:converged, :lower_bound, :upper_bound, :maxiter)
        @test fit.loglikelihood ≈ sampling_time_loglikelihood(multi_cache, fit.t0_hat) rtol=1e-12 atol=1e-12

        coarse_grid = collect(range(-0.5, stop=0.3, length=41))
        coarse_best = maximum(sampling_time_loglikelihood(multi_cache, t0) for t0 in coarse_grid)
        @test fit.loglikelihood + 1e-6 >= coarse_best

        @test_throws ArgumentError origin_time_mle(multi_cache; lower=0.3, upper=0.2)
        @test_throws ArgumentError origin_time_mle(multi_cache; lower=-0.5, upper=first(times))
        @test_throws ArgumentError origin_time_mle(multi_cache; lower=-Inf, upper=0.2)

        lower_boundary_fit = origin_time_mle(multi_cache; lower=0.3, upper=prevfloat(first(times)), tolerance=1e-8)
        @test lower_boundary_fit.status == :lower_bound
        @test lower_boundary_fit.t0_hat == lower_boundary_fit.lower

        no_sampling_cache = cache_sampling_time_likelihood([0.45], [1], 1, no_sampling_pars; tℓ=tl)
        no_sampling_fit = origin_time_mle(no_sampling_cache; lower=-0.2, upper=0.3)
        @test no_sampling_fit.status == :nonfinite_objective
        @test !no_sampling_fit.converged
        @test no_sampling_fit.loglikelihood == -Inf

        convenience = origin_time_mle(
            times,
            pars;
            sample_counts=counts,
            terminal_count=1,
            tℓ=tl,
            labelled_samples=true,
            lower=-0.5,
            upper=0.3,
            tolerance=1e-8,
        )
        @test convenience.t0_hat ≈ fit.t0_hat atol=1e-6
        @test convenience.loglikelihood ≈ fit.loglikelihood atol=1e-8
    end

    @testset "stress: Monte Carlo window-density validation" begin
        if get(ENV, "BDUTILS_STRESS_TESTS", "false") == "true"
            pars3 = ConstantRateBDParameters(0.8, 0.25, 0.45, 1.0, 0.7)
            rng = MersenneTwister(20240607)
            t0, s, tl = 0.0, 0.55, 1.0
            terminal_count = 1
            half_width = 0.03
            nsims = 120_000
            conditioned = 0
            hits = 0

            for _ in 1:nsims
                log = simulate_bd(rng, pars3, tl; apply_ρ₀=true)
                A_at(log, t0, tl) == 1 || continue
                conditioned += 1
                serial_window = count(
                    log.kind[i] == SerialSampling &&
                    s - half_width < log.time[i] < s + half_width
                    for i in eachindex(log.time)
                )
                terminal = count(
                    log.kind[i] == SerialSampling && log.time[i] == tl
                    for i in eachindex(log.time)
                )
                outside_serial = count(
                    log.kind[i] == SerialSampling &&
                    t0 < log.time[i] < tl &&
                    !(s - half_width < log.time[i] < s + half_width)
                    for i in eachindex(log.time)
                )
                if serial_window == 1 && terminal == terminal_count && outside_serial == 0
                    hits += 1
                end
            end

            empirical_density = hits / conditioned / (2half_width)
            analytical = sampling_time_likelihood(t0, [s], [1], terminal_count, pars3; tℓ=tl)
            @test conditioned > 20_000
            @test empirical_density ≈ analytical rtol=0.25 atol=0.02
        else
            @test_skip "set BDUTILS_STRESS_TESTS=true to run Monte Carlo window-density validation"
        end
    end
end
