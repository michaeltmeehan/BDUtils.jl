    @testset "public API: conditioned reconstructed series, PMF, marginals, and truncation" begin
        ti = 0.0
        tj = 0.85
        tk = 2.4
        reconstructed_pars = ConstantRateBDParameters(1.8, 0.5, 0.7, 0.4)
        smax = 30
        raw_αs, raw_βs, raw_γs = reconstructed_pgf_series(smax, ti, tj, tk, reconstructed_pars)
        αs, βs, γs = conditioned_reconstructed_pgf_series(smax, ti, tj, tk, reconstructed_pars)

        pᵢ = unsampled_probability(ti, tk, reconstructed_pars)
        qᵢ = 1 - pᵢ
        expected_αs = copy(raw_αs)
        expected_αs[1] -= pᵢ
        expected_αs ./= qᵢ
        @test αs ≈ expected_αs
        @test βs ≈ raw_βs ./ qᵢ
        @test γs ≈ raw_γs

        for w in (0.0, 0.25, 0.65)
            powers = w .^ (0:smax)
            @test sum(αs .* powers) ≈ conditioned_reconstructed_alpha_bd(w, ti, tj, tk, reconstructed_pars) atol=1e-11
            @test sum(βs .* powers) ≈ conditioned_reconstructed_beta_bd(w, ti, tj, tk, reconstructed_pars) atol=1e-11
            @test sum(γs .* powers) ≈ conditioned_reconstructed_gamma_bd(w, ti, tj, tk, reconstructed_pars) atol=1e-11
        end

        table = conditioned_reconstructed_joint_pmf_table(9, 7, ti, tj, tk, reconstructed_pars)
        @test size(table) == (10, 8)
        @test table[1, 3] ≈ conditioned_reconstructed_joint_pmf(0, 2, ti, tj, tk, reconstructed_pars)
        @test table[2, 3] ≈ conditioned_reconstructed_joint_pmf(1, 2, ti, tj, tk, reconstructed_pars)
        @test table[5, 4] ≈ conditioned_reconstructed_joint_pmf(4, 3, ti, tj, tk, reconstructed_pars)
        @test all(x -> x >= -1e-13, table)

        f = copy(βs)
        for _ in 2:4
            next = zeros(eltype(f), length(f))
            for i in eachindex(next), k in 1:i
                next[i] += γs[k] * f[i - k + 1]
            end
            f = next
        end
        @test conditioned_reconstructed_joint_pmf(0, 5, ti, tj, tk, reconstructed_pars) ≈ αs[6]
        @test conditioned_reconstructed_joint_pmf(4, 5, ti, tj, tk, reconstructed_pars) ≈ f[6]

        z = 0.45
        w_inside = 0.35
        count_cut = conditioned_reconstructed_count_truncation(ti, tj, tk, reconstructed_pars; atol=1e-11)
        sampling_cut = conditioned_reconstructed_sampling_truncation(ti, tj, tk, reconstructed_pars; atol=1e-11, max_smax=2_000)
        pgf_table = conditioned_reconstructed_joint_pmf_table(count_cut, sampling_cut, ti, tj, tk, reconstructed_pars)
        @test table_pgf_sum(pgf_table, z, w_inside) ≈ conditioned_reconstructed_pgf(z, w_inside, ti, tj, tk, reconstructed_pars) atol=3e-10

        @test conditioned_reconstructed_count_pmf(0, ti, tj, tk, reconstructed_pars) ≈ conditioned_reconstructed_xi(ti, tj, tk, reconstructed_pars)
        @test conditioned_reconstructed_count_pmf(4, ti, tj, tk, reconstructed_pars) ≈ sum(conditioned_reconstructed_joint_pmf(4, s, ti, tj, tk, reconstructed_pars) for s in 0:120) atol=1e-12
        @test conditioned_reconstructed_sampling_marginal_pmf(3, ti, tj, tk, reconstructed_pars) ≈ sum(conditioned_reconstructed_joint_pmf(a, 3, ti, tj, tk, reconstructed_pars) for a in 0:120) atol=1e-12

        @test conditioned_reconstructed_count_tail(5, ti, tj, tk, reconstructed_pars) >= 0
        @test conditioned_reconstructed_count_tail(6, ti, tj, tk, reconstructed_pars) <= conditioned_reconstructed_count_tail(5, ti, tj, tk, reconstructed_pars)
        @test conditioned_reconstructed_count_tail(count_cut, ti, tj, tk, reconstructed_pars) <= 1e-11
        if count_cut > 0
            @test conditioned_reconstructed_count_tail(count_cut - 1, ti, tj, tk, reconstructed_pars) > 1e-11
        end
        @test sum(conditioned_reconstructed_count_pmf(a, ti, tj, tk, reconstructed_pars) for a in 0:count_cut) + conditioned_reconstructed_count_tail(count_cut, ti, tj, tk, reconstructed_pars) ≈ 1.0

        @test conditioned_reconstructed_sampling_tail(5, ti, tj, tk, reconstructed_pars) ≈ 1 - sum(conditioned_reconstructed_sampling_marginal_pmf(s, ti, tj, tk, reconstructed_pars) for s in 0:5) atol=1e-12
        @test conditioned_reconstructed_sampling_tail(5, ti, tj, tk, reconstructed_pars) >= 0
        @test conditioned_reconstructed_sampling_tail(6, ti, tj, tk, reconstructed_pars) <= conditioned_reconstructed_sampling_tail(5, ti, tj, tk, reconstructed_pars)
        @test conditioned_reconstructed_sampling_tail(sampling_cut, ti, tj, tk, reconstructed_pars) <= 1e-11
        if sampling_cut > 0
            @test conditioned_reconstructed_sampling_tail(sampling_cut - 1, ti, tj, tk, reconstructed_pars) > 1e-11
        end

        diagnostic = conditioned_reconstructed_joint_pmf_table(9, 7, ti, tj, tk, reconstructed_pars; diagnostics=true)
        @test diagnostic.table == table
        @test diagnostic.amax == 9
        @test diagnostic.smax == 7
        @test diagnostic.retained_mass ≈ sum(table)
        @test diagnostic.count_tail_mass ≈ conditioned_reconstructed_count_tail(9, ti, tj, tk, reconstructed_pars)
        @test diagnostic.sampling_tail_mass ≈ conditioned_reconstructed_sampling_tail(7, ti, tj, tk, reconstructed_pars)
        @test diagnostic.missing_mass ≈ 1 - diagnostic.retained_mass
        @test diagnostic.retained_mass + diagnostic.missing_mass ≈ 1.0 atol=1e-12
        @test diagnostic.count_only_tail_mass + diagnostic.sampling_only_tail_mass + diagnostic.joint_tail_overlap_mass + diagnostic.retained_mass ≈ 1.0 atol=1e-11

        audit_amax = conditioned_reconstructed_count_truncation(ti, tj, tk, reconstructed_pars; atol=1e-10)
        audit_smax = conditioned_reconstructed_sampling_truncation(ti, tj, tk, reconstructed_pars; atol=1e-10, max_smax=2_000)
        audit_table = conditioned_reconstructed_joint_pmf_table(audit_amax, audit_smax, ti, tj, tk, reconstructed_pars)
        for a in 0:audit_amax
            @test sum(audit_table[a + 1, :]) ≈ conditioned_reconstructed_count_pmf(a, ti, tj, tk, reconstructed_pars) atol=2e-10
        end
        for s in 0:audit_smax
            @test sum(audit_table[:, s + 1]) ≈ conditioned_reconstructed_sampling_marginal_pmf(s, ti, tj, tk, reconstructed_pars) atol=2e-10
        end

        @test_throws ArgumentError conditioned_reconstructed_pgf_series(-1, ti, tj, tk, reconstructed_pars)
        @test_throws ArgumentError conditioned_reconstructed_joint_pmf(-1, 0, ti, tj, tk, reconstructed_pars)
        @test_throws ArgumentError conditioned_reconstructed_joint_pmf(0, -1, ti, tj, tk, reconstructed_pars)
        @test_throws ArgumentError conditioned_reconstructed_count_truncation(ti, tj, tk, reconstructed_pars; atol=-1.0)
        @test_throws ArgumentError conditioned_reconstructed_sampling_truncation(ti, tj, tk, reconstructed_pars; atol=-1.0)
        @test_throws ArgumentError conditioned_reconstructed_sampling_truncation(ti, tj, tk, reconstructed_pars; atol=1e-14, max_smax=0)
        @test_throws ArgumentError conditioned_reconstructed_pgf_series(2, 1.0, 1.0, 1.0, reconstructed_pars)
        @test_throws ArgumentError conditioned_reconstructed_joint_pmf_table(2, 2, 1.0, 1.0, 1.0, reconstructed_pars)
        @test_throws ArgumentError conditioned_reconstructed_count_tail(2, 1.0, 1.0, 1.0, reconstructed_pars)
        @test_throws ArgumentError conditioned_reconstructed_sampling_tail(2, 1.0, 1.0, 1.0, reconstructed_pars)
    end

    @testset "public API: grouped reconstructed sampling-time likelihood" begin
        removal_pars = ConstantRateBDParameters(1.8, 0.5, 0.7, 1.0)
        nonremoval_pars = ConstantRateBDParameters(1.8, 0.5, 0.7, 0.4)
        t0 = 0.0
        times = [0.4, 0.9, 1.3]
        counts = [1, 0, 2]
        tl = 1.8

        function brute_grouped_likelihood(
            t0,
            sampling_times,
            sample_counts,
            pars;
            tℓ=last(sampling_times),
            labelled_samples=false,
            terminal_condition=:terminated,
        )
            remaining = reverse(cumsum(reverse(sample_counts)))

            f = Dict{Int,Float64}(1 => 1.0)
            u = t0

            for i in eachindex(sampling_times)
                ti = sampling_times[i]
                c = sample_counts[i]
                before_max = remaining[i]
                after_max = i == length(sampling_times) ? 0 : remaining[i + 1]

                g = Dict{Int,Float64}()

                for (a, mass) in f
                    a < 1 && continue
                    for b in a:before_max
                        k = BDUtils._no_sample_reconstructed_kernel(
                            u, ti, tℓ, a, b, pars
                        )
                        g[b] = get(g, b, 0.0) + mass * k
                    end
                end

                ψ̃ = transformed_sampling_rate(ti, tℓ, pars)
                f_new = Dict{Int,Float64}()

                for (b, mass) in g
                    d = b - c
                    0 <= d <= after_max || continue

                    jump = BDUtils._grouped_removal_sampling_jump(
                        b, c, ψ̃; labelled_samples=labelled_samples
                    )

                    f_new[d] = get(f_new, d, 0.0) + mass * jump
                end

                f = f_new
                u = ti
            end

            if terminal_condition == :terminated
                return get(f, 0, 0.0)
            elseif terminal_condition == :any
                return sum(values(f))
            else
                throw(ArgumentError("unsupported terminal_condition"))
            end
        end

        @test_throws ArgumentError grouped_sampling_time_likelihood(t0, [0.4, 0.9], [1], removal_pars; tℓ=tl)
        @test_throws ArgumentError grouped_sampling_time_likelihood(t0, Float64[], Int[], removal_pars)
        @test_throws ArgumentError grouped_sampling_time_likelihood(t0, [0.9, 0.4], [1, 1], removal_pars; tℓ=tl)
        @test_throws ArgumentError grouped_sampling_time_likelihood(t0, [0.4], [-1], removal_pars; tℓ=tl)
        @test_throws ArgumentError grouped_sampling_time_likelihood(t0, [0.4], [0], removal_pars; tℓ=tl)
        @test_throws ArgumentError grouped_sampling_time_likelihood(0.4, [0.4], [1], removal_pars; tℓ=tl)
        @test_throws ArgumentError grouped_sampling_time_likelihood(t0, [0.4, 1.9], [1, 1], removal_pars; tℓ=tl)
        @test_throws ArgumentError grouped_sampling_time_likelihood(t0, [0.4], [1], nonremoval_pars; tℓ=tl)
        @test_throws ArgumentError grouped_sampling_time_likelihood(t0, [0.4], [1], removal_pars; tℓ=tl, terminal_condition=:unsupported)

        @test grouped_sampling_time_likelihood(t0, times, counts, removal_pars) ≈
            grouped_sampling_time_likelihood(t0, times, counts, removal_pars; tℓ=last(times))

        t1 = 0.6
        explicit_tl = 1.4
        ψ̃ = transformed_sampling_rate(t1, explicit_tl, removal_pars)
        @test grouped_sampling_time_likelihood(t0, [t1], [1], removal_pars; tℓ=explicit_tl) ≈
            BDUtils._no_sample_reconstructed_kernel(t0, t1, explicit_tl, 1, 1, removal_pars) * ψ̃

        c = 3
        expected_unlabelled =
            BDUtils._no_sample_reconstructed_kernel(t0, t1, explicit_tl, 1, c, removal_pars) *
            binomial(c, c) * ψ̃^c
        expected_labelled =
            BDUtils._no_sample_reconstructed_kernel(t0, t1, explicit_tl, 1, c, removal_pars) *
            BDUtils._falling_factorial(c, c) * ψ̃^c
        @test grouped_sampling_time_likelihood(t0, [t1], [c], removal_pars; tℓ=explicit_tl) ≈ expected_unlabelled
        @test grouped_sampling_time_likelihood(t0, [t1], [c], removal_pars; tℓ=explicit_tl, labelled_samples=true) ≈ expected_labelled

        ratio_times = [0.4, 0.9, 1.3]
        ratio_counts = [2, 1, 3]
        ratio_tl = 1.8
        unlabelled = grouped_sampling_time_likelihood(
            t0, ratio_times, ratio_counts, removal_pars; tℓ=ratio_tl
        )
        labelled = grouped_sampling_time_likelihood(
            t0, ratio_times, ratio_counts, removal_pars;
            tℓ=ratio_tl,
            labelled_samples=true,
        )
        @test labelled ≈ unlabelled * prod(factorial, ratio_counts)

        small_times = [0.5, 1.1]
        small_counts = [1, 2]
        small_tl = 1.4
        @test grouped_sampling_time_likelihood(
            t0, small_times, small_counts, removal_pars; tℓ=small_tl
        ) ≈ brute_grouped_likelihood(
            t0, small_times, small_counts, removal_pars; tℓ=small_tl
        )
        @test grouped_sampling_time_likelihood(
            t0, small_times, small_counts, removal_pars;
            tℓ=small_tl,
            labelled_samples=true,
        ) ≈ brute_grouped_likelihood(
            t0, small_times, small_counts, removal_pars;
            tℓ=small_tl,
            labelled_samples=true,
        )
        @test grouped_sampling_time_likelihood(
            t0, small_times, small_counts, removal_pars;
            tℓ=small_tl,
            terminal_condition=:any,
        ) ≈ brute_grouped_likelihood(
            t0, small_times, small_counts, removal_pars;
            tℓ=small_tl,
            terminal_condition=:any,
        )

        b = 5
        ψ = 0.8
        @test BDUtils._grouped_removal_sampling_jump(b, 2, ψ) ≈ binomial(b, 2) * ψ^2
        @test BDUtils._grouped_removal_sampling_jump(b, 3, ψ; labelled_samples=true) ≈ BDUtils._falling_factorial(b, 3) * ψ^3
        @test BDUtils._grouped_removal_sampling_jump(2, 3, ψ) == 0

        state_lengths = Int[]
        remaining = reverse(cumsum(reverse(counts)))
        f = zeros(Float64, remaining[1] + 1)
        f[2] = 1.0
        u = t0
        for i in eachindex(times)
            before_max = remaining[i]
            g = zeros(Float64, before_max + 1)
            for a in 1:(length(f) - 1), b2 in a:before_max
                g[b2 + 1] += f[a + 1] *
                    BDUtils._no_sample_reconstructed_kernel(u, times[i], tl, a, b2, removal_pars)
            end
            after_max = i == length(times) ? 0 : remaining[i + 1]
            next = zeros(Float64, after_max + 1)
            ψi = transformed_sampling_rate(times[i], tl, removal_pars)
            for b2 in counts[i]:before_max
                d = b2 - counts[i]
                d <= after_max || continue
                next[d + 1] += g[b2 + 1] * BDUtils._grouped_removal_sampling_jump(b2, counts[i], ψi)
            end
            push!(state_lengths, length(next))
            f = next
            u = times[i]
        end
        @test state_lengths == [remaining[2] + 1, remaining[3] + 1, 1]
        @test length(BDUtils._grouped_sampling_time_filter(t0, times, counts, removal_pars; tℓ=tl)) == 1

        terminated = grouped_sampling_time_likelihood(t0, times, counts, removal_pars; tℓ=tl)
        any_terminal = grouped_sampling_time_likelihood(t0, times, counts, removal_pars; tℓ=tl, terminal_condition=:any)
        @test any_terminal ≈ sum(BDUtils._grouped_sampling_time_filter(t0, times, counts, removal_pars; tℓ=tl))
        @test any_terminal >= terminated
    end

    @testset "algebraic identities: conditioned reconstructed process" begin
        identity_pars = (
            ConstantRateBDParameters(1.4, 0.6, 0.7, 1.0),
            ConstantRateBDParameters(1.8, 0.5, 0.7, 0.4),
            ConstantRateBDParameters(0.9, 1.1, 0.4, 0.6),
            ConstantRateBDParameters(1.5, 0.2, 2.5, 0.95),
        )
        triples = ((0.0, 0.55, 1.8), (0.2, 0.9, 2.4))
        quadruples = ((0.0, 0.45, 1.1, 2.0), (0.2, 0.8, 1.5, 2.7))

        function count_convolve(x, y)
            out = zeros(promote_type(eltype(x), eltype(y)), length(x) + length(y) - 1)
            for i in eachindex(x), j in eachindex(y)
                out[i + j - 1] += x[i] * y[j]
            end
            return out
        end

        for reconstructed_pars in identity_pars
            for (ti, tj, tk) in triples
                @test conditioned_reconstructed_pgf(1.0, 1.0, ti, tj, tk, reconstructed_pars) ≈ 1.0 rtol=1e-10 atol=1e-12

                α1 = conditioned_reconstructed_alpha_bd(1.0, ti, tj, tk, reconstructed_pars)
                β1 = conditioned_reconstructed_beta_bd(1.0, ti, tj, tk, reconstructed_pars)
                γ1 = conditioned_reconstructed_gamma_bd(1.0, ti, tj, tk, reconstructed_pars)
                @test α1 + β1 / (1 - γ1) ≈ 1.0 rtol=1e-10 atol=1e-12

                @test conditioned_reconstructed_alpha_bd(0.0, ti, tj, tk, reconstructed_pars) ≈ 0.0 atol=1e-12
                @test conditioned_reconstructed_pgf(0.0, 0.0, ti, tj, tk, reconstructed_pars) ≈ 0.0 atol=1e-12

                β0 = conditioned_reconstructed_beta_bd(0.0, ti, tj, tk, reconstructed_pars)
                γ0 = conditioned_reconstructed_gamma_bd(0.0, ti, tj, tk, reconstructed_pars)
                for z in (0.0, 0.2, 0.6, 0.9)
                    @test conditioned_reconstructed_pgf(z, 0.0, ti, tj, tk, reconstructed_pars) ≈
                        β0 * z / (1 - γ0 * z) rtol=1e-10 atol=1e-12
                end

                ξ = conditioned_reconstructed_xi(ti, tj, tk, reconstructed_pars)
                η = conditioned_reconstructed_eta(ti, tj, tk, reconstructed_pars)
                single = [conditioned_reconstructed_count_pmf(a, ti, tj, tk, reconstructed_pars) for a in 0:400]
                two_lineage = count_convolve(single, single)
                three_lineage = count_convolve(two_lineage, single)
                @test sum(two_lineage) ≈ 1.0 rtol=1e-10 atol=1e-11
                @test sum(three_lineage) ≈ 1.0 rtol=1e-10 atol=1e-11
                @test ξ + (1 - ξ) * (1 - η) / (1 - η) ≈ 1.0 rtol=1e-10 atol=1e-12
            end

            for (ti, tj, tk, tl) in quadruples
                β_ik = conditioned_reconstructed_beta_bd(0.0, ti, tk, tl, reconstructed_pars)
                β_ij = conditioned_reconstructed_beta_bd(0.0, ti, tj, tl, reconstructed_pars)
                β_jk = conditioned_reconstructed_beta_bd(0.0, tj, tk, tl, reconstructed_pars)
                γ_ik = conditioned_reconstructed_gamma_bd(0.0, ti, tk, tl, reconstructed_pars)
                γ_ij = conditioned_reconstructed_gamma_bd(0.0, ti, tj, tl, reconstructed_pars)
                γ_jk = conditioned_reconstructed_gamma_bd(0.0, tj, tk, tl, reconstructed_pars)

                @test β_ik ≈ β_ij * β_jk rtol=1e-10 atol=1e-12
                @test γ_ik ≈ γ_jk + γ_ij * β_jk rtol=1e-10 atol=1e-12

                for z in (0.0, 0.15, 0.5, 0.85)
                    direct = conditioned_reconstructed_pgf(z, 0.0, ti, tk, tl, reconstructed_pars)
                    inner = conditioned_reconstructed_pgf(z, 0.0, tj, tk, tl, reconstructed_pars)
                    composed = conditioned_reconstructed_pgf(inner, 0.0, ti, tj, tl, reconstructed_pars)
                    @test direct ≈ composed rtol=1e-10 atol=1e-12
                end
            end
        end
    end
