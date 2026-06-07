    @testset "multitype simulation foundation" begin
        pars_mt = MultitypeBDParameters(
            [0.0 1.0; 0.5 0.0],
            [0.1, 0.2],
            [0.3, 0.4],
            [0.0, 1.0],
            [0.0 0.7; 0.2 0.0],
            [0.0, 0.0],
        )
        @test length(pars_mt) == 2
        @test sprint(show, pars_mt) == "MultitypeBDParameters(K=2)"

        handcrafted_mt = MultitypeBDEventLog(
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [2, 1, 2, 2, 1],
            [1, 0, 0, 0, 0],
            [MultitypeBirth, MultitypeTransition, MultitypeFossilizedSampling,
             MultitypeSerialSampling, MultitypeDeath],
            [1, 1, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [1],
            1.0,
        )
        @test length(handcrafted_mt) == 5
        @test handcrafted_mt[1] == MultitypeBDEventRecord(0.2, 2, 1, MultitypeBirth, 1, 2)
        @test collect(record.kind for record in handcrafted_mt) == [
            MultitypeBirth,
            MultitypeTransition,
            MultitypeFossilizedSampling,
            MultitypeSerialSampling,
            MultitypeDeath,
        ]
        @test sprint(show, handcrafted_mt) == "MultitypeBDEventLog(5 events, K=2, initial_lineages=1, tmax=1.0)"
        @test validate_multitype_eventlog(handcrafted_mt)

        @test multitype_NS_at(handcrafted_mt, 0.0) == (N=[1, 0], S=[0, 0])
        @test multitype_NS_at(handcrafted_mt, 0.2) == (N=[1, 1], S=[0, 0])
        @test multitype_NS_at(handcrafted_mt, 0.35) == (N=[0, 2], S=[0, 0])
        @test multitype_NS_at(handcrafted_mt, 0.45) == (N=[0, 2], S=[0, 1])
        @test multitype_NS_at(handcrafted_mt, 1.0) == (N=[0, 0], S=[0, 2])
        @test multitype_N_at(handcrafted_mt, 0.35) == [0, 2]
        @test multitype_S_at(handcrafted_mt, 1.0) == [0, 2]
        @test multitype_N_over_time(handcrafted_mt, [0.0, 0.35, 1.0]) == [[1, 0], [0, 2], [0, 0]]
        @test multitype_S_over_time(handcrafted_mt, [0.0, 0.45, 1.0]) == [[0, 0], [0, 1], [0, 2]]
        @test multitype_mean_N([handcrafted_mt, handcrafted_mt], 0.35) == [0.0, 2.0]

        zero_mt = simulate_multitype_bd(
            MersenneTwister(11),
            MultitypeBDParameters(zeros(2, 2), zeros(2), zeros(2), zeros(2), zeros(2, 2), zeros(2)),
            1.0;
            initial_types=[1, 2],
        )
        @test isempty(zero_mt)
        @test multitype_N_at(zero_mt, 1.0) == [1, 1]
        @test multitype_S_at(zero_mt, 1.0) == [0, 0]
        @test validate_multitype_eventlog(zero_mt)

        transition_only = simulate_multitype_bd(
            MersenneTwister(12),
            MultitypeBDParameters(zeros(2, 2), zeros(2), zeros(2), zeros(2), [0.0 20.0; 0.0 0.0], zeros(2)),
            0.5;
            initial_types=[1],
        )
        @test all(==(MultitypeTransition), transition_only.kind)
        @test validate_multitype_eventlog(transition_only)
        @test sum(multitype_N_at(transition_only, 0.5)) == 1

        k1_pars = MultitypeBDParameters([1.0;;], [0.0], [0.0], [0.0], [0.0;;], [0.0])
        k1_log = simulate_multitype_bd(MersenneTwister(1), k1_pars, 0.5)
        @test all(==(MultitypeBirth), k1_log.kind)
        @test validate_multitype_eventlog(k1_log)
        @test multitype_N_at(k1_log, 0.5) == [1 + length(k1_log)]
        @test multitype_S_at(k1_log, 0.5) == [0]

        sampled_k1 = simulate_multitype_bd(
            MersenneTwister(3),
            MultitypeBDParameters([0.0;;], [0.0], [0.0], [1.0], [0.0;;], [1.0]),
            0.0,
        )
        @test sampled_k1.kind == [MultitypeSerialSampling]
        @test multitype_NS_at(sampled_k1, 0.0) == (N=[0], S=[1])

        bad_mt = MultitypeBDEventLog([0.1], [1], [0], [MultitypeDeath], [2], [2], [1], 1.0)
        @test_throws ArgumentError validate_multitype_eventlog(bad_mt)
        @test_throws ArgumentError MultitypeBDParameters(zeros(2, 2), zeros(1), zeros(1), zeros(1), zeros(2, 2))
        @test_throws ArgumentError MultitypeBDParameters([-1.0;;], [0.0], [0.0], [0.0], [0.0;;])
        @test_throws ArgumentError simulate_multitype_bd(k1_pars, 1.0; initial_types=[2])
        @test_throws ArgumentError multitype_NS_at(handcrafted_mt, -0.1)
        @test_throws ArgumentError multitype_mean_N(MultitypeBDEventLog[], 0.0)
    end

    @testset "multitype analytical backbone" begin
        analytical_mt = MultitypeBDParameters(
            [0.6 0.2; 0.1 0.4],
            [0.3, 0.5],
            [0.2, 0.1],
            [0.0, 0.8],
            [0.0 0.15; 0.05 0.0],
            [0.1, 0.25],
        )

        E0 = multitype_E(0.0, analytical_mt)
        @test E0 ≈ [0.9, 0.75]
        @test all(0 .<= multitype_E(1.0, analytical_mt) .<= 1)
        @test all(0 .<= multitype_E(2.0, analytical_mt) .<= 1)
        @test multitype_E_over_time([0.0, 0.5, 1.0], analytical_mt) == [
            multitype_E(0.0, analytical_mt),
            multitype_E(0.5, analytical_mt),
            multitype_E(1.0, analytical_mt),
        ]

        k1_analytical = MultitypeBDParameters([1.4;;], [0.6], [0.3], [0.7], [0.0;;], [0.2])
        @test only(multitype_E(0.0, k1_analytical)) ≈ E_constant(0.0, ConstantRateBDParameters(1.4, 0.6, 0.3, 0.0, 0.2))
        @test only(multitype_E(1.2, k1_analytical; steps_per_unit=1024)) ≈
              E_constant(1.2, ConstantRateBDParameters(1.4, 0.6, 0.3, 0.0, 0.2)) atol=2e-6

        no_observation = MultitypeBDParameters(
            [0.5 0.1; 0.2 0.4],
            [0.3, 0.7],
            zeros(2),
            zeros(2),
            [0.0 0.2; 0.1 0.0],
            zeros(2),
        )
        @test multitype_E(3.0, no_observation) ≈ [1.0, 1.0]

        no_birth = MultitypeBDParameters(zeros(2, 2), [0.4, 0.2], [0.6, 0.8], [0.0, 1.0], zeros(2, 2), [0.0, 0.0])
        E_no_birth = multitype_E(1.5, no_birth)
        @test all(multitype_E(2.0, no_birth) .<= multitype_E(1.0, no_birth) .+ 1e-8)
        @test E_no_birth[1] ≈ 0.4 / 1.0 + (1 - 0.4 / 1.0) * exp(-1.0 * 1.5) atol=1e-8
        @test E_no_birth[2] ≈ 0.2 / 1.0 + (1 - 0.2 / 1.0) * exp(-1.0 * 1.5) atol=1e-8

        death_only_with_present_sampling = MultitypeBDParameters(zeros(1, 1), [0.7], [0.0], [0.0], zeros(1, 1), [0.3])
        @test only(multitype_E(2.0, death_only_with_present_sampling)) ≈ 1 - 0.3 * exp(-0.7 * 2.0) atol=1e-8

        flow_no_birth = multitype_log_flow(1.25, no_birth)
        @test flow_no_birth ≈ [-1.0 * 1.25, -1.0 * 1.25] atol=1e-8
        @test multitype_flow(1.25, no_birth) ≈ exp.(flow_no_birth)
        @test multitype_log_flow(0.0, analytical_mt) == [0.0, 0.0]

        rng = MersenneTwister(20260417)
        sim_pars = MultitypeBDParameters(zeros(1, 1), [0.4], [0.6], [1.0], zeros(1, 1), [0.0])
        nsims = 2_000
        no_sample_count = 0
        for _ in 1:nsims
            log = simulate_multitype_bd(rng, sim_pars, 1.0; apply_ρ₀=true)
            no_sample_count += multitype_S_at(log, 1.0)[1] == 0
        end
        @test no_sample_count / nsims ≈ only(multitype_E(1.0, sim_pars)) atol=0.04

        @test_throws ArgumentError multitype_E(-0.1, analytical_mt)
        @test_throws ArgumentError multitype_E(1.0, analytical_mt; steps_per_unit=0)
        @test_throws ArgumentError multitype_E_over_time([0.0, Inf], analytical_mt)
        @test_throws ArgumentError multitype_log_flow(NaN, analytical_mt)
    end

    @testset "multitype fully colored likelihood" begin
        one_type_pars = MultitypeBDParameters([1.2;;], [0.4], [0.3], [0.5], [0.0;;], [0.8])
        one_type_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.0, 1.0)],
            present_samples=[1],
        )
        @test validate_multitype_colored_tree(one_type_tree, one_type_pars)
        expected_one_type = log(0.8) + only(multitype_log_flow(1.0, one_type_pars))
        @test multitype_colored_loglikelihood(one_type_tree, one_type_pars) ≈ expected_one_type
        @test multitype_colored_likelihood(one_type_tree, one_type_pars) ≈ exp(expected_one_type)

        two_type_pars = MultitypeBDParameters(
            [0.5 0.7; 0.2 0.4],
            [0.2, 0.3],
            [0.6, 0.5],
            [0.25, 0.8],
            [0.0 0.9; 0.1 0.0],
            [0.4, 0.6],
        )
        birth_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.5, 1.0),
                MultitypeColoredSegment(1, 0.0, 0.5),
                MultitypeColoredSegment(2, 0.0, 0.5),
            ],
            births=[MultitypeColoredBirth(0.5, 1, 2)],
            present_samples=[1, 2],
        )
        logflow_05 = multitype_log_flow(0.5, two_type_pars)
        logflow_10 = multitype_log_flow(1.0, two_type_pars)
        expected_birth = (logflow_10[1] - logflow_05[1]) + logflow_05[1] + logflow_05[2] +
                         log(two_type_pars.birth[1, 2]) + log(two_type_pars.ρ₀[1]) + log(two_type_pars.ρ₀[2])
        @test multitype_colored_loglikelihood(birth_tree, two_type_pars) ≈ expected_birth

        transition_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.4, 1.0),
                MultitypeColoredSegment(2, 0.0, 0.4),
            ],
            transitions=[MultitypeColoredTransition(0.4, 1, 2)],
            present_samples=[2],
        )
        expected_transition = (logflow_10[1] - multitype_log_flow(0.4, two_type_pars)[1]) +
                              multitype_log_flow(0.4, two_type_pars)[2] +
                              log(two_type_pars.transition[1, 2]) + log(two_type_pars.ρ₀[2])
        @test multitype_colored_loglikelihood(transition_tree, two_type_pars) ≈ expected_transition

        hidden_birth_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.4, 1.0),
                MultitypeColoredSegment(2, 0.0, 0.4),
            ],
            hidden_births=[MultitypeColoredHiddenBirth(0.4, 1, 2)],
            present_samples=[2],
        )
        E04 = multitype_E(0.4, two_type_pars)[1]
        expected_hidden_birth = (logflow_10[1] - multitype_log_flow(0.4, two_type_pars)[1]) +
                                multitype_log_flow(0.4, two_type_pars)[2] +
                                log(two_type_pars.birth[1, 2] * E04) + log(two_type_pars.ρ₀[2])
        @test multitype_colored_loglikelihood(hidden_birth_tree, two_type_pars) ≈ expected_hidden_birth
        @test multitype_colored_loglikelihood(hidden_birth_tree, two_type_pars) !=
              multitype_colored_loglikelihood(transition_tree, two_type_pars)

        sampling_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.7, 1.0),
                MultitypeColoredSegment(1, 0.2, 0.7),
            ],
            terminal_samples=[MultitypeColoredSampling(0.2, 1)],
            ancestral_samples=[MultitypeColoredSampling(0.7, 1)],
        )
        E02 = multitype_E(0.2, two_type_pars)[1]
        lf02 = multitype_log_flow(0.2, two_type_pars)
        lf07 = multitype_log_flow(0.7, two_type_pars)
        expected_sampling = (logflow_10[1] - lf07[1]) + (lf07[1] - lf02[1]) +
                            log(two_type_pars.sampling[1] * (1 - two_type_pars.removal_probability[1])) +
                            log(two_type_pars.sampling[1] * (two_type_pars.removal_probability[1] +
                                (1 - two_type_pars.removal_probability[1]) * E02))
        @test multitype_colored_loglikelihood(sampling_tree, two_type_pars) ≈ expected_sampling

        higher_birth = MultitypeBDParameters(
            [0.5 1.4; 0.2 0.4],
            two_type_pars.death,
            two_type_pars.sampling,
            two_type_pars.removal_probability,
            two_type_pars.transition,
            two_type_pars.ρ₀,
        )
        @test multitype_colored_loglikelihood(birth_tree, higher_birth) >
              multitype_colored_loglikelihood(birth_tree, two_type_pars)

        bad_root = MultitypeColoredTree(1.0, 1; segments=[MultitypeColoredSegment(2, 0.0, 1.0)])
        @test_throws ArgumentError validate_multitype_colored_tree(bad_root, two_type_pars)
        @test_throws ArgumentError MultitypeColoredTree(-1.0, 1; segments=[MultitypeColoredSegment(1, 0.0, 1.0)]) |>
                                   tree -> validate_multitype_colored_tree(tree, one_type_pars)
        bad_birth = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.0, 1.0)],
            births=[MultitypeColoredBirth(0.5, 1, 2)],
        )
        @test_throws ArgumentError multitype_colored_loglikelihood(bad_birth, one_type_pars)
        bad_hidden_birth = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.0, 1.0)],
            hidden_births=[MultitypeColoredHiddenBirth(0.5, 1, 2)],
        )
        @test_throws ArgumentError multitype_colored_loglikelihood(bad_hidden_birth, one_type_pars)
        impossible_present = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.0, 1.0)],
            present_samples=[1],
        )
        no_present_sampling = MultitypeBDParameters([0.0;;], [0.1], [0.0], [0.0], [0.0;;], [0.0])
        @test_throws ArgumentError multitype_colored_loglikelihood(impossible_present, no_present_sampling)
    end

    @testset "uncoloured MTBD-2 likelihood" begin
        mtbd2_pars = UncolouredMTBD2ConstantParameters(
            [0.8 0.35; 0.25 0.7],
            [0.2, 0.3],
            [0.45, 0.55],
            [0.7, 0.4],
            [0.0 0.12; 0.08 0.0],
            [0.2, 0.3],
        )
        tree = tiny_tree()
        tip_states = Dict(3 => 1, 4 => 2)

        ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, tip_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8)
        @test isfinite(ll)
        @test likelihood_uncoloured_mtbd2(tree, mtbd2_pars, tip_states;
                                                     root_prior=[0.6, 0.4],
                                                     steps_per_unit=64,
                                                     min_steps=8) ≈ exp(ll)
        @test ll ≈ -4.404102999291537
        @test loglikelihood_uncoloured_mtbd2_known_tips(tree, mtbd2_pars, tip_states;
                                                        root_prior=[0.6, 0.4],
                                                        steps_per_unit=64,
                                                        min_steps=8) ≈ ll
        @test likelihood_uncoloured_mtbd2_known_tips(tree, mtbd2_pars, tip_states;
                                                     root_prior=[0.6, 0.4],
                                                     steps_per_unit=64,
                                                     min_steps=8) ≈ exp(ll)

        vector_tip_states = [0, 0, 1, 2]
        @test loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, vector_tip_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ ll
        @test loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => (1,), 4 => [2]);
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ ll

        unknown_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => missing, 4 => nothing);
                                                              root_prior=[0.6, 0.4],
                                                              steps_per_unit=64,
                                                              min_steps=8)
        @test isfinite(unknown_ll)
        @test unknown_ll >= ll
        @test loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => (1, 2), 4 => [true, true]);
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ unknown_ll

        mixed_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => 1, 4 => :);
                                                            root_prior=[0.6, 0.4],
                                                            steps_per_unit=64,
                                                            min_steps=8)
        @test isfinite(mixed_ll)
        @test ll <= mixed_ll <= unknown_ll

        swapped = Tree(
            [0.0, 0.6, 1.0, 1.4],
            [2, 4, 0, 0],
            [0, 3, 0, 0],
            [0, 1, 2, 2],
            [Root, Binary, SampledLeaf, SampledLeaf],
            [0, 0, 0, 0],
            [0, 0, 101, 102],
        )
        @test loglikelihood_uncoloured_mtbd2(swapped, mtbd2_pars, tip_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ ll
        @test loglikelihood_uncoloured_mtbd2(swapped, mtbd2_pars, Dict(3 => missing, 4 => nothing);
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ unknown_ll

        present_tip_tree = Tree(
            [0.0, 1.0, 1.0],
            [2, 0, 0],
            [3, 0, 0],
            [0, 1, 1],
            [Root, SampledLeaf, SampledLeaf],
            [0, 0, 0],
            [0, 101, 102],
        )
        present_pars = UncolouredMTBD2ConstantParameters(
            [0.9 0.1; 0.2 0.8],
            [0.1, 0.1],
            [0.3, 0.3],
            [1.0, 1.0],
            zeros(2, 2),
            [0.25, 0.75],
        )
        root1_ll = loglikelihood_uncoloured_mtbd2(present_tip_tree, present_pars, Dict(2 => 1, 3 => 2);
                                                             root_prior=[1.0, 0.0])
        root2_ll = loglikelihood_uncoloured_mtbd2(present_tip_tree, present_pars, Dict(2 => 1, 3 => 2);
                                                             root_prior=[0.0, 1.0])
        richer_present_pars = UncolouredMTBD2ConstantParameters(
            present_pars.birth,
            present_pars.death,
            present_pars.sampling,
            present_pars.removal_probability,
            present_pars.transition,
            [0.5, 0.9],
        )
        richer_present_ll = loglikelihood_uncoloured_mtbd2(present_tip_tree, richer_present_pars, Dict(2 => 1, 3 => 2);
                                                                      root_prior=[1.0, 0.0])
        @test isfinite(root1_ll)
        @test isfinite(root2_ll)
        @test root1_ll != root2_ll
        @test richer_present_ll > root1_ll

        sampled_unary_tree = Tree(
            [0.0, 0.4, 0.8, 1.0, 1.2],
            [2, 3, 4, 0, 0],
            [0, 5, 0, 0, 0],
            [0, 1, 2, 3, 2],
            [Root, Binary, SampledUnary, SampledLeaf, SampledLeaf],
            [0, 0, 0, 0, 0],
            [0, 0, 201, 202, 203],
        )
        sampled_unary_states = Dict(3 => 1, 4 => 2, 5 => 1)
        sampled_unary_ll = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars, sampled_unary_states;
                                                          root_prior=[0.6, 0.4],
                                                          steps_per_unit=64,
                                                          min_steps=8)
        @test isfinite(sampled_unary_ll)

        sampled_unary_unknown_ll = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars,
                                                                  Dict(3 => missing, 4 => nothing, 5 => :);
                                                                  root_prior=[0.6, 0.4],
                                                                  steps_per_unit=64,
                                                                  min_steps=8)
        sampled_unary_mixed_ll = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars,
                                                                Dict(3 => 1, 4 => missing, 5 => 1);
                                                                root_prior=[0.6, 0.4],
                                                                steps_per_unit=64,
                                                                min_steps=8)
        @test isfinite(sampled_unary_unknown_ll)
        @test isfinite(sampled_unary_mixed_ll)
        @test sampled_unary_ll <= sampled_unary_mixed_ll <= sampled_unary_unknown_ll

        sampled_unary_swapped = Tree(
            [0.0, 0.4, 0.8, 1.0, 1.2],
            [2, 5, 4, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 1, 2, 3, 2],
            [Root, Binary, SampledUnary, SampledLeaf, SampledLeaf],
            [0, 0, 0, 0, 0],
            [0, 0, 201, 202, 203],
        )
        @test loglikelihood_uncoloured_mtbd2(sampled_unary_swapped, mtbd2_pars, sampled_unary_states;
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8) ≈ sampled_unary_ll
        @test loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars, Dict(3 => [1], 4 => [false, true], 5 => (1,));
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8) ≈ sampled_unary_ll

        batch_trees = [tree, swapped, sampled_unary_tree]
        batch_observations = [
            tip_states,
            Dict(3 => missing, 4 => nothing),
            Dict(3 => 1, 4 => missing, 5 => 1),
        ]
        repeated_batch_ll = [
            loglikelihood_uncoloured_mtbd2(batch_trees[i], mtbd2_pars, batch_observations[i];
                                           root_prior=[0.6, 0.4],
                                           steps_per_unit=64,
                                           min_steps=8)
            for i in eachindex(batch_trees)
        ]
        batch_ll = loglikelihoods_uncoloured_mtbd2(batch_trees, mtbd2_pars, batch_observations;
                                                   root_prior=[0.6, 0.4],
                                                   steps_per_unit=64,
                                                   min_steps=8)
        @test batch_ll ≈ repeated_batch_ll
        @test likelihoods_uncoloured_mtbd2(batch_trees, mtbd2_pars, batch_observations;
                                           root_prior=[0.6, 0.4],
                                           steps_per_unit=64,
                                           min_steps=8) ≈ exp.(batch_ll)
        @test total_loglikelihood_uncoloured_mtbd2(batch_trees, mtbd2_pars, batch_observations;
                                                   root_prior=[0.6, 0.4],
                                                   steps_per_unit=64,
                                                   min_steps=8) ≈ sum(batch_ll)
        batch_score = score_uncoloured_mtbd2(batch_trees, mtbd2_pars, batch_observations;
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8)
        @test batch_score.per_tree_loglikelihood ≈ batch_ll
        @test batch_score.total_loglikelihood ≈ sum(batch_ll)
        @test batch_score.mean_loglikelihood ≈ sum(batch_ll) / length(batch_ll)
        @test batch_score.n_scored == 3
        @test_throws ArgumentError loglikelihoods_uncoloured_mtbd2(batch_trees, mtbd2_pars, batch_observations[1:2])

        θ = uncoloured_mtbd2_parameter_vector(mtbd2_pars)
        @test length(θ) == length(UNCOLOURED_MTBD2_PARAMETER_ORDER) == 16
        @test θ == [
            mtbd2_pars.birth[1, 1], mtbd2_pars.birth[1, 2], mtbd2_pars.birth[2, 1], mtbd2_pars.birth[2, 2],
            mtbd2_pars.death[1], mtbd2_pars.death[2],
            mtbd2_pars.sampling[1], mtbd2_pars.sampling[2],
            mtbd2_pars.removal_probability[1], mtbd2_pars.removal_probability[2],
            mtbd2_pars.transition[1, 1], mtbd2_pars.transition[1, 2], mtbd2_pars.transition[2, 1], mtbd2_pars.transition[2, 2],
            mtbd2_pars.ρ₀[1], mtbd2_pars.ρ₀[2],
        ]
        roundtrip_pars = uncoloured_mtbd2_parameters_from_vector(θ)
        @test roundtrip_pars.birth == mtbd2_pars.birth
        @test roundtrip_pars.death == mtbd2_pars.death
        @test roundtrip_pars.sampling == mtbd2_pars.sampling
        @test roundtrip_pars.removal_probability == mtbd2_pars.removal_probability
        @test roundtrip_pars.transition == mtbd2_pars.transition
        @test roundtrip_pars.ρ₀ == mtbd2_pars.ρ₀

        all_free_spec = UncolouredMTBD2ParameterSpec(mtbd2_pars)
        @test free_parameter_vector(all_free_spec) == θ
        @test free_parameter_vector(mtbd2_pars, all_free_spec) == θ
        all_free_rebuilt = uncoloured_mtbd2_parameters_from_free_vector(θ, all_free_spec)
        @test uncoloured_mtbd2_parameter_vector(all_free_rebuilt) == θ

        partial_spec = UncolouredMTBD2ParameterSpec(
            mtbd2_pars;
            birth=[true false; false true],
            death=[false, true],
            sampling=[true, false],
            removal_probability=[false, false],
            transition=[false true; true false],
            ρ₀=[true, false],
        )
        partial_free = free_parameter_vector(mtbd2_pars, partial_spec)
        @test partial_free == [
            mtbd2_pars.birth[1, 1],
            mtbd2_pars.birth[2, 2],
            mtbd2_pars.death[2],
            mtbd2_pars.sampling[1],
            mtbd2_pars.transition[1, 2],
            mtbd2_pars.transition[2, 1],
            mtbd2_pars.ρ₀[1],
        ]
        changed_free = partial_free .+ [0.1, 0.2, 0.03, 0.04, 0.01, 0.02, -0.05]
        changed_pars = uncoloured_mtbd2_parameters_from_free_vector(changed_free, partial_spec)
        @test changed_pars.birth[1, 1] == changed_free[1]
        @test changed_pars.birth[1, 2] == mtbd2_pars.birth[1, 2]
        @test changed_pars.birth[2, 1] == mtbd2_pars.birth[2, 1]
        @test changed_pars.birth[2, 2] == changed_free[2]
        @test changed_pars.death[1] == mtbd2_pars.death[1]
        @test changed_pars.death[2] == changed_free[3]
        @test changed_pars.sampling[1] == changed_free[4]
        @test changed_pars.sampling[2] == mtbd2_pars.sampling[2]
        @test changed_pars.transition[1, 2] == changed_free[5]
        @test changed_pars.transition[2, 1] == changed_free[6]
        @test changed_pars.ρ₀[1] == changed_free[7]
        @test changed_pars.ρ₀[2] == mtbd2_pars.ρ₀[2]

        fixed_spec = UncolouredMTBD2ParameterSpec(mtbd2_pars; birth=falses(2, 2), death=falses(2),
                                                  sampling=falses(2), removal_probability=falses(2),
                                                  transition=falses(2, 2), ρ₀=falses(2))
        @test isempty(free_parameter_vector(fixed_spec))
        @test uncoloured_mtbd2_parameter_vector(uncoloured_mtbd2_parameters_from_free_vector(Float64[], fixed_spec)) == θ

        @test loglikelihood_uncoloured_mtbd2_from_free(θ, tree, all_free_spec, tip_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ ll
        @test loglikelihood_uncoloured_mtbd2_from_free(θ, sampled_unary_tree, all_free_spec, sampled_unary_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8) ≈ sampled_unary_ll
        @test total_loglikelihood_uncoloured_mtbd2_from_free(θ, batch_trees, all_free_spec, batch_observations;
                                                             root_prior=[0.6, 0.4],
                                                             steps_per_unit=64,
                                                             min_steps=8) ≈ sum(batch_ll)
        @test loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, tip_states;
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8) ≈ ll

        @test_throws ArgumentError uncoloured_mtbd2_parameters_from_vector(θ[1:15])
        @test_throws ArgumentError uncoloured_mtbd2_parameters_from_vector(vcat(-0.1, θ[2:end]))
        @test_throws ArgumentError UncolouredMTBD2ParameterSpec(mtbd2_pars; birth=trues(3, 2))
        @test_throws ArgumentError UncolouredMTBD2ParameterSpec(mtbd2_pars; death=trues(3))
        @test_throws ArgumentError uncoloured_mtbd2_parameters_from_free_vector(partial_free[1:6], partial_spec)

        mle_spec = UncolouredMTBD2ParameterSpec(
            mtbd2_pars;
            birth=falses(2, 2),
            death=falses(2),
            sampling=[true, false],
            removal_probability=falses(2),
            transition=falses(2, 2),
            ρ₀=falses(2),
        )
        θ0_mle = free_parameter_vector(mtbd2_pars, mle_spec)
        initial_single_nll = -loglikelihood_uncoloured_mtbd2_from_free(θ0_mle, tree, mle_spec, tip_states;
                                                                       root_prior=[0.6, 0.4],
                                                                       steps_per_unit=64,
                                                                       min_steps=8)
        single_fit = fit_uncoloured_mtbd2_mle(tree, θ0_mle, mle_spec, tip_states;
                                              root_prior=[0.6, 0.4],
                                              steps_per_unit=64,
                                              min_steps=8,
                                              lower=[1e-6],
                                              upper=[2.0],
                                              initial_step=0.05,
                                              tolerance=1e-4,
                                              maxiter=100)
        @test length(single_fit.θ_free_hat) == 1
        @test single_fit.params_hat isa UncolouredMTBD2ConstantParameters
        @test isfinite(single_fit.maximum_loglikelihood)
        @test isfinite(single_fit.minimum_negloglikelihood)
        @test single_fit.minimum_negloglikelihood ≈ -single_fit.maximum_loglikelihood
        @test single_fit.initial_negloglikelihood ≈ initial_single_nll
        @test single_fit.minimum_negloglikelihood <= initial_single_nll + 1e-8
        @test single_fit.spec === mle_spec
        @test haskey(single_fit.optimizer_summary, :minimizer)

        single_fit_from_params = fit_uncoloured_mtbd2_mle(tree, mtbd2_pars, mle_spec, tip_states;
                                                         root_prior=[0.6, 0.4],
                                                         steps_per_unit=64,
                                                         min_steps=8,
                                                         lower=[1e-6],
                                                         upper=[2.0],
                                                         initial_step=0.05,
                                                         tolerance=1e-4,
                                                         maxiter=100)
        @test single_fit_from_params.initial_negloglikelihood ≈ single_fit.initial_negloglikelihood
        @test single_fit_from_params.minimum_negloglikelihood ≈ single_fit.minimum_negloglikelihood

        batch_fit = fit_uncoloured_mtbd2_mle(batch_trees, mtbd2_pars, mle_spec, batch_observations;
                                            root_prior=[0.6, 0.4],
                                            steps_per_unit=64,
                                            min_steps=8,
                                            lower=[1e-6],
                                            upper=[2.0],
                                            initial_step=0.05,
                                            tolerance=1e-4,
                                            maxiter=100)
        initial_batch_nll = -total_loglikelihood_uncoloured_mtbd2_from_free(θ0_mle, batch_trees, mle_spec, batch_observations;
                                                                            root_prior=[0.6, 0.4],
                                                                            steps_per_unit=64,
                                                                            min_steps=8)
        @test isfinite(batch_fit.maximum_loglikelihood)
        @test batch_fit.minimum_negloglikelihood ≈ -batch_fit.maximum_loglikelihood
        @test batch_fit.initial_negloglikelihood ≈ initial_batch_nll
        @test batch_fit.minimum_negloglikelihood <= initial_batch_nll + 1e-8

        @test_throws ArgumentError fit_uncoloured_mtbd2_mle(tree, Float64[], fixed_spec, tip_states)
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle(Tree[], θ0_mle, mle_spec, Dict[])
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle(batch_trees, θ0_mle, mle_spec, batch_observations[1:2])
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle(tree, Float64[], mle_spec, tip_states)
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle(tree, θ0_mle, mle_spec, tip_states; lower=[0.0, 0.0])

        transform_spec = UncolouredMTBD2ParameterSpec(
            mtbd2_pars;
            birth=[true false; false false],
            death=[false, true],
            sampling=[true, false],
            removal_probability=[true, false],
            transition=[false true; false false],
            ρ₀=[true, false],
        )
        θ_transform = free_parameter_vector(mtbd2_pars, transform_spec)
        η_transform = uncoloured_mtbd2_unconstrained_from_free(θ_transform, transform_spec)
        @test uncoloured_mtbd2_free_from_unconstrained(η_transform, transform_spec) ≈ θ_transform
        @test uncoloured_mtbd2_unconstrained_from_free(free_parameter_vector(mtbd2_pars, transform_spec), transform_spec) ≈ η_transform
        prob_spec = UncolouredMTBD2ParameterSpec(
            mtbd2_pars;
            birth=falses(2, 2),
            death=falses(2),
            sampling=falses(2),
            removal_probability=[true, false],
            transition=falses(2, 2),
            ρ₀=[true, false],
        )
        prob_free = free_parameter_vector(mtbd2_pars, prob_spec)
        prob_eta = uncoloured_mtbd2_unconstrained_from_free(prob_free, prob_spec)
        @test prob_eta ≈ log.(prob_free ./ (1 .- prob_free))
        @test uncoloured_mtbd2_free_from_unconstrained(prob_eta, prob_spec) ≈ prob_free

        @test loglikelihood_uncoloured_mtbd2_from_unconstrained(η_transform, tree, transform_spec, tip_states;
                                                                root_prior=[0.6, 0.4],
                                                                steps_per_unit=64,
                                                                min_steps=8) ≈
              loglikelihood_uncoloured_mtbd2_from_free(θ_transform, tree, transform_spec, tip_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8)
        @test loglikelihood_uncoloured_mtbd2_from_unconstrained(η_transform, sampled_unary_tree, transform_spec, sampled_unary_states;
                                                                root_prior=[0.6, 0.4],
                                                                steps_per_unit=64,
                                                                min_steps=8) ≈
              loglikelihood_uncoloured_mtbd2_from_free(θ_transform, sampled_unary_tree, transform_spec, sampled_unary_states;
                                                       root_prior=[0.6, 0.4],
                                                       steps_per_unit=64,
                                                       min_steps=8)
        @test total_loglikelihood_uncoloured_mtbd2_from_unconstrained(η_transform, batch_trees, transform_spec, batch_observations;
                                                                      root_prior=[0.6, 0.4],
                                                                      steps_per_unit=64,
                                                                      min_steps=8) ≈
              total_loglikelihood_uncoloured_mtbd2_from_free(θ_transform, batch_trees, transform_spec, batch_observations;
                                                             root_prior=[0.6, 0.4],
                                                             steps_per_unit=64,
                                                             min_steps=8)

        transformed_fit = fit_uncoloured_mtbd2_mle_transformed(tree, θ0_mle, mle_spec, tip_states;
                                                              root_prior=[0.6, 0.4],
                                                              steps_per_unit=64,
                                                              min_steps=8,
                                                              initial_step=0.05,
                                                              tolerance=1e-4,
                                                              maxiter=100)
        @test length(transformed_fit.η_free_hat) == length(transformed_fit.θ_free_hat) == 1
        @test transformed_fit.params_hat isa UncolouredMTBD2ConstantParameters
        @test transformed_fit.θ_free_hat ≈ free_parameter_vector(transformed_fit.params_hat, mle_spec)
        @test transformed_fit.minimum_negloglikelihood ≈ -transformed_fit.maximum_loglikelihood
        @test transformed_fit.minimum_negloglikelihood <= transformed_fit.initial_negloglikelihood + 1e-8
        @test transformed_fit.spec === mle_spec

        transformed_fit_from_eta = fit_uncoloured_mtbd2_mle_transformed(tree, log.(θ0_mle), mle_spec, tip_states;
                                                                        init_is_unconstrained=true,
                                                                        root_prior=[0.6, 0.4],
                                                                        steps_per_unit=64,
                                                                        min_steps=8,
                                                                        initial_step=0.05,
                                                                        tolerance=1e-4,
                                                                        maxiter=100)
        @test transformed_fit_from_eta.initial_negloglikelihood ≈ transformed_fit.initial_negloglikelihood

        transformed_batch_fit = fit_uncoloured_mtbd2_mle_transformed(batch_trees, mtbd2_pars, mle_spec, batch_observations;
                                                                     root_prior=[0.6, 0.4],
                                                                     steps_per_unit=64,
                                                                     min_steps=8,
                                                                     initial_step=0.05,
                                                                     tolerance=1e-4,
                                                                     maxiter=100)
        @test isfinite(transformed_batch_fit.maximum_loglikelihood)
        @test transformed_batch_fit.minimum_negloglikelihood <= transformed_batch_fit.initial_negloglikelihood + 1e-8

        @test_throws ArgumentError uncoloured_mtbd2_unconstrained_from_free([0.0], mle_spec)
        @test_throws ArgumentError uncoloured_mtbd2_unconstrained_from_free([1.0], prob_spec)
        @test_throws ArgumentError uncoloured_mtbd2_free_from_unconstrained(Float64[], mle_spec)
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle_transformed(tree, Float64[], fixed_spec, tip_states)
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle_transformed(tree, mtbd2_pars, mle_spec, tip_states; init_is_unconstrained=true)
        @test_throws ArgumentError fit_uncoloured_mtbd2_mle_transformed(batch_trees, θ0_mle, mle_spec, batch_observations[1:2])

        superspreader_pars = UncolouredMTBD2SuperspreaderParameters(
            1.8,
            0.2,
            6.0,
            [0.15, 0.25],
            [0.5, 0.7],
            [0.6, 0.4],
            [0.1, 0.2],
        )
        native_from_superspreader = uncoloured_mtbd2_native_parameters(superspreader_pars)
        q = superspreader_pars.superspreader_fraction
        ptypes = [1 - q, q]
        τ = [1.0, superspreader_pars.relative_transmissibility]
        δ1 = superspreader_pars.death[1] + superspreader_pars.sampling[1] * superspreader_pars.removal_probability[1]
        δ2 = superspreader_pars.death[2] + superspreader_pars.sampling[2] * superspreader_pars.removal_probability[2]
        c = superspreader_pars.total_R0 / (ptypes[1] * τ[1] + ptypes[2] * τ[2])
        λ1 = c * τ[1] * δ1
        λ2 = c * τ[2] * δ2
        @test native_from_superspreader.birth ≈ [λ1 * ptypes[1] λ1 * ptypes[2];
                                                 λ2 * ptypes[1] λ2 * ptypes[2]]
        @test native_from_superspreader.death == superspreader_pars.death
        @test native_from_superspreader.sampling == superspreader_pars.sampling
        @test native_from_superspreader.removal_probability == superspreader_pars.removal_probability
        @test native_from_superspreader.transition == zeros(2, 2)
        @test native_from_superspreader.ρ₀ == superspreader_pars.ρ₀
        @test sum(ptypes[i] * (sum(native_from_superspreader.birth[i, :]) / (i == 1 ? δ1 : δ2)) for i in 1:2) ≈
              superspreader_pars.total_R0
        @test sum(native_from_superspreader.birth[2, :]) / δ2 ≈
              superspreader_pars.relative_transmissibility * sum(native_from_superspreader.birth[1, :]) / δ1

        superspreader_vector = uncoloured_mtbd2_superspreader_parameter_vector(superspreader_pars)
        @test length(superspreader_vector) == length(UNCOLOURED_MTBD2_SUPERSPREADER_PARAMETER_ORDER) == 11
        superspreader_roundtrip = uncoloured_mtbd2_superspreader_parameters_from_vector(superspreader_vector)
        @test uncoloured_mtbd2_superspreader_parameter_vector(superspreader_roundtrip) == superspreader_vector

        @test loglikelihood_uncoloured_mtbd2_superspreader(tree, superspreader_pars, tip_states;
                                                           root_prior=[0.6, 0.4],
                                                           steps_per_unit=64,
                                                           min_steps=8) ≈
              loglikelihood_uncoloured_mtbd2(tree, native_from_superspreader, tip_states;
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8)
        @test likelihood_uncoloured_mtbd2_superspreader(tree, superspreader_pars, tip_states;
                                                        root_prior=[0.6, 0.4],
                                                        steps_per_unit=64,
                                                        min_steps=8) ≈
              exp(loglikelihood_uncoloured_mtbd2(tree, native_from_superspreader, tip_states;
                                                 root_prior=[0.6, 0.4],
                                                 steps_per_unit=64,
                                                 min_steps=8))
        @test total_loglikelihood_uncoloured_mtbd2_superspreader(batch_trees, superspreader_pars, batch_observations;
                                                                 root_prior=[0.6, 0.4],
                                                                 steps_per_unit=64,
                                                                 min_steps=8) ≈
              total_loglikelihood_uncoloured_mtbd2(batch_trees, native_from_superspreader, batch_observations;
                                                   root_prior=[0.6, 0.4],
                                                   steps_per_unit=64,
                                                   min_steps=8)

        @test_throws ArgumentError UncolouredMTBD2SuperspreaderParameters(0.0, 0.2, 6.0, [0.15, 0.25], [0.5, 0.7], [0.6, 0.4], [0.1, 0.2])
        @test_throws ArgumentError UncolouredMTBD2SuperspreaderParameters(1.8, 0.0, 6.0, [0.15, 0.25], [0.5, 0.7], [0.6, 0.4], [0.1, 0.2])
        @test_throws ArgumentError UncolouredMTBD2SuperspreaderParameters(1.8, 0.2, 0.0, [0.15, 0.25], [0.5, 0.7], [0.6, 0.4], [0.1, 0.2])
        @test_throws ArgumentError UncolouredMTBD2SuperspreaderParameters(1.8, 0.2, 6.0, [0.0, 0.25], [0.0, 0.7], [0.0, 0.4], [0.1, 0.2])
        @test_throws ArgumentError uncoloured_mtbd2_superspreader_parameters_from_vector(superspreader_vector[1:10])

        superspreader_roundtrip2 = uncoloured_mtbd2_superspreader_parameters_from_vector(
            uncoloured_mtbd2_superspreader_parameter_vector(superspreader_pars),
        )
        @test superspreader_roundtrip2.total_R0 == superspreader_pars.total_R0
        @test superspreader_roundtrip2.superspreader_fraction == superspreader_pars.superspreader_fraction
        @test superspreader_roundtrip2.relative_transmissibility == superspreader_pars.relative_transmissibility
        @test superspreader_roundtrip2.death == superspreader_pars.death
        @test superspreader_roundtrip2.sampling == superspreader_pars.sampling
        @test superspreader_roundtrip2.removal_probability == superspreader_pars.removal_probability
        @test superspreader_roundtrip2.ρ₀ == superspreader_pars.ρ₀

        superspreader_spec = UncolouredMTBD2SuperspreaderSpec(
            superspreader_pars;
            total_R0=true,
            superspreader_fraction=false,
            relative_transmissibility=true,
            death=[true, false],
            sampling=[false, true],
            removal_probability=[true, false],
            ρ₀=[false, true],
        )
        θ_super_free = free_parameter_vector(superspreader_pars, superspreader_spec)
        @test θ_super_free == [
            superspreader_pars.total_R0,
            superspreader_pars.relative_transmissibility,
            superspreader_pars.death[1],
            superspreader_pars.sampling[2],
            superspreader_pars.removal_probability[1],
            superspreader_pars.ρ₀[2],
        ]
        superspreader_rebuilt = uncoloured_mtbd2_superspreader_parameters_from_free_vector(
            θ_super_free, superspreader_spec,
        )
        @test uncoloured_mtbd2_superspreader_parameter_vector(superspreader_rebuilt) == superspreader_vector
        θ_super_changed = copy(θ_super_free)
        θ_super_changed[1] = 2.1
        θ_super_changed[2] = 4.0
        superspreader_changed = uncoloured_mtbd2_superspreader_parameters_from_free_vector(
            θ_super_changed, superspreader_spec,
        )
        @test superspreader_changed.total_R0 == 2.1
        @test superspreader_changed.superspreader_fraction == superspreader_pars.superspreader_fraction
        @test superspreader_changed.relative_transmissibility == 4.0
        @test superspreader_changed.death[2] == superspreader_pars.death[2]
        @test_throws ArgumentError UncolouredMTBD2SuperspreaderSpec(superspreader_pars; death=[true])
        @test_throws ArgumentError uncoloured_mtbd2_superspreader_parameters_from_free_vector(θ_super_free[1:5], superspreader_spec)

        η_super_free = uncoloured_mtbd2_superspreader_unconstrained_from_free(θ_super_free, superspreader_spec)
        @test uncoloured_mtbd2_superspreader_free_from_unconstrained(η_super_free, superspreader_spec) ≈ θ_super_free
        @test_throws ArgumentError uncoloured_mtbd2_superspreader_unconstrained_from_free([0.0], UncolouredMTBD2SuperspreaderSpec(superspreader_pars; total_R0=true, superspreader_fraction=false, relative_transmissibility=false, death=falses(2), sampling=falses(2), removal_probability=falses(2), ρ₀=falses(2)))

        @test loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_super_free, tree, superspreader_spec, tip_states;
                                                                     root_prior=[0.6, 0.4],
                                                                     steps_per_unit=64,
                                                                     min_steps=8) ≈
              loglikelihood_uncoloured_mtbd2(tree, native_from_superspreader, tip_states;
                                             root_prior=[0.6, 0.4],
                                             steps_per_unit=64,
                                             min_steps=8)
        @test total_loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_super_free, batch_trees, superspreader_spec, batch_observations;
                                                                           root_prior=[0.6, 0.4],
                                                                           steps_per_unit=64,
                                                                           min_steps=8) ≈
              total_loglikelihood_uncoloured_mtbd2(batch_trees, native_from_superspreader, batch_observations;
                                                   root_prior=[0.6, 0.4],
                                                   steps_per_unit=64,
                                                   min_steps=8)
        @test loglikelihood_uncoloured_mtbd2_superspreader_from_unconstrained(η_super_free, tree, superspreader_spec, tip_states;
                                                                              root_prior=[0.6, 0.4],
                                                                              steps_per_unit=64,
                                                                              min_steps=8) ≈
              loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_super_free, tree, superspreader_spec, tip_states;
                                                                     root_prior=[0.6, 0.4],
                                                                     steps_per_unit=64,
                                                                     min_steps=8)
        @test total_loglikelihood_uncoloured_mtbd2_superspreader_from_unconstrained(η_super_free, batch_trees, superspreader_spec, batch_observations;
                                                                                    root_prior=[0.6, 0.4],
                                                                                    steps_per_unit=64,
                                                                                    min_steps=8) ≈
              total_loglikelihood_uncoloured_mtbd2_superspreader_from_free(θ_super_free, batch_trees, superspreader_spec, batch_observations;
                                                                           root_prior=[0.6, 0.4],
                                                                           steps_per_unit=64,
                                                                           min_steps=8)

        superspreader_mle_spec = UncolouredMTBD2SuperspreaderSpec(
            superspreader_pars;
            total_R0=true,
            superspreader_fraction=false,
            relative_transmissibility=true,
            death=falses(2),
            sampling=falses(2),
            removal_probability=falses(2),
            ρ₀=falses(2),
        )
        θ_super_mle0 = free_parameter_vector(superspreader_pars, superspreader_mle_spec)
        η_super_mle0 = uncoloured_mtbd2_superspreader_unconstrained_from_free(θ_super_mle0, superspreader_mle_spec)
        initial_super_nll = -total_loglikelihood_uncoloured_mtbd2_superspreader_from_unconstrained(
            η_super_mle0, batch_trees, superspreader_mle_spec, batch_observations;
            root_prior=[0.6, 0.4], steps_per_unit=64, min_steps=8,
        )
        superspreader_fit = fit_uncoloured_mtbd2_superspreader_mle(
            batch_trees, superspreader_pars, superspreader_mle_spec, batch_observations;
            root_prior=[0.6, 0.4], steps_per_unit=64, min_steps=8,
            initial_step=0.02, tolerance=1e-4, maxiter=20,
        )
        @test isfinite(superspreader_fit.maximum_loglikelihood)
        @test superspreader_fit.minimum_negloglikelihood <= initial_super_nll + 1e-8
        @test superspreader_fit.initial_negloglikelihood ≈ initial_super_nll
        @test superspreader_fit.θ_free_hat ≈ free_parameter_vector(superspreader_fit.superspreader_params_hat, superspreader_mle_spec)
        @test uncoloured_mtbd2_parameter_vector(superspreader_fit.params_hat) ≈
              uncoloured_mtbd2_parameter_vector(uncoloured_mtbd2_native_parameters(superspreader_fit.superspreader_params_hat))
        superspreader_fit_from_η = fit_uncoloured_mtbd2_superspreader_mle(
            tree, η_super_mle0, superspreader_mle_spec, tip_states;
            init_is_unconstrained=true, root_prior=[0.6, 0.4],
            steps_per_unit=64, min_steps=8, initial_step=0.02,
            tolerance=1e-4, maxiter=5,
        )
        @test isfinite(superspreader_fit_from_η.minimum_negloglikelihood)
        no_free_superspreader_spec = UncolouredMTBD2SuperspreaderSpec(
            superspreader_pars;
            total_R0=false,
            superspreader_fraction=false,
            relative_transmissibility=false,
            death=falses(2),
            sampling=falses(2),
            removal_probability=falses(2),
            ρ₀=falses(2),
        )
        @test_throws ArgumentError fit_uncoloured_mtbd2_superspreader_mle(tree, superspreader_pars, no_free_superspreader_spec, tip_states)

        # Packet-12 diagnostic caveat: in the zero-transition superspreader map,
        # unknown sampled-node states can make superspreader coordinates weakly
        # identified. These checks are qualitative slice diagnostics, not formal
        # identifiability claims.
        function superspreader_with(base; total_R0=base.total_R0,
                                    superspreader_fraction=base.superspreader_fraction,
                                    relative_transmissibility=base.relative_transmissibility)
            return UncolouredMTBD2SuperspreaderParameters(
                total_R0,
                superspreader_fraction,
                relative_transmissibility,
                base.death,
                base.sampling,
                base.removal_probability,
                base.ρ₀,
            )
        end
        function superspreader_slice(grid, coordinate::Symbol, observations)
            return [
                total_loglikelihood_uncoloured_mtbd2_superspreader(
                    batch_trees,
                    coordinate === :total_R0 ? superspreader_with(superspreader_pars; total_R0=x) :
                    coordinate === :superspreader_fraction ? superspreader_with(superspreader_pars; superspreader_fraction=x) :
                    coordinate === :relative_transmissibility ? superspreader_with(superspreader_pars; relative_transmissibility=x) :
                    error("unexpected coordinate"),
                    observations;
                    root_prior=[0.6, 0.4],
                    steps_per_unit=64,
                    min_steps=8,
                )
                for x in grid
            ]
        end
        diagnostic_known = [
            Dict(3 => 1, 4 => 2),
            Dict(3 => 2, 4 => 1),
            Dict(3 => 1, 4 => 2, 5 => 1),
        ]
        diagnostic_unknown = [
            Dict(3 => missing, 4 => missing),
            Dict(3 => missing, 4 => missing),
            Dict(3 => missing, 4 => missing, 5 => missing),
        ]
        diagnostic_mixed = [
            Dict(3 => 1, 4 => missing),
            Dict(3 => missing, 4 => 1),
            Dict(3 => 1, 4 => missing, 5 => 1),
        ]
        R0_grid = collect(0.8:0.25:2.8)
        q_grid = collect(0.05:0.05:0.55)
        rel_grid = collect(1.0:1.0:10.0)
        for (grid, coordinate) in ((R0_grid, :total_R0),
                                   (q_grid, :superspreader_fraction),
                                   (rel_grid, :relative_transmissibility))
            known_values = superspreader_slice(grid, coordinate, diagnostic_known)
            unknown_values = superspreader_slice(grid, coordinate, diagnostic_unknown)
            mixed_values = superspreader_slice(grid, coordinate, diagnostic_mixed)
            @test all(isfinite, known_values)
            @test all(isfinite, unknown_values)
            @test all(isfinite, mixed_values)
            @test maximum(known_values) - minimum(known_values) >=
                  maximum(unknown_values) - minimum(unknown_values) - 1e-8
        end
        unknown_R0_values = superspreader_slice(R0_grid, :total_R0, diagnostic_unknown)
        @test 1 < argmax(unknown_R0_values) < length(unknown_R0_values)
        @test maximum(unknown_R0_values) - minimum(unknown_R0_values) > 0.25

        swap_mtbd2_pars(pars) = UncolouredMTBD2ConstantParameters(
            pars.birth[[2, 1], [2, 1]],
            pars.death[[2, 1]],
            pars.sampling[[2, 1]],
            pars.removal_probability[[2, 1]],
            pars.transition[[2, 1], [2, 1]],
            pars.ρ₀[[2, 1]],
        )
        swap_mtbd2_obs(obs::Integer) = 3 - obs
        swap_mtbd2_obs(::Missing) = missing
        swap_mtbd2_obs(::Nothing) = nothing
        swap_mtbd2_obs(::Colon) = (:)
        swap_mtbd2_obs(obs::Tuple) = tuple((swap_mtbd2_obs(x) for x in obs)...)
        swap_mtbd2_obs(obs::AbstractVector{Bool}) = reverse(obs)
        swap_mtbd2_obs(obs::AbstractVector{<:Integer}) = [swap_mtbd2_obs(x) for x in obs]
        swap_mtbd2_observations(obs::AbstractDict) = Dict(k => swap_mtbd2_obs(v) for (k, v) in obs)

        swapped_pars = swap_mtbd2_pars(mtbd2_pars)
        symmetry_obs = Dict(3 => 1, 4 => (1, 2))
        symmetry_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, symmetry_obs;
                                                     root_prior=[0.65, 0.35],
                                                     steps_per_unit=64,
                                                     min_steps=8)
        swapped_symmetry_ll = loglikelihood_uncoloured_mtbd2(tree, swapped_pars, swap_mtbd2_observations(symmetry_obs);
                                                             root_prior=[0.35, 0.65],
                                                             steps_per_unit=64,
                                                             min_steps=8)
        @test swapped_symmetry_ll ≈ symmetry_ll

        sampled_unary_symmetry_ll = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars,
                                                                   Dict(3 => 1, 4 => [true, true], 5 => 2);
                                                                   root_prior=[0.65, 0.35],
                                                                   steps_per_unit=64,
                                                                   min_steps=8)
        swapped_sampled_unary_symmetry_ll = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, swapped_pars,
                                                                           swap_mtbd2_observations(Dict(3 => 1, 4 => [true, true], 5 => 2));
                                                                           root_prior=[0.35, 0.65],
                                                                           steps_per_unit=64,
                                                                           min_steps=8)
        @test swapped_sampled_unary_symmetry_ll ≈ sampled_unary_symmetry_ll

        one_type_pars = UncolouredMTBD2ConstantParameters(
            [1.05 0.0; 0.0 1.05],
            [0.25, 0.25],
            [0.5, 0.5],
            [0.65, 0.65],
            zeros(2, 2),
            [0.2, 0.2],
        )
        one_type_binary_unknown = loglikelihood_uncoloured_mtbd2(tree, one_type_pars, Dict(3 => missing, 4 => missing);
                                                                 root_prior=[0.5, 0.5],
                                                                 steps_per_unit=64,
                                                                 min_steps=8)
        one_type_binary_known1 = loglikelihood_uncoloured_mtbd2(tree, one_type_pars, Dict(3 => 1, 4 => 1);
                                                                root_prior=[1.0, 0.0],
                                                                steps_per_unit=64,
                                                                min_steps=8)
        one_type_binary_known2 = loglikelihood_uncoloured_mtbd2(tree, one_type_pars, Dict(3 => 2, 4 => 2);
                                                                root_prior=[0.0, 1.0],
                                                                steps_per_unit=64,
                                                                min_steps=8)
        @test one_type_binary_unknown ≈ one_type_binary_known1
        @test one_type_binary_known1 ≈ one_type_binary_known2

        one_type_sampled_unary_unknown = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, one_type_pars,
                                                                        Dict(3 => missing, 4 => missing, 5 => missing);
                                                                        root_prior=[0.5, 0.5],
                                                                        steps_per_unit=64,
                                                                        min_steps=8)
        one_type_sampled_unary_known1 = loglikelihood_uncoloured_mtbd2(sampled_unary_tree, one_type_pars,
                                                                       Dict(3 => 1, 4 => 1, 5 => 1);
                                                                       root_prior=[1.0, 0.0],
                                                                       steps_per_unit=64,
                                                                       min_steps=8)
        @test one_type_sampled_unary_unknown ≈ one_type_sampled_unary_known1

        unknown_missing_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => missing, 4 => missing);
                                                           root_prior=[0.6, 0.4],
                                                           steps_per_unit=64,
                                                           min_steps=8)
        unknown_nothing_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => nothing, 4 => nothing);
                                                           root_prior=[0.6, 0.4],
                                                           steps_per_unit=64,
                                                           min_steps=8)
        unknown_colon_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => :, 4 => :);
                                                         root_prior=[0.6, 0.4],
                                                         steps_per_unit=64,
                                                         min_steps=8)
        unknown_mask_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => [true, true], 4 => [true, true]);
                                                        root_prior=[0.6, 0.4],
                                                        steps_per_unit=64,
                                                        min_steps=8)
        singleton_mask_ll = loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => [true, false], 4 => [false, true]);
                                                          root_prior=[0.6, 0.4],
                                                          steps_per_unit=64,
                                                          min_steps=8)
        @test unknown_missing_ll ≈ unknown_nothing_ll
        @test unknown_missing_ll ≈ unknown_colon_ll
        @test unknown_missing_ll ≈ unknown_mask_ll
        @test singleton_mask_ll ≈ ll

        function scaled_tree(base_tree, scale)
            return Tree(
                base_tree.time .* scale,
                copy(base_tree.left),
                copy(base_tree.right),
                copy(base_tree.parent),
                copy(base_tree.kind),
                copy(base_tree.host),
                copy(base_tree.label),
            )
        end
        stability_parameters = (
            mtbd2_pars,
            UncolouredMTBD2ConstantParameters(
                [0.15 0.03; 0.02 0.12],
                [0.01, 0.015],
                [0.02, 0.025],
                [0.2, 0.8],
                [0.0 0.01; 0.015 0.0],
                [0.05, 0.08],
            ),
            UncolouredMTBD2ConstantParameters(
                [2.5 0.4; 0.35 2.0],
                [0.7, 0.5],
                [1.1, 0.9],
                [0.95, 0.15],
                [0.0 0.3; 0.25 0.0],
                [0.4, 0.3],
            ),
        )
        for pars_i in stability_parameters, scale in (0.25, 1.0, 3.0)
            binary_scaled = scaled_tree(tree, scale)
            unary_scaled = scaled_tree(sampled_unary_tree, scale)
            coarse_binary = loglikelihood_uncoloured_mtbd2(binary_scaled, pars_i, Dict(3 => missing, 4 => 2);
                                                           root_prior=[0.55, 0.45],
                                                           steps_per_unit=48,
                                                           min_steps=8)
            fine_binary = loglikelihood_uncoloured_mtbd2(binary_scaled, pars_i, Dict(3 => missing, 4 => 2);
                                                         root_prior=[0.55, 0.45],
                                                         steps_per_unit=96,
                                                         min_steps=16)
            coarse_unary = loglikelihood_uncoloured_mtbd2(unary_scaled, pars_i, Dict(3 => 1, 4 => missing, 5 => 2);
                                                          root_prior=[0.55, 0.45],
                                                          steps_per_unit=48,
                                                          min_steps=8)
            fine_unary = loglikelihood_uncoloured_mtbd2(unary_scaled, pars_i, Dict(3 => 1, 4 => missing, 5 => 2);
                                                        root_prior=[0.55, 0.45],
                                                        steps_per_unit=96,
                                                        min_steps=16)
            @test isfinite(coarse_binary)
            @test isfinite(fine_binary)
            @test isfinite(coarse_unary)
            @test isfinite(fine_unary)
            @test coarse_binary ≈ fine_binary atol=0.05
            @test coarse_unary ≈ fine_unary atol=0.05
        end

        sim_true = UncolouredMTBD2ConstantParameters(
            [1.25 0.25; 0.12 0.85],
            [0.18, 0.28],
            [0.75, 0.55],
            [0.55, 0.35],
            [0.0 0.08; 0.05 0.0],
            [0.25, 0.20],
        )
        sim_perturbed = UncolouredMTBD2ConstantParameters(
            [0.65 0.08; 0.30 1.45],
            [0.35, 0.12],
            [0.35, 0.95],
            [0.20, 0.80],
            [0.0 0.22; 0.01 0.0],
            [0.08, 0.35],
        )
        multitype_for_validation(pars) = MultitypeBDParameters(
            pars.birth,
            pars.death,
            pars.sampling,
            pars.removal_probability,
            pars.transition,
            pars.ρ₀,
        )
        function plain_validation_eventlog(log::MultitypeBDEventLog)
            times = Float64[]
            lineages = Int[]
            parents = Int[]
            kinds = BDEventKind[]
            for i in 1:length(log)
                if log.kind[i] == MultitypeBirth
                    push!(times, log.time[i]); push!(lineages, log.lineage[i]); push!(parents, log.parent[i]); push!(kinds, Birth)
                elseif log.kind[i] == MultitypeDeath
                    push!(times, log.time[i]); push!(lineages, log.lineage[i]); push!(parents, log.parent[i]); push!(kinds, Death)
                elseif log.kind[i] == MultitypeFossilizedSampling
                    push!(times, log.time[i]); push!(lineages, log.lineage[i]); push!(parents, log.parent[i]); push!(kinds, FossilizedSampling)
                elseif log.kind[i] == MultitypeSerialSampling
                    push!(times, log.time[i]); push!(lineages, log.lineage[i]); push!(parents, log.parent[i]); push!(kinds, SerialSampling)
                end
            end
            return BDEventLog(times, lineages, parents, kinds, length(log.initial_types), log.tmax)
        end
        function sampled_validation_observations(tree::TreeSim.Tree, log::MultitypeBDEventLog)
            observations = Dict{Int,Int}()
            for node in eachindex(tree)
                tree.kind[node] in (TreeSim.SampledLeaf, TreeSim.SampledUnary) || continue
                event = findfirst(i -> log.lineage[i] == tree.host[node] &&
                                       isapprox(log.time[i], tree.time[node]; atol=1e-10, rtol=0.0) &&
                                       log.kind[i] in (MultitypeFossilizedSampling, MultitypeSerialSampling),
                                  1:length(log))
                event === nothing && error("missing sampled state for validation node $node")
                observations[node] = log.type_before[event]
            end
            return observations
        end
        validation_rng = MersenneTwister(20260418)
        validation_rows = []
        validation_attempts = 0
        while validation_attempts < 40 && length(validation_rows) < 8
            validation_attempts += 1
            log = simulate_multitype_bd(validation_rng, multitype_for_validation(sim_true), 2.0;
                                        initial_types=[1], apply_ρ₀=true)
            forest = forest_from_eventlog(plain_validation_eventlog(log); tj=0.0, tk=log.tmax)
            length(forest) == 1 || continue
            simulated_tree = only(forest)
            all(kind -> kind in (TreeSim.Root, TreeSim.Binary, TreeSim.SampledLeaf, TreeSim.SampledUnary), simulated_tree.kind) || continue
            known_obs = sampled_validation_observations(simulated_tree, log)
            unknown_obs = Dict(node => missing for node in keys(known_obs))
            mixed_obs = Dict(node => (isodd(node) ? state : missing) for (node, state) in known_obs)
            true_known = loglikelihood_uncoloured_mtbd2(simulated_tree, sim_true, known_obs)
            perturbed_known = loglikelihood_uncoloured_mtbd2(simulated_tree, sim_perturbed, known_obs)
            true_unknown = loglikelihood_uncoloured_mtbd2(simulated_tree, sim_true, unknown_obs)
            true_mixed = loglikelihood_uncoloured_mtbd2(simulated_tree, sim_true, mixed_obs)
            push!(validation_rows, (
                tree=simulated_tree,
                true_known=true_known,
                perturbed_known=perturbed_known,
                true_unknown=true_unknown,
                true_mixed=true_mixed,
            ))
        end
        @test length(validation_rows) >= 6
        @test any(row -> any(==(TreeSim.SampledUnary), row.tree.kind), validation_rows)
        @test all(row -> isfinite(row.true_known) &&
                         isfinite(row.perturbed_known) &&
                         isfinite(row.true_unknown) &&
                         isfinite(row.true_mixed),
                  validation_rows)
        @test sum(row.true_known for row in validation_rows) >
              sum(row.perturbed_known for row in validation_rows)

        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => 1))
        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => 1, 4 => 3))
        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => 1, 4 => Int[]))
        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(tree, mtbd2_pars, Dict(3 => 1, 4 => [true, false, true]))
        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(sampled_unary_tree, mtbd2_pars, Dict(4 => 2, 5 => 1))
        @test_throws ArgumentError loglikelihood_uncoloured_mtbd2(unsampled_unary_tree(), mtbd2_pars, Dict(3 => 1))
        @test_throws ArgumentError UncolouredMTBD2ConstantParameters([0.1 0.2], [0.1, 0.1], [0.1, 0.1], [1.0, 1.0], zeros(2, 2))
    end

    @testset "multitype event-log to colored-tree bridge" begin
        bridge_log = MultitypeBDEventLog(
            [0.2, 0.4, 0.5, 0.7, 1.0],
            [2, 1, 2, 1, 2],
            [1, 0, 0, 0, 0],
            [MultitypeBirth, MultitypeTransition, MultitypeFossilizedSampling,
             MultitypeSerialSampling, MultitypeSerialSampling],
            [1, 1, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [1],
            1.0,
        )
        bridge_tree = multitype_colored_tree_from_eventlog(bridge_log)
        @test bridge_tree.origin_time == 1.0
        @test bridge_tree.root_type == 1
        @test [segment.type for segment in bridge_tree.segments] == [1, 1, 2, 2, 2]
        @test [segment.start_time for segment in bridge_tree.segments] ≈ [0.8, 0.6, 0.5, 0.3, 0.0]
        @test [segment.end_time for segment in bridge_tree.segments] ≈ [1.0, 0.8, 0.8, 0.6, 0.5]
        @test bridge_tree.births == [MultitypeColoredBirth(0.8, 1, 2)]
        @test bridge_tree.transitions == [MultitypeColoredTransition(0.6, 1, 2)]
        @test bridge_tree.ancestral_samples == [MultitypeColoredSampling(0.5, 2)]
        @test [sample.time for sample in bridge_tree.terminal_samples] ≈ [0.3]
        @test [sample.type for sample in bridge_tree.terminal_samples] == [2]
        @test bridge_tree.present_samples == [2]

        bridge_pars = MultitypeBDParameters(
            [0.4 0.8; 0.2 0.5],
            [0.1, 0.2],
            [0.3, 0.6],
            [0.2, 0.7],
            [0.0 0.9; 0.1 0.0],
            [0.4, 0.5],
        )
        @test validate_multitype_colored_tree(bridge_tree, bridge_pars)
        checked_bridge_tree = validate_multitype_colored_tree_from_eventlog(bridge_log, bridge_pars)
        @test checked_bridge_tree.origin_time == bridge_tree.origin_time
        @test checked_bridge_tree.root_type == bridge_tree.root_type
        @test length(checked_bridge_tree.segments) == length(bridge_tree.segments)
        @test isfinite(multitype_colored_loglikelihood(bridge_tree, bridge_pars))
        @test multitype_colored_likelihood(bridge_tree, bridge_pars) > 0

        terminal_at_tmax = multitype_colored_tree_from_eventlog(bridge_log; serial_at_tmax=:terminal)
        @test terminal_at_tmax.present_samples == Int[]
        @test terminal_at_tmax.terminal_samples[end] == MultitypeColoredSampling(0.0, 2)

        k1_bridge_log = MultitypeBDEventLog(
            [1.0],
            [1],
            [0],
            [MultitypeSerialSampling],
            [1],
            [1],
            [1],
            1.0,
        )
        k1_bridge_tree = multitype_colored_tree_from_eventlog(k1_bridge_log)
        @test k1_bridge_tree.segments == [MultitypeColoredSegment(1, 0.0, 1.0)]
        @test k1_bridge_tree.present_samples == [1]
        k1_bridge_pars = MultitypeBDParameters([0.2;;], [0.1], [0.0], [0.0], [0.0;;], [0.8])
        @test multitype_colored_loglikelihood(k1_bridge_tree, k1_bridge_pars) ≈
              log(0.8) + only(multitype_log_flow(1.0, k1_bridge_pars))

        death_log = MultitypeBDEventLog([0.5], [1], [0], [MultitypeDeath], [1], [1], [1], 1.0)
        @test_throws ArgumentError multitype_colored_tree_from_eventlog(death_log)
        survivor_log = MultitypeBDEventLog(Float64[], Int[], Int[], MultitypeBDEventKind[], Int[], Int[], [1], 1.0)
        @test_throws ArgumentError multitype_colored_tree_from_eventlog(survivor_log)
        multi_root_log = MultitypeBDEventLog([1.0, 1.0], [1, 2], [0, 0],
                                             [MultitypeSerialSampling, MultitypeSerialSampling],
                                             [1, 1], [1, 1], [1, 1], 1.0)
        @test_throws ArgumentError multitype_colored_tree_from_eventlog(multi_root_log)
        @test_throws ArgumentError multitype_colored_tree_from_eventlog(bridge_log; serial_at_tmax=:unknown)
    end

    @testset "multitype pruned event-log extraction" begin
        pruned_log = MultitypeBDEventLog(
            [0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0],
            [2, 3, 3, 1, 1, 1, 2],
            [1, 1, 0, 0, 0, 0, 0],
            [MultitypeBirth, MultitypeBirth, MultitypeDeath, MultitypeTransition,
             MultitypeFossilizedSampling, MultitypeSerialSampling, MultitypeSerialSampling],
            [1, 1, 2, 1, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [1],
            1.0,
        )
        pruned_tree = pruned_multitype_colored_tree_from_eventlog(pruned_log)
        @test pruned_tree.origin_time == 1.0
        @test pruned_tree.root_type == 1
        @test [segment.type for segment in pruned_tree.segments] == [1, 1, 2, 2, 2]
        @test [segment.start_time for segment in pruned_tree.segments] ≈ [0.8, 0.5, 0.35, 0.2, 0.0]
        @test [segment.end_time for segment in pruned_tree.segments] ≈ [1.0, 0.8, 0.5, 0.35, 0.8]
        @test pruned_tree.births == [MultitypeColoredBirth(0.8, 1, 2)]
        @test pruned_tree.hidden_births == MultitypeColoredHiddenBirth[]
        @test pruned_tree.transitions == [MultitypeColoredTransition(0.5, 1, 2)]
        @test pruned_tree.ancestral_samples == [MultitypeColoredSampling(0.35, 2)]
        @test [sample.time for sample in pruned_tree.terminal_samples] ≈ [0.2]
        @test pruned_tree.present_samples == [2]

        pruned_pars = MultitypeBDParameters(
            [0.4 0.8; 0.2 0.5],
            [0.1, 0.2],
            [0.3, 0.6],
            [0.2, 0.7],
            [0.0 0.9; 0.1 0.0],
            [0.4, 0.5],
        )
        @test validate_multitype_colored_tree(pruned_tree, pruned_pars)
        @test validate_pruned_multitype_colored_tree_from_eventlog(pruned_log, pruned_pars).root_type == 1
        @test isfinite(multitype_colored_loglikelihood(pruned_tree, pruned_pars))

        terminal_fossil_log = MultitypeBDEventLog(
            [0.4, 0.9],
            [1, 1],
            [0, 0],
            [MultitypeFossilizedSampling, MultitypeDeath],
            [1, 1],
            [1, 1],
            [1],
            1.0,
        )
        terminal_fossil_tree = pruned_multitype_colored_tree_from_eventlog(terminal_fossil_log)
        @test terminal_fossil_tree.ancestral_samples == MultitypeColoredSampling[]
        @test terminal_fossil_tree.terminal_samples == [MultitypeColoredSampling(0.6, 1)]
        @test terminal_fossil_tree.segments == [MultitypeColoredSegment(1, 0.6, 1.0)]

        k1_pruned_log = MultitypeBDEventLog(
            [0.2, 0.4, 0.6],
            [2, 2, 1],
            [1, 0, 0],
            [MultitypeBirth, MultitypeDeath, MultitypeSerialSampling],
            [1, 1, 1],
            [1, 1, 1],
            [1],
            1.0,
        )
        k1_pruned_tree = pruned_multitype_colored_tree_from_eventlog(k1_pruned_log)
        @test k1_pruned_tree.births == MultitypeColoredBirth[]
        @test k1_pruned_tree.hidden_births == MultitypeColoredHiddenBirth[]
        @test k1_pruned_tree.terminal_samples == [MultitypeColoredSampling(0.4, 1)]
        @test k1_pruned_tree.segments == [MultitypeColoredSegment(1, 0.4, 1.0)]

        child_only_log = MultitypeBDEventLog(
            [0.2, 0.5, 0.8],
            [2, 1, 2],
            [1, 0, 0],
            [MultitypeBirth, MultitypeDeath, MultitypeSerialSampling],
            [1, 1, 2],
            [2, 1, 2],
            [1],
            1.0,
        )
        child_only_tree = pruned_multitype_colored_tree_from_eventlog(child_only_log)
        @test child_only_tree.births == MultitypeColoredBirth[]
        @test child_only_tree.hidden_births == [MultitypeColoredHiddenBirth(0.8, 1, 2)]
        @test [segment.type for segment in child_only_tree.segments] == [1, 2]
        @test [segment.start_time for segment in child_only_tree.segments] ≈ [0.8, 0.2]
        @test [segment.end_time for segment in child_only_tree.segments] ≈ [1.0, 0.8]
        @test [sample.time for sample in child_only_tree.terminal_samples] ≈ [0.2]
        @test [sample.type for sample in child_only_tree.terminal_samples] == [2]
        child_only_pars = MultitypeBDParameters([0.2 0.7; 0.1 0.3], [0.2, 0.2], [0.4, 0.5],
                                                [0.5, 0.5], [0.0 0.1; 0.1 0.0], [0.0, 0.0])
        @test isfinite(multitype_colored_loglikelihood(child_only_tree, child_only_pars))

        unsampled_log = MultitypeBDEventLog([0.5], [1], [0], [MultitypeDeath], [1], [1], [1], 1.0)
        @test_throws ArgumentError pruned_multitype_colored_tree_from_eventlog(unsampled_log)
        @test_throws ArgumentError pruned_multitype_colored_tree_from_eventlog(pruned_log; serial_at_tmax=:unknown)
    end

    @testset "multitype scoring and fitting wrappers" begin
        score_pars = MultitypeBDParameters(
            [0.4 0.8; 0.2 0.5],
            [0.1, 0.2],
            [0.3, 0.6],
            [0.2, 0.7],
            [0.0 0.9; 0.1 0.0],
            [0.4, 0.5],
        )
        score_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.8, 1.0),
                MultitypeColoredSegment(1, 0.5, 0.8),
                MultitypeColoredSegment(2, 0.0, 0.8),
            ],
            births=[MultitypeColoredBirth(0.8, 1, 2)],
            transitions=[MultitypeColoredTransition(0.5, 1, 2)],
            present_samples=[2],
        )
        raw_ll = multitype_colored_loglikelihood(score_tree, score_pars)
        @test multitype_loglikelihood(score_tree, score_pars) ≈ raw_ll
        @test multitype_loglikelihood([score_tree, score_tree], score_pars) ≈ 2raw_ll

        spec = MultitypeMLESpec(
            score_pars;
            fit_birth=[false true; false false],
            fit_death=[false, false],
            fit_sampling=[false, false],
            fit_transition=[false false; false false],
        )
        θ = multitype_pack_parameters(score_pars, spec)
        @test length(θ) == 1
        unpacked = multitype_unpack_parameters(θ, spec)
        @test unpacked.birth == score_pars.birth
        @test unpacked.death == score_pars.death
        @test unpacked.sampling == score_pars.sampling
        @test unpacked.transition == score_pars.transition
        @test multitype_negloglikelihood(θ, score_tree, spec) ≈ -raw_ll
        @test multitype_negloglikelihood(θ, [score_tree, score_tree], spec) ≈ -2raw_ll

        lower_birth = MultitypeBDParameters(
            [0.4 0.2; 0.2 0.5],
            score_pars.death,
            score_pars.sampling,
            score_pars.removal_probability,
            score_pars.transition,
            score_pars.ρ₀,
        )
        fit_spec = MultitypeMLESpec(
            lower_birth;
            fit_birth=[false true; false false],
            fit_death=[false, false],
            fit_sampling=[false, false],
            fit_transition=[false false; false false],
        )
        before = multitype_loglikelihood(score_tree, lower_birth)
        fit = fit_multitype_mle(
            score_tree;
            spec=fit_spec,
            lower=[log(0.05)],
            upper=[log(5.0)],
            initial_step=0.5,
            tolerance=1e-4,
            maxiter=200,
        )
        @test fit.loglikelihood >= before
        @test fit.parameters.birth[1, 2] > lower_birth.birth[1, 2]
        @test fit.result.converged || fit.result.iterations == 200

        k1_score_pars = MultitypeBDParameters([0.2;;], [0.1], [0.4], [0.5], [0.0;;], [0.8])
        k1_score_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.0, 1.0)],
            present_samples=[1],
        )
        @test multitype_loglikelihood(k1_score_tree, k1_score_pars) ≈
              multitype_colored_loglikelihood(k1_score_tree, k1_score_pars)

        @test_throws ArgumentError multitype_loglikelihood(MultitypeColoredTree[], score_pars)
        @test_throws ArgumentError multitype_unpack_parameters(Float64[], spec)
        @test_throws ArgumentError MultitypeMLESpec(score_pars; fit_birth=[true false])
        zero_support_pars = MultitypeBDParameters([0.0 0.8; 0.2 0.5], score_pars.death,
                                                  score_pars.sampling, score_pars.removal_probability,
                                                  score_pars.transition, score_pars.ρ₀)
        @test_throws ArgumentError MultitypeMLESpec(zero_support_pars; fit_birth=[true false; false false],
                                                         fit_death=[false, false],
                                                         fit_sampling=[false, false],
                                                         fit_transition=[false false; false false])
        @test_throws ArgumentError fit_multitype_mle(MultitypeColoredTree[]; spec=spec)
        no_fit_spec = MultitypeMLESpec(
            score_pars;
            fit_birth=falses(2, 2),
            fit_death=falses(2),
            fit_sampling=falses(2),
            fit_transition=falses(2, 2),
        )
        @test_throws ArgumentError fit_multitype_mle(score_tree; spec=no_fit_spec)

        rng_fit = MersenneTwister(20260418)
        sim_fit_pars = MultitypeBDParameters([0.0 0.9; 0.0 0.0], [0.15, 0.25], [0.05, 0.8],
                                             [1.0, 1.0], zeros(2, 2), [0.0, 0.0])
        extracted = MultitypeColoredTree[]
        attempts = 0
        while length(extracted) < 2 && attempts < 200
            attempts += 1
            log = simulate_multitype_bd(rng_fit, sim_fit_pars, 1.0; initial_types=[1], apply_ρ₀=false)
            try
                push!(extracted, pruned_multitype_colored_tree_from_eventlog(log))
            catch err
                err isa ArgumentError || rethrow()
            end
        end
        @test length(extracted) == 2
        @test multitype_loglikelihood(extracted, sim_fit_pars) ≈
              sum(tree -> multitype_loglikelihood(tree, sim_fit_pars), extracted)

        perturbed_fit_pars = MultitypeBDParameters([0.0 0.25; 0.0 0.0], sim_fit_pars.death,
                                                   sim_fit_pars.sampling, sim_fit_pars.removal_probability,
                                                   sim_fit_pars.transition, sim_fit_pars.ρ₀)
        extracted_spec = MultitypeMLESpec(perturbed_fit_pars;
                                          fit_birth=[false true; false false],
                                          fit_death=falses(2),
                                          fit_sampling=falses(2),
                                          fit_transition=falses(2, 2))
        extracted_before = multitype_loglikelihood(extracted, perturbed_fit_pars)
        extracted_fit = fit_multitype_mle(extracted; spec=extracted_spec, lower=[log(0.05)], upper=[log(5.0)],
                                          initial_step=0.5, tolerance=1e-4, maxiter=200)
        @test extracted_fit.loglikelihood >= extracted_before
    end

    @testset "multitype validation comparisons" begin
        validation_pars = MultitypeBDParameters(
            [0.15 0.08; 0.04 0.12],
            [0.35, 0.25],
            [0.20, 0.30],
            [0.70, 0.40],
            [0.0 0.10; 0.06 0.0],
            [0.15, 0.25],
        )
        t_validation = 0.8
        analytical_E = multitype_E(t_validation, validation_pars; steps_per_unit=512)
        rng = MersenneTwister(90417)
        nsims = 2_500
        for initial_type in 1:2
            no_observed = 0
            for _ in 1:nsims
                log = simulate_multitype_bd(rng, validation_pars, t_validation; initial_types=[initial_type])
                no_observed += sum(multitype_S_at(log, t_validation)) == 0
            end
            @test no_observed / nsims ≈ analytical_E[initial_type] atol=0.04
        end

        k1_validation = MultitypeBDParameters([0.45;;], [0.30], [0.25], [0.60], [0.0;;], [0.20])
        k1_single = ConstantRateBDParameters(0.45, 0.30, 0.25, 0.60, 0.20)
        @test only(multitype_E(1.1, k1_validation; steps_per_unit=1024)) ≈ E_constant(1.1, k1_single) atol=2e-6
        rng_k1 = MersenneTwister(90418)
        k1_no_observed = 0
        for _ in 1:nsims
            log = simulate_multitype_bd(rng_k1, k1_validation, 1.1)
            k1_no_observed += sum(multitype_S_at(log, 1.1)) == 0
        end
        @test k1_no_observed / nsims ≈ E_constant(1.1, k1_single) atol=0.04

        flow_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[MultitypeColoredSegment(1, 0.25, 1.0)],
            terminal_samples=[MultitypeColoredSampling(0.25, 1)],
        )
        logflow_025 = multitype_log_flow(0.25, validation_pars)
        logflow_100 = multitype_log_flow(1.0, validation_pars)
        E025 = multitype_E(0.25, validation_pars)[1]
        expected_flow_tree = (logflow_100[1] - logflow_025[1]) +
                             log(validation_pars.sampling[1] *
                                 (validation_pars.removal_probability[1] +
                                  (1 - validation_pars.removal_probability[1]) * E025))
        @test multitype_colored_loglikelihood(flow_tree, validation_pars) ≈ expected_flow_tree
        @test multitype_colored_likelihood(flow_tree, validation_pars) ≈ exp(expected_flow_tree)

        transition_tree = MultitypeColoredTree(
            1.0,
            1;
            segments=[
                MultitypeColoredSegment(1, 0.5, 1.0),
                MultitypeColoredSegment(2, 0.0, 0.5),
            ],
            transitions=[MultitypeColoredTransition(0.5, 1, 2)],
            present_samples=[2],
        )
        higher_transition = MultitypeBDParameters(
            validation_pars.birth,
            validation_pars.death,
            validation_pars.sampling,
            validation_pars.removal_probability,
            [0.0 0.25; 0.06 0.0],
            validation_pars.ρ₀,
        )
        @test multitype_colored_loglikelihood(transition_tree, higher_transition) >
              multitype_colored_loglikelihood(transition_tree, validation_pars)
    end
