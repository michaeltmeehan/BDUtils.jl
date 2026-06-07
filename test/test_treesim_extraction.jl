    @testset "TreeSim reconstructed extraction from BDEventLog" begin
        simple = BDEventLog([0.2, 0.5, 0.7], [2, 1, 2], [1, 0, 0], [Birth, FossilizedSampling, SerialSampling], 1, 1.0)

        forest = forest_from_eventlog(simple; tj=0.0, tk=1.0)
        @test length(forest) == A_at(simple, 0.0, 1.0) == 1
        @test validate_tree(only(forest); require_single_root=true, require_reachable=true)
        @test only(forest).kind == [Root, Binary, SampledLeaf, SampledLeaf]
        @test only(forest).host == [1, 1, 1, 2]

        after_birth = forest_from_eventlog(simple; tj=0.2, tk=1.0)
        @test length(after_birth) == A_at(simple, 0.2, 1.0) == 2
        @test sort([tree.host[TreeSim.root(tree)] for tree in after_birth]) == retained_lineages_at(simple, 0.2, 1.0)
        @test all(tree -> validate_tree(tree; require_single_root=true, require_reachable=true), after_birth)
        @test_throws ErrorException tree_from_eventlog(simple; tj=0.2, tk=1.0)

        fossil_chain = BDEventLog([0.3, 0.6], [1, 1], [0, 0], [FossilizedSampling, SerialSampling], 1, 1.0)
        chain_tree = tree_from_eventlog(fossil_chain; tj=0.0, tk=1.0)
        @test validate_tree(chain_tree; require_single_root=true, require_reachable=true)
        @test chain_tree.kind == [Root, SampledUnary, SampledLeaf]
        @test tree_from_eventlog(fossil_chain; tj=0.3, tk=1.0).kind == [Root, SampledLeaf]

        sample_at_tj = BDEventLog([0.5], [1], [0], [FossilizedSampling], 1, 1.0)
        @test forest_from_eventlog(sample_at_tj; tj=0.5, tk=1.0) == Tree[]
        @test length(forest_from_eventlog(sample_at_tj; tj=0.49, tk=0.5)) == A_at(sample_at_tj, 0.49, 0.5) == 1

        sample_at_tk = BDEventLog([0.5], [1], [0], [SerialSampling], 1, 1.0)
        @test length(forest_from_eventlog(sample_at_tk; tj=0.49, tk=0.5)) == A_at(sample_at_tk, 0.49, 0.5) == 1
        @test forest_from_eventlog(sample_at_tk; tj=0.5, tk=1.0) == Tree[]

        extinct_unsampled = BDEventLog([0.1], [1], [0], [Death], 1, 1.0)
        @test forest_from_eventlog(extinct_unsampled; tj=0.0, tk=1.0) == Tree[]
        @test isempty(tree_from_eventlog(extinct_unsampled; tj=0.0, tk=1.0))

        rng = MersenneTwister(20260417)
        sim_pars = ConstantRateBDParameters(1.4, 0.35, 0.8, 0.55, 0.25)
        for _ in 1:200
            log = simulate_bd(rng, sim_pars, 1.5; initial_lineages=2)
            for (tj, tk) in ((0.0, 1.5), (0.4, 1.2), (0.9, 1.5))
                forest = forest_from_eventlog(log; tj, tk)
                @test length(forest) == A_at(log, tj, tk)
                @test sort([tree.host[TreeSim.root(tree)] for tree in forest]) == retained_lineages_at(log, tj, tk)
                @test all(tree -> validate_tree(tree; require_single_root=true, require_reachable=true), forest)
                if length(forest) > 1
                    @test_throws ErrorException tree_from_eventlog(log; tj, tk)
                end
            end
        end

        @test_throws ArgumentError forest_from_eventlog(simple; tj=-0.1, tk=1.0)
        @test_throws ArgumentError forest_from_eventlog(simple; tj=0.8, tk=0.7)
        @test_throws ArgumentError forest_from_eventlog(simple; tj=0.0, tk=1.1)
    end

    @testset "TreeSim full sampled-ancestry extraction from BDEventLog" begin
        same_tree(a, b) = a.time == b.time &&
                          a.left == b.left &&
                          a.right == b.right &&
                          a.parent == b.parent &&
                          a.kind == b.kind &&
                          a.host == b.host &&
                          a.label == b.label
        same_forest(a, b) = length(a) == length(b) && all(same_tree(x, y) for (x, y) in zip(a, b))

        unary_birth = BDEventLog([0.2, 0.7], [2, 2], [1, 0], [Birth, SerialSampling], 1, 1.0)
        full = full_forest_from_eventlog(unary_birth; tj=0.0, tk=1.0)
        reconstructed = forest_from_eventlog(unary_birth; tj=0.0, tk=1.0)
        @test length(full) == length(reconstructed) == A_at(unary_birth, 0.0, 1.0) == 1
        @test validate_tree(only(full); require_single_root=true, require_reachable=true)
        @test only(full).kind == [Root, UnsampledUnary, SampledLeaf]
        @test only(full).host == [1, 1, 2]
        @test only(reconstructed).kind == [Root, SampledLeaf]
        @test same_forest([BDUtils._bd_collapse_unsampled_unary(tree) for tree in full], reconstructed)

        full_single = full_tree_from_eventlog(unary_birth; tj=0.0, tk=1.0)
        @test same_tree(full_single, only(full))

        sampled_parent_and_child = BDEventLog([0.2, 0.5, 0.7], [2, 1, 2], [1, 0, 0], [Birth, FossilizedSampling, SerialSampling], 1, 1.0)
        full_binary = full_forest_from_eventlog(sampled_parent_and_child; tj=0.0, tk=1.0)
        @test same_forest(full_binary, forest_from_eventlog(sampled_parent_and_child; tj=0.0, tk=1.0))
        @test only(full_binary).kind == [Root, Binary, SampledLeaf, SampledLeaf]

        after_birth = full_forest_from_eventlog(sampled_parent_and_child; tj=0.2, tk=1.0)
        @test length(after_birth) == A_at(sampled_parent_and_child, 0.2, 1.0) == 2
        @test all(tree -> validate_tree(tree; require_single_root=true, require_reachable=true), after_birth)
        @test_throws ErrorException full_tree_from_eventlog(sampled_parent_and_child; tj=0.2, tk=1.0)

        sample_at_tj = BDEventLog([0.5], [1], [0], [FossilizedSampling], 1, 1.0)
        @test full_forest_from_eventlog(sample_at_tj; tj=0.5, tk=1.0) == Tree[]
        @test length(full_forest_from_eventlog(sample_at_tj; tj=0.49, tk=0.5)) == 1

        sample_at_tk = BDEventLog([0.5], [1], [0], [SerialSampling], 1, 1.0)
        @test length(full_forest_from_eventlog(sample_at_tk; tj=0.49, tk=0.5)) == 1
        @test full_forest_from_eventlog(sample_at_tk; tj=0.5, tk=0.5) == Tree[]

        rng = MersenneTwister(20260418)
        sim_pars = ConstantRateBDParameters(1.6, 0.4, 0.9, 0.5, 0.2)
        for _ in 1:200
            log = simulate_bd(rng, sim_pars, 1.5; initial_lineages=2)
            for (tj, tk) in ((0.0, 1.5), (0.3, 1.1), (0.8, 1.5), (1.0, 1.0))
                full = full_forest_from_eventlog(log; tj, tk)
                reconstructed = forest_from_eventlog(log; tj, tk)
                @test length(full) == length(reconstructed) == A_at(log, tj, tk)
                @test sort([tree.host[TreeSim.root(tree)] for tree in full]) == retained_lineages_at(log, tj, tk)
                @test all(tree -> validate_tree(tree; require_single_root=true, require_reachable=true), full)
                @test same_forest([BDUtils._bd_collapse_unsampled_unary(tree) for tree in full], reconstructed)
                if length(full) > 1
                    @test_throws ErrorException full_tree_from_eventlog(log; tj, tk)
                end
            end
        end

        @test_throws ArgumentError full_forest_from_eventlog(unary_birth; tj=-0.1, tk=1.0)
        @test_throws ArgumentError full_forest_from_eventlog(unary_birth; tj=0.8, tk=0.7)
        @test_throws ArgumentError full_forest_from_eventlog(unary_birth; tj=0.0, tk=1.1)
    end
