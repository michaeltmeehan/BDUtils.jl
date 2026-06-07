    @testset "constant-rate native simulation and empirical extraction" begin
        handcrafted = BDEventLog(
            [0.2, 0.4, 0.6, 0.8],
            [2, 2, 1, 3],
            [1, 0, 0, 2],
            [Birth, FossilizedSampling, SerialSampling, Death],
            1,
            1.0,
        )

        @test length(handcrafted) == 4
        @test handcrafted[1] == BDEventRecord(0.2, 2, 1, Birth)
        @test collect(record.kind for record in handcrafted) == [Birth, FossilizedSampling, SerialSampling, Death]
        @test sprint(show, handcrafted) == "BDEventLog(4 events, initial_lineages=1, tmax=1.0)"

        @test NS_at(handcrafted, 0.0) == (N=1, S=0)
        @test NS_at(handcrafted, 0.2) == (N=2, S=0)
        @test NS_at(handcrafted, 0.5) == (N=2, S=1)
        @test NS_at(handcrafted, 0.7) == (N=1, S=2)
        @test NS_at(handcrafted, 1.0) == (N=0, S=2)
        @test N_at(handcrafted, 0.7) == 1
        @test S_at(handcrafted, 0.7) == 2
        @test N_over_time(handcrafted, [0.0, 0.5, 1.0]) == [1, 2, 0]
        @test S_over_time(handcrafted, [0.0, 0.5, 1.0]) == [0, 1, 2]
        @test NS_over_time(handcrafted, [0.0, 0.7]) == [(N=1, S=0), (N=1, S=2)]

        @test extant_lineages_at(handcrafted, 0.0) == [1]
        @test extant_lineages_at(handcrafted, 0.2) == [1, 2]
        @test extant_lineages_at(handcrafted, 0.7) == [2]
        @test retained_lineages_at(handcrafted, 0.0) == [1]
        @test retained_lineages_at(handcrafted, 0.2) == [1, 2]
        @test retained_lineages_at(handcrafted, 0.5) == [1]
        @test retained_lineages_at(handcrafted, 0.7) == Int[]
        @test A_at(handcrafted, 0.0) == 1
        @test A_over_time(handcrafted, [0.0, 0.2, 0.7, 1.0]) == [1, 2, 0, 0]

        terminal_sample = BDEventLog([0.25, 1.0], [2, 2], [1, 0], [Birth, SerialSampling], 1, 1.0)
        @test extant_lineages_at(terminal_sample, 0.5) == [1, 2]
        @test retained_lineages_at(terminal_sample, 0.5) == [2]
        @test retained_lineages_at(terminal_sample, 1.0) == Int[]
        @test A_at(terminal_sample, 0.5) == 1

        fossil_at_tj = BDEventLog([0.5], [1], [0], [FossilizedSampling], 1, 1.0)
        @test retained_lineages_at(fossil_at_tj, 0.5) == Int[]
        @test A_at(fossil_at_tj, 0.49, 0.5) == 1

        serial_at_tj = BDEventLog([0.5], [1], [0], [SerialSampling], 1, 1.0)
        @test extant_lineages_at(serial_at_tj, 0.5) == Int[]
        @test A_at(serial_at_tj, 0.5) == 0
        @test A_at(serial_at_tj, 0.49, 0.5) == 1

        sample_after_tj = BDEventLog([0.500001], [1], [0], [FossilizedSampling], 1, 1.0)
        @test A_at(sample_after_tj, 0.5) == 1
        @test A_at(sample_after_tj, 0.500001) == 0

        truncated = BDEventLog([0.8, 0.9], [1, 1], [0, 0], [FossilizedSampling, SerialSampling], 1, 1.0)
        @test A_at(truncated, 0.5, 0.7) == 0
        @test A_at(truncated, 0.5, 0.8) == 1
        @test A_at(truncated, 0.85, 0.87) == 0
        @test A_at(truncated, 0.85, 0.9) == 1
        @test A_at(truncated, 0.8, 0.8) == 0
        @test A_over_time(truncated, [0.5, 0.8, 0.85]; tk=0.9) == [1, 1, 1]
        @test A_over_time(truncated, [0.5, 0.8]; tk=0.8) == [1, 0]

        extinct_unsampled = BDEventLog([0.1], [1], [0], [Death], 1, 1.0)
        counts = joint_counts_NS([handcrafted, extinct_unsampled], 1.0)
        @test counts == Dict((0, 2) => 1, (0, 0) => 1)
        @test joint_pmf_NS(counts) == Dict((0, 2) => 0.5, (0, 0) => 0.5)
        @test marginal_counts_NS(counts) == (N=Dict(0 => 2), S=Dict(2 => 1, 0 => 1))
        @test marginal_pmf_NS(counts) == (N=Dict(0 => 1.0), S=Dict(2 => 0.5, 0 => 0.5))
        @test reconstructed_counts_A([handcrafted, extinct_unsampled], 0.0) == Dict(1 => 1, 0 => 1)
        @test reconstructed_pmf_A([handcrafted, extinct_unsampled], 0.0) == Dict(1 => 0.5, 0 => 0.5)
        @test reconstructed_joint_counts_AS([handcrafted, extinct_unsampled], 0.5) == Dict((1, 1) => 1, (0, 0) => 1)
        @test reconstructed_joint_pmf_AS([handcrafted, extinct_unsampled], 0.5) == Dict((1, 1) => 0.5, (0, 0) => 0.5)
        @test empirical_retention_probability([handcrafted, extinct_unsampled], 0.0) == 0.5
        @test joint_counts_NS([handcrafted, extinct_unsampled], [0.0, 1.0]) == [
            Dict((1, 0) => 2),
            Dict((0, 2) => 1, (0, 0) => 1),
        ]

        birth_only = simulate_bd(MersenneTwister(1), ConstantRateBDParameters(1.0, 0.0, 0.0, 0.0), 0.5)
        @test birth_only isa BDEventLog
        @test all(==(Birth), birth_only.kind)
        @test issorted(birth_only.time)
        @test N_at(birth_only, 0.5) == birth_only.initial_lineages + length(birth_only)
        @test S_at(birth_only, 0.5) == 0

        serial_only = simulate_bd(MersenneTwister(2), ConstantRateBDParameters(1.0, 0.0, 10.0, 1.0), 10.0; apply_ρ₀=false)
        @test SerialSampling in serial_only.kind
        @test all(kind -> kind == Birth || kind == SerialSampling, serial_only.kind)
        @test N_at(serial_only, 10.0) >= 0
        @test S_at(serial_only, 10.0) == count(==(SerialSampling), serial_only.kind)

        contemp = simulate_bd(MersenneTwister(3), ConstantRateBDParameters(1.0, 0.0, 0.0, 0.0, 1.0), 0.0)
        @test length(contemp) == 1
        @test contemp.time == [0.0]
        @test contemp.kind == [SerialSampling]
        @test NS_at(contemp, 0.0) == (N=0, S=1)

        @test_throws ArgumentError simulate_bd(ConstantRateBDParameters(1.0, 0.0, 0.0, 0.0), -0.1)
        @test_throws ArgumentError simulate_bd(ConstantRateBDParameters(1.0, 0.0, 0.0, 0.0), 1.0; initial_lineages=-1)
        @test_throws ArgumentError NS_at(handcrafted, -0.1)
        @test_throws ArgumentError joint_pmf_NS(Dict{Tuple{Int,Int},Int}())
        @test_throws ArgumentError A_at(handcrafted, -0.1)
        @test_throws ArgumentError A_at(handcrafted, 0.8, 0.7)
        @test_throws ArgumentError A_at(handcrafted, 0.5, 1.1)
        @test_throws ArgumentError retained_lineages_at(handcrafted, 0.8, 0.7)
        @test_throws ArgumentError A_over_time(handcrafted, [0.1]; tk=1.1)
        @test_throws ArgumentError reconstructed_pmf_A(Dict{Int,Int}())
        @test_throws ArgumentError reconstructed_joint_pmf_AS(Dict{Tuple{Int,Int},Int}())
        @test_throws ArgumentError empirical_retention_probability([extinct_unsampled], 0.5)
    end
