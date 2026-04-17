#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BDUtils
using Printf
using Random
using Statistics
using TreeSim

function simpson(f, a::Float64, b::Float64; n::Int=1024)
    a == b && return 0.0
    n = iseven(n) ? n : n + 1
    h = (b - a) / n
    acc = f(a) + f(b)
    for i in 1:(n - 1)
        acc += (isodd(i) ? 4.0 : 2.0) * f(a + i * h)
    end
    return acc * h / 3.0
end

function branch_lengths_by_kind(tree::TreeSim.Tree)
    internal = Float64[]
    external = Float64[]
    sampled_unary = Float64[]
    isempty(tree) && return (internal=internal, external=external, sampled_unary=sampled_unary)

    for i in eachindex(tree)
        p = tree.parent[i]
        p == 0 && continue
        len = tree.time[i] - tree.time[p]
        if tree.kind[i] == TreeSim.Binary
            push!(internal, len)
        elseif tree.kind[i] == TreeSim.SampledLeaf
            push!(external, len)
        elseif tree.kind[i] == TreeSim.SampledUnary
            push!(sampled_unary, len)
        end
    end
    return (internal=internal, external=external, sampled_unary=sampled_unary)
end

function forest_branch_lengths(forest)
    internal = Float64[]
    external = Float64[]
    sampled_unary = Float64[]
    for tree in forest
        x = branch_lengths_by_kind(tree)
        append!(internal, x.internal)
        append!(external, x.external)
        append!(sampled_unary, x.sampled_unary)
    end
    return (internal=internal, external=external, sampled_unary=sampled_unary)
end

function safe_mean(xs)
    isempty(xs) ? NaN : mean(xs)
end

function summarize_case(; seed, pars, tj, tk, nsims)
    rng = MersenneTwister(seed)

    n_empty = 0
    n_single = 0
    n_multi = 0
    cherries = Int[]
    nodes = Int[]
    internal_counts = Int[]
    external_counts = Int[]
    sampled_unary_counts = Int[]
    internal_lengths = Float64[]
    external_lengths = Float64[]
    sampled_unary_lengths = Float64[]

    for _ in 1:nsims
        log = simulate_bd(rng, pars, tk; initial_lineages=1, apply_ρ₀=false)
        forest = TreeSim.forest_from_eventlog(log; tj, tk)

        if isempty(forest)
            n_empty += 1
            continue
        elseif length(forest) == 1
            n_single += 1
        else
            n_multi += 1
            continue
        end

        counts = reconstructed_forest_stat_counts(forest)
        lengths = forest_branch_lengths(forest)
        push!(cherries, counts.cherries)
        push!(nodes, counts.node_count)
        push!(internal_counts, counts.internal_branches)
        push!(external_counts, counts.external_branches)
        push!(sampled_unary_counts, counts.sampled_unary_branches)
        append!(internal_lengths, lengths.internal)
        append!(external_lengths, lengths.external)
        append!(sampled_unary_lengths, lengths.sampled_unary)
    end

    analytical_cherries = expected_reconstructed_cherries(tj, tk, pars)
    analytical_nodes = simpson(x -> reconstructed_node_depth_intensity(x, tj, tk, pars), 0.0, tk - tj)

    println()
    println("Parameter regime")
    println("  pars = $pars")
    println("  window = ($tj, $tk], initial_lineages = 1, apply_ρ₀ = false")
    println("  RNG seed = $seed, replicates = $nsims")
    println()
    println("Forest handling")
    println("  Empty forests are excluded from conditional tree-statistic means.")
    println("  Multiple-component forests are reported and excluded; with tj=0 and one initial lineage they should be zero.")
    @printf("  empty=%d single=%d multi=%d retained_fraction=%.4f\n", n_empty, n_single, n_multi, n_single / nsims)
    @printf("  analytical retention probability P(A(tj)=1)=%.4f\n", 1 - unsampled_probability(tj, tk, pars))
    println()
    println("Conditional single-component comparison")
    @printf("%-24s %14s %14s %14s\n", "quantity", "empirical", "analytical", "difference")
    @printf("%-24s %14.6f %14.6f %14.6f\n", "cherries", mean(cherries), analytical_cherries, mean(cherries) - analytical_cherries)
    @printf("%-24s %14.6f %14.6f %14.6f\n", "binary nodes", mean(nodes), analytical_nodes, mean(nodes) - analytical_nodes)
    @printf("%-24s %14.6f %14s %14s\n", "internal branches", mean(internal_counts), "reported", "n/a")
    @printf("%-24s %14.6f %14s %14s\n", "external branches", mean(external_counts), "reported", "n/a")
    @printf("%-24s %14.6f %14s %14s\n", "sampled-unary branches", mean(sampled_unary_counts), "reported", "n/a")
    println()
    println("Pooled empirical branch lengths")
    println("  Branch spectra in the TeX are ideal reconstructed-process branch-count densities.")
    println("  TreeSim reconstructed forests retain sampled-unary sampled-ancestor nodes, so these pools are reported rather than treated as a direct equality check.")
    @printf("  internal mean length: %.6f from %d branches\n", safe_mean(internal_lengths), length(internal_lengths))
    @printf("  external mean length: %.6f from %d branches\n", safe_mean(external_lengths), length(external_lengths))
    @printf("  sampled-unary mean length: %.6f from %d branches\n", safe_mean(sampled_unary_lengths), length(sampled_unary_lengths))
end

cases = [
    (seed=20260417, pars=ConstantRateBDParameters(1.35, 0.45, 0.75, 0.65), tj=0.0, tk=3.0, nsims=20_000),
    (seed=20260418, pars=ConstantRateBDParameters(0.95, 0.60, 0.55, 0.80), tj=0.0, tk=4.0, nsims=20_000),
]

for case in cases
    summarize_case(; case...)
end
