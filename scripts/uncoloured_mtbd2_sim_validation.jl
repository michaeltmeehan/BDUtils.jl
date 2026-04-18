using BDUtils
using Random
using TreeSim

const TRUE_PARS = UncolouredMTBD2ConstantParameters(
    [1.25 0.25; 0.12 0.85],
    [0.18, 0.28],
    [0.75, 0.55],
    [0.55, 0.35],
    [0.0 0.08; 0.05 0.0],
    [0.25, 0.20],
)

const PERTURBED_PARS = UncolouredMTBD2ConstantParameters(
    [0.65 0.08; 0.30 1.45],
    [0.35, 0.12],
    [0.35, 0.95],
    [0.20, 0.80],
    [0.0 0.22; 0.01 0.0],
    [0.08, 0.35],
)

function multitype_for_simulation(pars::UncolouredMTBD2ConstantParameters)
    return MultitypeBDParameters(
        pars.birth,
        pars.death,
        pars.sampling,
        pars.removal_probability,
        pars.transition,
        pars.ρ₀,
    )
end

function plain_eventlog(log::MultitypeBDEventLog)
    times = Float64[]
    lineages = Int[]
    parents = Int[]
    kinds = BDEventKind[]
    for i in 1:length(log)
        kind = log.kind[i]
        if kind == MultitypeBirth
            push!(times, log.time[i])
            push!(lineages, log.lineage[i])
            push!(parents, log.parent[i])
            push!(kinds, Birth)
        elseif kind == MultitypeDeath
            push!(times, log.time[i])
            push!(lineages, log.lineage[i])
            push!(parents, log.parent[i])
            push!(kinds, Death)
        elseif kind == MultitypeFossilizedSampling
            push!(times, log.time[i])
            push!(lineages, log.lineage[i])
            push!(parents, log.parent[i])
            push!(kinds, FossilizedSampling)
        elseif kind == MultitypeSerialSampling
            push!(times, log.time[i])
            push!(lineages, log.lineage[i])
            push!(parents, log.parent[i])
            push!(kinds, SerialSampling)
        end
    end
    return BDEventLog(times, lineages, parents, kinds, length(log.initial_types), log.tmax)
end

function sampled_state_observations(tree::TreeSim.Tree, log::MultitypeBDEventLog)
    observations = Dict{Int,Int}()
    for node in eachindex(tree)
        tree.kind[node] in (TreeSim.SampledLeaf, TreeSim.SampledUnary) || continue
        event = findfirst(i -> log.lineage[i] == tree.host[node] &&
                               isapprox(log.time[i], tree.time[node]; atol=1e-10, rtol=0.0) &&
                               log.kind[i] in (MultitypeFossilizedSampling, MultitypeSerialSampling),
                          1:length(log))
        event === nothing && error("could not recover sampled state for tree node $node")
        observations[node] = log.type_before[event]
    end
    return observations
end

function mixed_observations(known::Dict{Int,Int})
    mixed = Dict{Int,Any}()
    for node in sort(collect(keys(known)))
        mixed[node] = isodd(node) ? known[node] : missing
    end
    return mixed
end

function admissible_sample(log::MultitypeBDEventLog)
    forest = forest_from_eventlog(plain_eventlog(log); tj=0.0, tk=log.tmax)
    length(forest) == 1 || return nothing
    tree = only(forest)
    all(kind -> kind in (TreeSim.Root, TreeSim.Binary, TreeSim.SampledLeaf, TreeSim.SampledUnary), tree.kind) || return nothing
    return (tree=tree, known=sampled_state_observations(tree, log))
end

function score_summary(rows, mode)
    diffs = [row[mode].true_ll - row[mode].perturbed_ll for row in rows]
    wins = count(>=(0), diffs)
    return (
        n=length(diffs),
        wins=wins,
        proportion=wins / length(diffs),
        mean_difference=sum(diffs) / length(diffs),
    )
end

function main(; seed=20260418, nsims=80, target_admissible=20, tmax=2.0)
    rng = MersenneTwister(seed)
    sim_pars = multitype_for_simulation(TRUE_PARS)
    rows = []
    simulated = 0

    while simulated < nsims && length(rows) < target_admissible
        simulated += 1
        log = simulate_multitype_bd(rng, sim_pars, tmax; initial_types=[1], apply_ρ₀=true)
        sample = admissible_sample(log)
        sample === nothing && continue

        tree = sample.tree
        known = sample.known
        unknown = Dict(node => missing for node in keys(known))
        mixed = mixed_observations(known)
        mode_observations = [known, unknown, mixed]
        repeated_tree = [tree, tree, tree]
        true_ll = loglikelihoods_uncoloured_mtbd2(repeated_tree, TRUE_PARS, mode_observations)
        perturbed_ll = loglikelihoods_uncoloured_mtbd2(repeated_tree, PERTURBED_PARS, mode_observations)

        push!(rows, (
            node_count=length(tree),
            sampled_unary=count(==(TreeSim.SampledUnary), tree.kind),
            known=(
                true_ll=true_ll[1],
                perturbed_ll=perturbed_ll[1],
            ),
            unknown=(
                true_ll=true_ll[2],
                perturbed_ll=perturbed_ll[2],
            ),
            mixed=(
                true_ll=true_ll[3],
                perturbed_ll=perturbed_ll[3],
            ),
        ))
    end

    println("simulated=", simulated)
    println("admissible=", length(rows))
    println("with_sampled_unary=", count(row -> row.sampled_unary > 0, rows))
    for mode in (:known, :unknown, :mixed)
        s = score_summary(rows, mode)
        println(mode,
                " n=", s.n,
                " true_ge_perturbed=", s.wins,
                " proportion=", round(s.proportion; digits=3),
                " mean_loglik_diff=", round(s.mean_difference; digits=3))
    end
end

main()
