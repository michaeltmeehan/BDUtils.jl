using BDUtils
using Printf
using TreeSim

# Lightweight packet-12 diagnostic for the narrow zero-transition superspreader
# parameterisation. This is not an identifiability proof: with unknown sampled-node
# states, total_R0, superspreader_fraction, and relative_transmissibility can be
# weakly separated because the state labels are latent and anagenetic transitions
# are fixed to zero.

const ROOT_PRIOR = [0.6, 0.4]
const STEPS_PER_UNIT = 64
const MIN_STEPS = 8

function diagnostic_trees()
    binary = Tree(
        [0.0, 0.6, 1.0, 1.4],
        [2, 3, 0, 0],
        [0, 4, 0, 0],
        [0, 1, 2, 2],
        [Root, Binary, SampledLeaf, SampledLeaf],
        [0, 0, 0, 0],
        [0, 0, 101, 102],
    )
    sampled_unary = Tree(
        [0.0, 0.4, 0.8, 1.0, 1.2],
        [2, 3, 4, 0, 0],
        [0, 5, 0, 0, 0],
        [0, 1, 2, 3, 2],
        [Root, Binary, SampledUnary, SampledLeaf, SampledLeaf],
        [0, 0, 0, 0, 0],
        [0, 0, 201, 202, 203],
    )
    return [binary, sampled_unary]
end

function diagnostic_observations()
    return (
        known=[
            Dict(3 => 1, 4 => 2),
            Dict(3 => 1, 4 => 2, 5 => 1),
        ],
        unknown=[
            Dict(3 => missing, 4 => missing),
            Dict(3 => missing, 4 => missing, 5 => missing),
        ],
        mixed=[
            Dict(3 => 1, 4 => missing),
            Dict(3 => 1, 4 => missing, 5 => 1),
        ],
    )
end

function baseline_superspreader_parameters()
    return UncolouredMTBD2SuperspreaderParameters(
        1.8,
        0.2,
        6.0,
        [0.15, 0.25],
        [0.5, 0.7],
        [0.6, 0.4],
        [0.1, 0.2],
    )
end

function replace_coordinate(base::UncolouredMTBD2SuperspreaderParameters;
                            total_R0=base.total_R0,
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

function slice_loglikelihoods(trees, observations, base, coordinate::Symbol, grid)
    return [
        total_loglikelihood_uncoloured_mtbd2_superspreader(
            trees,
            coordinate === :total_R0 ? replace_coordinate(base; total_R0=x) :
            coordinate === :superspreader_fraction ? replace_coordinate(base; superspreader_fraction=x) :
            coordinate === :relative_transmissibility ? replace_coordinate(base; relative_transmissibility=x) :
            throw(ArgumentError("unsupported coordinate $coordinate")),
            observations;
            root_prior=ROOT_PRIOR,
            steps_per_unit=STEPS_PER_UNIT,
            min_steps=MIN_STEPS,
        )
        for x in grid
    ]
end

function print_slice_summary(label, grid, values)
    order = sortperm(values; rev=true)
    best = order[1]
    contrast = maximum(values) - minimum(values)
    @printf("  %-8s argmax=%7.3f  max=%10.4f  contrast=%9.4f\n",
            label, grid[best], values[best], contrast)
    top = order[1:min(3, length(order))]
    print("           top: ")
    println(join((@sprintf("%.3f=>%.4f", grid[i], values[i]) for i in top), ", "))
end

function print_2d_summary(trees, observations, base)
    q_grid = collect(0.05:0.05:0.45)
    rel_grid = collect(1.0:1.0:9.0)
    best = (ll=-Inf, q=NaN, rel=NaN)
    worst = Inf
    for q in q_grid, rel in rel_grid
        pars = replace_coordinate(base; superspreader_fraction=q, relative_transmissibility=rel)
        ll = total_loglikelihood_uncoloured_mtbd2_superspreader(
            trees, pars, observations;
            root_prior=ROOT_PRIOR,
            steps_per_unit=STEPS_PER_UNIT,
            min_steps=MIN_STEPS,
        )
        if ll > best.ll
            best = (ll=ll, q=q, rel=rel)
        end
        worst = min(worst, ll)
    end
    @printf("  q x rel grid: argmax=(q=%.3f, rel=%.3f)  max=%10.4f  contrast=%9.4f\n",
            best.q, best.rel, best.ll, best.ll - worst)
end

function main()
    trees = diagnostic_trees()
    modes = diagnostic_observations()
    base = baseline_superspreader_parameters()
    slices = (
        total_R0=collect(0.8:0.25:2.8),
        superspreader_fraction=collect(0.05:0.05:0.55),
        relative_transmissibility=collect(1.0:1.0:10.0),
    )

    println("Uncoloured MTBD-2 superspreader likelihood slices")
    println("baseline = ", uncoloured_mtbd2_superspreader_parameter_vector(base))
    println("trees = ", length(trees), " (binary + sampled-unary toy examples)")
    println()

    for coordinate in keys(slices)
        grid = getfield(slices, coordinate)
        println("Slice: ", coordinate)
        for mode in keys(modes)
            values = slice_loglikelihoods(trees, getfield(modes, mode), base, coordinate, grid)
            print_slice_summary(String(mode), grid, values)
        end
        println()
    end

    println("Compact 2D q x relative_transmissibility summaries")
    for mode in keys(modes)
        println("Mode: ", mode)
        print_2d_summary(trees, getfield(modes, mode), base)
    end
end

main()
