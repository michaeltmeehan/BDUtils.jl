using BDUtils
using TreeSim

tree = Tree(
    [0.0, 0.6, 1.0, 1.4],
    [2, 3, 0, 0],
    [0, 4, 0, 0],
    [0, 1, 2, 2],
    [Root, Binary, SampledLeaf, SampledLeaf],
    [0, 0, 0, 0],
    [0, 0, 101, 102],
)

params = UncolouredMTBD2ConstantParameters(
    [0.8 0.35; 0.25 0.7],
    [0.2, 0.3],
    [0.45, 0.55],
    [0.7, 0.4],
    [0.0 0.12; 0.08 0.0],
    [0.2, 0.3],
)

tip_states = Dict(3 => 1, 4 => 2)
ll = loglikelihood_uncoloured_mtbd2(
    tree,
    params,
    tip_states;
    root_prior=[0.6, 0.4],
)

println("uncoloured MTBD-2 known-tip log likelihood = ", ll)

mixed_tip_states = Dict(3 => 1, 4 => missing)
mixed_ll = loglikelihood_uncoloured_mtbd2(
    tree,
    params,
    mixed_tip_states;
    root_prior=[0.6, 0.4],
)

println("uncoloured MTBD-2 mixed known/unknown-tip log likelihood = ", mixed_ll)

sampled_unary_tree = Tree(
    [0.0, 0.4, 0.8, 1.0, 1.2],
    [2, 3, 4, 0, 0],
    [0, 5, 0, 0, 0],
    [0, 1, 2, 3, 2],
    [Root, Binary, SampledUnary, SampledLeaf, SampledLeaf],
    [0, 0, 0, 0, 0],
    [0, 0, 201, 202, 203],
)

sampled_unary_states = Dict(3 => 1, 4 => missing, 5 => 1)
sampled_unary_ll = loglikelihood_uncoloured_mtbd2(
    sampled_unary_tree,
    params,
    sampled_unary_states;
    root_prior=[0.6, 0.4],
)

println("uncoloured MTBD-2 sampled-unary log likelihood = ", sampled_unary_ll)
