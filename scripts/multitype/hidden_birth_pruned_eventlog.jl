using BDUtils

log = MultitypeBDEventLog(
    [0.2, 0.5, 0.8],
    [2, 1, 2],
    [1, 0, 0],
    [MultitypeBirth, MultitypeDeath, MultitypeSerialSampling],
    [1, 1, 2],
    [2, 1, 2],
    [1],
    1.0,
)

pars = MultitypeBDParameters(
    [0.2 0.7; 0.1 0.3],
    [0.2, 0.2],
    [0.4, 0.5],
    [0.5, 0.5],
    [0.0 0.1; 0.1 0.0],
    [0.0, 0.0],
)

tree = validate_pruned_multitype_colored_tree_from_eventlog(log, pars)
println("segments: ", length(tree.segments))
println("ordinary birth events: ", length(tree.births))
println("hidden birth events: ", length(tree.hidden_births))
println("terminal samples: ", length(tree.terminal_samples))
println("loglikelihood: ", multitype_colored_loglikelihood(tree, pars))
