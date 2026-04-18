using BDUtils

log = MultitypeBDEventLog(
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

pars = MultitypeBDParameters(
    [0.4 0.8; 0.2 0.5],
    [0.1, 0.2],
    [0.3, 0.6],
    [0.2, 0.7],
    [0.0 0.9; 0.1 0.0],
    [0.4, 0.5],
)

tree = validate_pruned_multitype_colored_tree_from_eventlog(log, pars)
println("pruned segments: ", length(tree.segments))
println("retained birth events: ", length(tree.births))
println("hidden birth events: ", length(tree.hidden_births))
println("retained transition events: ", length(tree.transitions))
println("terminal samples: ", length(tree.terminal_samples))
println("ancestral samples: ", length(tree.ancestral_samples))
println("present samples: ", length(tree.present_samples))
println("loglikelihood: ", multitype_colored_loglikelihood(tree, pars))
