using BDUtils
using Random

truth = MultitypeBDParameters(
    [0.0 0.9; 0.0 0.0],
    [0.15, 0.25],
    [0.05, 0.8],
    [1.0, 1.0],
    zeros(2, 2),
    [0.0, 0.0],
)

initial = MultitypeBDParameters(
    [0.0 0.25; 0.0 0.0],
    truth.death,
    truth.sampling,
    truth.removal_probability,
    truth.transition,
    truth.ρ₀,
)

function collect_observed_trees(pars; seed=20260418, target=8, max_attempts=1_000)
    rng = MersenneTwister(seed)
    trees = MultitypeColoredTree[]
    rejected = 0
    while length(trees) < target && length(trees) + rejected < max_attempts
        log = simulate_multitype_bd(rng, pars, 1.0; initial_types=[1], apply_ρ₀=false)
        try
            push!(trees, pruned_multitype_colored_tree_from_eventlog(log))
        catch err
            err isa ArgumentError || rethrow()
            rejected += 1
        end
    end
    return trees, rejected
end

trees, rejected = collect_observed_trees(truth)
isempty(trees) && error("no observed trees were extracted")

spec = MultitypeMLESpec(
    initial;
    fit_birth=[false true; false false],
    fit_death=falses(2),
    fit_sampling=falses(2),
    fit_transition=falses(2, 2),
)

truth_ll = multitype_loglikelihood(trees, truth)
initial_ll = multitype_loglikelihood(trees, initial)
fit = fit_multitype_mle(trees; spec=spec, lower=[log(0.05)], upper=[log(5.0)],
                        initial_step=0.5, tolerance=1e-4, maxiter=300)

hidden_birth_count = count(tree -> !isempty(tree.hidden_births), trees)

println("BDUtils constant-rate multitype worked example")
println("retained observed trees: ", length(trees))
println("rejected simulated histories: ", rejected)
println("trees with hidden births: ", hidden_birth_count)
println()
println("fitted parameter: birth[1,2]")
println("truth:   ", truth.birth[1, 2])
println("initial: ", initial.birth[1, 2])
println("fitted:  ", fit.parameters.birth[1, 2])
println()
println("loglikelihood at truth:   ", truth_ll)
println("loglikelihood at initial: ", initial_ll)
println("loglikelihood at fitted:  ", fit.loglikelihood)
println()
println("Interpretation:")
println("  The fitted value is the MLE for this small observed/pruned sample,")
println("  not a guarantee of exact recovery from one finite simulation.")
