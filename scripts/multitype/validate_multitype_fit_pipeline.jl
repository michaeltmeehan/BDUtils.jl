using BDUtils
using Random

function collect_pruned_trees(pars; seed, target_trees, max_attempts, tmax=1.0, initial_type=1)
    rng = MersenneTwister(seed)
    trees = MultitypeColoredTree[]
    rejected = 0
    hidden_birth_trees = 0
    for _ in 1:max_attempts
        length(trees) >= target_trees && break
        log = simulate_multitype_bd(rng, pars, tmax; initial_types=[initial_type], apply_ρ₀=false)
        try
            tree = pruned_multitype_colored_tree_from_eventlog(log)
            push!(trees, tree)
            hidden_birth_trees += !isempty(tree.hidden_births)
        catch err
            err isa ArgumentError || rethrow()
            rejected += 1
        end
    end
    return (trees=trees, rejected=rejected, hidden_birth_trees=hidden_birth_trees)
end

function fit_birth12_scenario(; seed, target_trees=12, max_attempts=2_000)
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
    data = collect_pruned_trees(truth; seed=seed, target_trees=target_trees, max_attempts=max_attempts)
    spec = MultitypeMLESpec(initial;
                            fit_birth=[false true; false false],
                            fit_death=falses(2),
                            fit_sampling=falses(2),
                            fit_transition=falses(2, 2))
    initial_ll = multitype_loglikelihood(data.trees, initial)
    truth_ll = multitype_loglikelihood(data.trees, truth)
    fit = fit_multitype_mle(data.trees; spec=spec, lower=[log(0.05)], upper=[log(5.0)],
                            initial_step=0.5, tolerance=1e-4, maxiter=300)
    return (
        name="birth12_hidden_birth_pruning",
        true_value=truth.birth[1, 2],
        initial_value=initial.birth[1, 2],
        fitted_value=fit.parameters.birth[1, 2],
        initial_ll=initial_ll,
        truth_ll=truth_ll,
        fitted_ll=fit.loglikelihood,
        retained=length(data.trees),
        rejected=data.rejected,
        hidden_birth_trees=data.hidden_birth_trees,
        converged=fit.result.converged,
    )
end

function fit_transition12_scenario(; seed, target_trees=10, max_attempts=2_000)
    truth = MultitypeBDParameters(
        zeros(2, 2),
        [0.15, 0.25],
        [0.05, 0.8],
        [1.0, 1.0],
        [0.0 1.1; 0.0 0.0],
        [0.0, 0.0],
    )
    initial = MultitypeBDParameters(
        truth.birth,
        truth.death,
        truth.sampling,
        truth.removal_probability,
        [0.0 0.25; 0.0 0.0],
        truth.ρ₀,
    )
    data = collect_pruned_trees(truth; seed=seed, target_trees=target_trees, max_attempts=max_attempts)
    spec = MultitypeMLESpec(initial;
                            fit_birth=falses(2, 2),
                            fit_death=falses(2),
                            fit_sampling=falses(2),
                            fit_transition=[false true; false false])
    initial_ll = multitype_loglikelihood(data.trees, initial)
    truth_ll = multitype_loglikelihood(data.trees, truth)
    fit = fit_multitype_mle(data.trees; spec=spec, lower=[log(0.05)], upper=[log(5.0)],
                            initial_step=0.5, tolerance=1e-4, maxiter=300)
    return (
        name="transition12_two_type",
        true_value=truth.transition[1, 2],
        initial_value=initial.transition[1, 2],
        fitted_value=fit.parameters.transition[1, 2],
        initial_ll=initial_ll,
        truth_ll=truth_ll,
        fitted_ll=fit.loglikelihood,
        retained=length(data.trees),
        rejected=data.rejected,
        hidden_birth_trees=data.hidden_birth_trees,
        converged=fit.result.converged,
    )
end

function fit_k1_sampling_scenario(; seed, target_trees=12, max_attempts=2_000)
    truth = MultitypeBDParameters([0.0;;], [0.2], [0.7], [1.0], [0.0;;], [0.0])
    initial = MultitypeBDParameters([0.0;;], truth.death, [0.2], truth.removal_probability, truth.transition, truth.ρ₀)
    data = collect_pruned_trees(truth; seed=seed, target_trees=target_trees, max_attempts=max_attempts)
    spec = MultitypeMLESpec(initial;
                            fit_birth=falses(1, 1),
                            fit_death=falses(1),
                            fit_sampling=[true],
                            fit_transition=falses(1, 1))
    initial_ll = multitype_loglikelihood(data.trees, initial)
    truth_ll = multitype_loglikelihood(data.trees, truth)
    fit = fit_multitype_mle(data.trees; spec=spec, lower=[log(0.05)], upper=[log(5.0)],
                            initial_step=0.5, tolerance=1e-4, maxiter=300)
    return (
        name="k1_sampling",
        true_value=truth.sampling[1],
        initial_value=initial.sampling[1],
        fitted_value=fit.parameters.sampling[1],
        initial_ll=initial_ll,
        truth_ll=truth_ll,
        fitted_ll=fit.loglikelihood,
        retained=length(data.trees),
        rejected=data.rejected,
        hidden_birth_trees=data.hidden_birth_trees,
        converged=fit.result.converged,
    )
end

function print_result(result)
    println()
    println("scenario: ", result.name)
    println("retained trees: ", result.retained, " rejected histories: ", result.rejected,
            " hidden-birth trees: ", result.hidden_birth_trees)
    println("truth: ", result.true_value, " initial: ", result.initial_value,
            " fitted: ", result.fitted_value)
    println("loglik initial: ", result.initial_ll,
            " truth: ", result.truth_ll,
            " fitted: ", result.fitted_ll)
    println("fit improved over initial: ", result.fitted_ll >= result.initial_ll,
            " converged: ", result.converged)
end

println("BDUtils multitype simulation-fit validation")
results = [
    fit_birth12_scenario(seed=20260418),
    fit_transition12_scenario(seed=20260419),
    fit_k1_sampling_scenario(seed=20260420),
]
foreach(print_result, results)

improved = count(r -> r.fitted_ll >= r.initial_ll, results)
println()
println("summary: ", improved, " / ", length(results), " scenarios improved over the initial template")
improved == length(results) || error("one or more fit scenarios failed to improve over the initial template")
