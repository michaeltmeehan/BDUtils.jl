using BDUtils
using Random

function empirical_no_observation(pars, t; initial_type::Int, nsims::Int, seed::Int)
    rng = MersenneTwister(seed)
    no_observed = 0
    for _ in 1:nsims
        log = simulate_multitype_bd(rng, pars, t; initial_types=[initial_type])
        no_observed += sum(multitype_S_at(log, t)) == 0
    end
    return no_observed / nsims
end

function print_no_observation_table(pars, t; nsims=10_000, tolerance=0.025)
    analytical = multitype_E(t, pars; steps_per_unit=512)
    println("no-observation validation at t = ", t, " with nsims = ", nsims)
    println("type, empirical, analytical, abs_error")
    ok = true
    for a in 1:length(pars)
        empirical = empirical_no_observation(pars, t; initial_type=a, nsims=nsims, seed=20260417 + a)
        err = abs(empirical - analytical[a])
        println(a, ", ", empirical, ", ", analytical[a], ", ", err)
        ok &= err <= tolerance
    end
    return ok
end

function check_likelihood_factorization(pars)
    tree = MultitypeColoredTree(
        1.0,
        1;
        segments=[
            MultitypeColoredSegment(1, 0.5, 1.0),
            MultitypeColoredSegment(1, 0.0, 0.5),
            MultitypeColoredSegment(2, 0.0, 0.5),
        ],
        births=[MultitypeColoredBirth(0.5, 1, 2)],
        present_samples=[1, 2],
    )
    logflow_05 = multitype_log_flow(0.5, pars)
    logflow_10 = multitype_log_flow(1.0, pars)
    manual = (logflow_10[1] - logflow_05[1]) + logflow_05[1] + logflow_05[2] +
             log(pars.birth[1, 2]) + log(pars.ρ₀[1]) + log(pars.ρ₀[2])
    computed = multitype_colored_loglikelihood(tree, pars)
    println("colored likelihood factorization")
    println("manual_loglik = ", manual)
    println("computed_loglik = ", computed)
    println("abs_error = ", abs(manual - computed))
    return isapprox(manual, computed; atol=1e-10, rtol=1e-10)
end

pars = MultitypeBDParameters(
    [0.15 0.08; 0.04 0.12],
    [0.35, 0.25],
    [0.20, 0.30],
    [0.70, 0.40],
    [0.0 0.10; 0.06 0.0],
    [0.15, 0.25],
)

println("BDUtils multitype semantic validation")
ok_E = print_no_observation_table(pars, 0.8)
ok_ll = check_likelihood_factorization(pars)

if ok_E && ok_ll
    println("validation passed")
else
    error("validation failed; inspect empirical/analytical discrepancies above")
end
