using BDUtils

pars = MultitypeBDParameters(
    [0.4 0.8; 0.2 0.5],
    [0.1, 0.2],
    [0.3, 0.6],
    [0.2, 0.7],
    [0.0 0.9; 0.1 0.0],
    [0.4, 0.5],
)

trees = [
    MultitypeColoredTree(
        1.0,
        1;
        segments=[
            MultitypeColoredSegment(1, 0.8, 1.0),
            MultitypeColoredSegment(1, 0.5, 0.8),
            MultitypeColoredSegment(2, 0.0, 0.8),
        ],
        births=[MultitypeColoredBirth(0.8, 1, 2)],
        transitions=[MultitypeColoredTransition(0.5, 1, 2)],
        present_samples=[2],
    ),
    MultitypeColoredTree(
        1.0,
        1;
        segments=[
            MultitypeColoredSegment(1, 0.8, 1.0),
            MultitypeColoredSegment(2, 0.2, 0.8),
        ],
        hidden_births=[MultitypeColoredHiddenBirth(0.8, 1, 2)],
        terminal_samples=[MultitypeColoredSampling(0.2, 2)],
    ),
]

println("loglikelihood at template: ", multitype_loglikelihood(trees, pars))

initial = MultitypeBDParameters(
    [0.4 0.2; 0.2 0.5],
    pars.death,
    pars.sampling,
    pars.removal_probability,
    pars.transition,
    pars.ρ₀,
)

spec = MultitypeMLESpec(
    initial;
    fit_birth=[false true; false false],
    fit_death=[false, false],
    fit_sampling=[false, false],
    fit_transition=[false false; false false],
)

fit = fit_multitype_mle(trees; spec=spec, lower=[log(0.05)], upper=[log(5.0)])
println("fitted birth[1,2]: ", fit.parameters.birth[1, 2])
println("fitted loglikelihood: ", fit.loglikelihood)
