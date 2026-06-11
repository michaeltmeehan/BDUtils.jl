using Test
using BDUtils
using TreeSim
using Random

include("test_helpers.jl")

const λ = 2.0
const μ = 0.5
const ψ = 0.4
const r = 0.7
const pars = ConstantRateBDParameters(λ, μ, ψ, r, 0.25)

@testset "BDUtils constant-rate core" begin
    include("test_parameters.jl")
    include("test_simulation.jl")
    include("test_multitype.jl")
    include("test_treesim_extraction.jl")
    include("test_pgf_primitives.jl")
    include("test_original_process_probabilities.jl")
    include("test_original_process_validation.jl")
    include("test_reconstructed_simulation.jl")
    include("test_stress.jl")
    include("test_ode_invariants.jl")
    include("test_reconstructed_process.jl")
    include("test_hidden_reconstructed_inversion.jl")
    include("test_conditioned_kernels.jl")
    include("test_sampling_time_likelihood.jl")
    include("test_reconstructed_ode_invariants.jl")
    include("test_likelihood_helpers.jl")
end
