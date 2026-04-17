module BDUtils

using TreeSim
using LinearAlgebra

include("parameters.jl")
include("pgf.jl")
include("likelihood_constant.jl")
include("fit_constant.jl")
include("derived.jl")

export ConstantRateBDParameters,
       bd_coefficients,
       gamma_bd,
       alpha_bd,
       beta_bd,
       pn_birthdeath,
       γ,
       α,
       β,
       pₙ,
       E_constant,
       g_constant,
       logaddexp,
       bd_loglikelihood_constant,
       BDFixedSpec,
       AbstractBDParameterization,
       RateParameterization,
       R0DeltaParameterization,
       expand_rates,
       backtransform,
       fit_bd_full,
       fit_bd_pars,
       fit_bd_ensemble_mle,
       compute_R0,
       compute_delta,
       compute_sampling_fraction,
       compute_sampled_removal_rate,
       parameters_from_R0_delta_s_r,
       reparameterize_R0_delta_s

end
