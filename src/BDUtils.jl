module BDUtils

using TreeSim
using LinearAlgebra

include("pgf.jl")
include("likelihood_constant.jl")
include("fit_constant.jl")
include("derived.jl")

export bd_coefficients,
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
       compute_delta

end
