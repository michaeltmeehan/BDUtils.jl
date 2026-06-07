"""
    conditioned_reconstructed_alpha_bd(w, tᵢ, tⱼ, tₖ, pars)

Zero-count parameter `α̃(w,tᵢ,tⱼ,tₖ)` for the reconstructed process
conditioned on the initial lineage being reconstructed (`Aᵢᵏ = 1`). These
parameters correspond to
`G̃ᵢⱼᵏ(z,w) = (Gᵢⱼ(pⱼᵏ + qⱼᵏ z,w) - pᵢᵏ) / qᵢᵏ`.
"""
function conditioned_reconstructed_alpha_bd(w::T, tᵢ::T, tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    return _conditioned_reconstructed_alpha_beta_gamma(w, tᵢ, tⱼ, tₖ, pars).α
end

function conditioned_reconstructed_alpha_bd(w::Real, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    wT, tiT, tjT, tkT, pT = _promote_reconstructed_inputs(w, tᵢ, tⱼ, tₖ, pars)
    return conditioned_reconstructed_alpha_bd(wT, tiT, tjT, tkT, pT)
end

"""
    conditioned_reconstructed_beta_bd(w, tᵢ, tⱼ, tₖ, pars)

Geometric mass parameter `β̃(w,tᵢ,tⱼ,tₖ)` for the reconstructed process
conditioned on the initial lineage being reconstructed (`Aᵢᵏ = 1`). These
parameters correspond to
`G̃ᵢⱼᵏ(z,w) = (Gᵢⱼ(pⱼᵏ + qⱼᵏ z,w) - pᵢᵏ) / qᵢᵏ`.
"""
function conditioned_reconstructed_beta_bd(w::T, tᵢ::T, tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    return _conditioned_reconstructed_alpha_beta_gamma(w, tᵢ, tⱼ, tₖ, pars).β
end

function conditioned_reconstructed_beta_bd(w::Real, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    wT, tiT, tjT, tkT, pT = _promote_reconstructed_inputs(w, tᵢ, tⱼ, tₖ, pars)
    return conditioned_reconstructed_beta_bd(wT, tiT, tjT, tkT, pT)
end

"""
    conditioned_reconstructed_gamma_bd(w, tᵢ, tⱼ, tₖ, pars)

Geometric ratio parameter `γ̃(w,tᵢ,tⱼ,tₖ)` for the reconstructed process
conditioned on the initial lineage being reconstructed (`Aᵢᵏ = 1`). These
parameters correspond to
`G̃ᵢⱼᵏ(z,w) = (Gᵢⱼ(pⱼᵏ + qⱼᵏ z,w) - pᵢᵏ) / qᵢᵏ`.
"""
function conditioned_reconstructed_gamma_bd(w::T, tᵢ::T, tⱼ::T, tₖ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    return _conditioned_reconstructed_alpha_beta_gamma(w, tᵢ, tⱼ, tₖ, pars).γ
end

function conditioned_reconstructed_gamma_bd(w::Real, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    wT, tiT, tjT, tkT, pT = _promote_reconstructed_inputs(w, tᵢ, tⱼ, tₖ, pars)
    return conditioned_reconstructed_gamma_bd(wT, tiT, tjT, tkT, pT)
end

"""
    conditioned_reconstructed_pgf(z, w, tᵢ, tⱼ, tₖ, pars)

Single-lineage PGF `α̃ + β̃*z/(1 - γ̃*z)` for the reconstructed process
conditioned on the initial lineage being reconstructed (`Aᵢᵏ = 1`):
`E[z^(Aⱼᵏ) w^(Sᵢⱼ) | Aᵢᵏ = 1, Sᵢ = 0]`.
"""
function conditioned_reconstructed_pgf(z::Real, w::Real, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    wT, tiT, tjT, tkT, pT = _promote_reconstructed_inputs(w, tᵢ, tⱼ, tₖ, pars)
    zT = typeof(wT)(z)
    pars3 = _conditioned_reconstructed_alpha_beta_gamma(wT, tiT, tjT, tkT, pT)
    return pars3.α + pars3.β * zT / _stabilize_denominator(one(typeof(wT)) - pars3.γ * zT)
end

"""
    conditioned_reconstructed_xi(tᵢ, tⱼ, tₖ, pars)

Zero-count probability `ξ̃ = α̃(1,tᵢ,tⱼ,tₖ)` for the reconstructed count
conditioned on the initial lineage being reconstructed (`Aᵢᵏ = 1`).
"""
conditioned_reconstructed_xi(tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters) =
    conditioned_reconstructed_alpha_bd(1.0, tᵢ, tⱼ, tₖ, pars)

"""
    conditioned_reconstructed_eta(tᵢ, tⱼ, tₖ, pars)

Geometric ratio `η̃ = γ̃(1,tᵢ,tⱼ,tₖ)` for the reconstructed count conditioned
on the initial lineage being reconstructed (`Aᵢᵏ = 1`).
"""
conditioned_reconstructed_eta(tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters) =
    conditioned_reconstructed_gamma_bd(1.0, tᵢ, tⱼ, tₖ, pars)

"""
    conditioned_reconstructed_count_pmf(a, tᵢ, tⱼ, tₖ, pars)

Preferred PMF for probabilities of the form
`P(Aⱼᵏ = a | Aᵢᵏ = 1)`, i.e. the reconstructed lineage count at `tⱼ`
conditioned on the initial lineage at `tᵢ` being reconstructed by horizon `tₖ`.
"""
function conditioned_reconstructed_count_pmf(a::Integer, tᵢ::Real, tⱼ::Real, tₖ::Real, pars::ConstantRateBDParameters)
    _check_count("a", a)
    ξ = conditioned_reconstructed_xi(tᵢ, tⱼ, tₖ, pars)
    a == 0 && return ξ
    η = conditioned_reconstructed_eta(tᵢ, tⱼ, tₖ, pars)
    return (1 - ξ) * (1 - η) * η^(a - 1)
end
