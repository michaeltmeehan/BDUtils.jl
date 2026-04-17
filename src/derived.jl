"""
    compute_sampled_removal_rate(pars::ConstantRateBDParameters)

Return the sampled-removal rate `r * ψ` for the canonical constant-rate
birth-death-sampling parameters.
"""
compute_sampled_removal_rate(pars::ConstantRateBDParameters) = pars.r * pars.ψ

compute_sampled_removal_rate(λ, μ, ψ, r) = r .* ψ

"""
    compute_delta(pars::ConstantRateBDParameters)

Return `δ = μ + r * ψ`, the total removal rate in the `(R0, δ, s, r, ρ₀)`
coordinate system. `ConstantRateBDParameters` remains the canonical stored model
object; this is a derived coordinate.
"""
compute_delta(pars::ConstantRateBDParameters) = pars.μ + compute_sampled_removal_rate(pars)

compute_delta(λ, μ, ψ, r) = μ .+ compute_sampled_removal_rate(λ, μ, ψ, r)

"""
    compute_R0(pars::ConstantRateBDParameters)

Return `R0 = λ / δ`, where `δ = μ + r * ψ`.
"""
compute_R0(pars::ConstantRateBDParameters) = pars.λ / compute_delta(pars)

compute_R0(λ, μ, ψ, r) = λ ./ compute_delta(λ, μ, ψ, r)

"""
    compute_sampling_fraction(pars::ConstantRateBDParameters)

Return `s = r * ψ / δ`, where `δ = μ + r * ψ`.
"""
compute_sampling_fraction(pars::ConstantRateBDParameters) = compute_sampled_removal_rate(pars) / compute_delta(pars)

compute_sampling_fraction(λ, μ, ψ, r) = compute_sampled_removal_rate(λ, μ, ψ, r) ./ compute_delta(λ, μ, ψ, r)

@inline function _check_finite_real(name::AbstractString, x::T) where {T<:AbstractFloat}
    isfinite(x) || throw(ArgumentError("$name must be finite."))
    return nothing
end

"""
    parameters_from_R0_delta_s_r(R0, δ, s, r, ρ₀=0)

Convert the alternative constant-rate coordinates `(R0, δ, s, r, ρ₀)` to the
canonical `ConstantRateBDParameters(λ, μ, ψ, r, ρ₀)`.

Here `δ = μ + r * ψ`, `R0 = λ / δ`, and `s = r * ψ / δ`. The conversion is
singular at `r = 0`: only `s = 0` is valid there, and the returned parameters
use `ψ = 0`, `μ = δ`.
"""
function parameters_from_R0_delta_s_r(R0::Real, δ::Real, s::Real, r::Real, ρ₀::Real=0.0)
    T = promote_type(typeof(R0), typeof(δ), typeof(s), typeof(r), typeof(ρ₀), Float64)
    R0T, δT, sT, rT, ρ₀T = T(R0), T(δ), T(s), T(r), T(ρ₀)

    _check_finite_real("R0", R0T)
    _check_finite_real("δ", δT)
    _check_probability("s", sT)
    _check_probability("r", rT)
    _check_probability("ρ₀", ρ₀T)

    R0T > zero(T) || throw(ArgumentError("R0 must be positive because ConstantRateBDParameters requires λ > 0."))
    δT > zero(T) || throw(ArgumentError("δ must be positive."))

    λ = R0T * δT
    μ = (one(T) - sT) * δT
    ψ = if rT == zero(T)
        sT == zero(T) || throw(ArgumentError("s must be 0 when r is 0."))
        zero(T)
    else
        sT * δT / rT
    end

    return ConstantRateBDParameters{T}(λ, μ, ψ, rT, ρ₀T)
end

"""
    reparameterize_R0_delta_s(pars::ConstantRateBDParameters)

Return the alternative constant-rate coordinates as a named tuple:
`(; R0, δ, s, r, ρ₀)`.

This is an invertible coordinate system for valid non-singular inputs, not a
replacement for the canonical `ConstantRateBDParameters` model object.
"""
function reparameterize_R0_delta_s(pars::ConstantRateBDParameters)
    δ = compute_delta(pars)
    return (;
        R0 = pars.λ / δ,
        δ = δ,
        s = compute_sampled_removal_rate(pars) / δ,
        r = pars.r,
        ρ₀ = pars.ρ₀,
    )
end
