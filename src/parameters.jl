"""
    ConstantRateBDParameters(λ, μ, ψ, r, ρ₀=0)

Parameters for the constant-rate generalized birth-death-sampling process.

Fields are the birth rate `λ`, death rate `μ`, sampling rate `ψ`, removal
probability at sampling `r`, and contemporaneous sampling probability `ρ₀`.
Validation requires finite rates, `λ > 0`, `μ >= 0`, `ψ >= 0`, and
`r, ρ₀ in [0, 1]`.

`λ > 0` is enforced at construction because the currently supported
constant-rate formulas divide by `λ` and the likelihood uses `log(λ)`.
The type stores floating-point values only; real inputs are promoted to a
floating type by the outer constructor.
"""
struct ConstantRateBDParameters{T<:AbstractFloat}
    λ::T
    μ::T
    ψ::T
    r::T
    ρ₀::T

    function ConstantRateBDParameters{T}(λ::T, μ::T, ψ::T, r::T, ρ₀::T=zero(T)) where {T<:AbstractFloat}
        _check_constant_bd_parameters(λ, μ, ψ, r; ρ₀=ρ₀)
        return new{T}(λ, μ, ψ, r, ρ₀)
    end
end

function ConstantRateBDParameters(λ::Real, μ::Real, ψ::Real, r::Real, ρ₀::Real=0.0)
    T = promote_type(typeof(λ), typeof(μ), typeof(ψ), typeof(r), typeof(ρ₀), Float64)
    return ConstantRateBDParameters{T}(T(λ), T(μ), T(ψ), T(r), T(ρ₀))
end

function Base.show(io::IO, pars::ConstantRateBDParameters)
    print(io, "ConstantRateBDParameters(",
          "λ=", pars.λ,
          ", μ=", pars.μ,
          ", ψ=", pars.ψ,
          ", r=", pars.r,
          ", ρ₀=", pars.ρ₀,
          ")")
end

@inline function _check_probability(name::AbstractString, x::T) where {T<:AbstractFloat}
    isfinite(x) || throw(ArgumentError("$name must be finite."))
    zero(T) <= x <= one(T) || throw(ArgumentError("$name must be in [0, 1]."))
    return nothing
end

@inline function _check_constant_bd_parameters(λ::T, μ::T, ψ::T, r::T; ρ₀::Union{Nothing,T}=nothing) where {T<:AbstractFloat}
    isfinite(λ) || throw(ArgumentError("λ must be finite."))
    isfinite(μ) || throw(ArgumentError("μ must be finite."))
    isfinite(ψ) || throw(ArgumentError("ψ must be finite."))
    λ > zero(T) || throw(ArgumentError("λ must be positive for the current constant-rate birth-death formulas."))
    μ >= zero(T) || throw(ArgumentError("μ must be non-negative."))
    ψ >= zero(T) || throw(ArgumentError("ψ must be non-negative."))
    _check_probability("r", r)
    ρ₀ === nothing || _check_probability("ρ₀", ρ₀)
    return nothing
end
