function _check_series_order(smax::Integer)
    smax >= 0 || throw(ArgumentError("smax must be non-negative."))
    return nothing
end

function _check_count(name::AbstractString, x::Integer)
    x >= 0 || throw(ArgumentError("$name must be non-negative."))
    return nothing
end

function _series_constant(x::T, smax::Integer) where {T<:AbstractFloat}
    out = zeros(T, smax + 1)
    out[1] = x
    return out
end

function _series_linear(c0::T, c1::T, smax::Integer) where {T<:AbstractFloat}
    out = _series_constant(c0, smax)
    smax >= 1 && (out[2] = c1)
    return out
end

function _series_add(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    return a .+ b
end

function _series_sub(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    return a .- b
end

function _series_scale(a::AbstractVector{T}, c::T) where {T<:AbstractFloat}
    return c .* a
end

function _series_mul(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(a)
    out = zeros(T, n)
    @inbounds for i in 1:n
        acc = zero(T)
        for k in 1:i
            acc += a[k] * b[i - k + 1]
        end
        out[i] = acc
    end
    return out
end

function _series_inv(a::AbstractVector{T}) where {T<:AbstractFloat}
    iszero(a[1]) && throw(ArgumentError("cannot invert a power series with zero constant term."))
    n = length(a)
    out = zeros(T, n)
    out[1] = inv(a[1])
    @inbounds for i in 2:n
        acc = zero(T)
        for k in 2:i
            acc += a[k] * out[i - k + 1]
        end
        out[i] = -acc / a[1]
    end
    return out
end

function _series_div(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    return _series_mul(a, _series_inv(b))
end

function _series_sqrt(a::AbstractVector{T}) where {T<:AbstractFloat}
    a[1] >= zero(T) || throw(ArgumentError("cannot take real square root of a power series with negative constant term."))
    n = length(a)
    out = zeros(T, n)
    out[1] = sqrt(a[1])
    iszero(out[1]) && n > 1 && throw(ArgumentError("series square root with zero constant term is not supported."))
    @inbounds for i in 2:n
        acc = zero(T)
        for k in 2:(i - 1)
            acc += out[k] * out[i - k + 1]
        end
        out[i] = (a[i] - acc) / (2out[1])
    end
    return out
end

function _series_exp(a::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(a)
    out = zeros(T, n)
    out[1] = exp(a[1])
    @inbounds for i in 2:n
        m = i - 1
        acc = zero(T)
        for k in 1:m
            acc += k * a[k + 1] * out[m - k + 1]
        end
        out[i] = acc / m
    end
    return out
end

function _constant_rate_series(smax::Integer, tᵢ::T, tⱼ::T, pars::ConstantRateBDParameters{T}) where {T<:AbstractFloat}
    _check_series_order(smax)
    τ = tⱼ - tᵢ
    τ < zero(T) && throw(ArgumentError("tⱼ must be greater than or equal to tᵢ."))

    one_series = _series_constant(one(T), smax)
    zero_series = zeros(T, smax + 1)
    a = _series_linear(pars.μ, pars.r * pars.ψ, smax)
    b = _series_linear(-(pars.λ + pars.μ + pars.ψ), (one(T) - pars.r) * pars.ψ, smax)
    disc = _series_sub(_series_mul(b, b), _series_scale(a, 4pars.λ))
    Δ = _series_sqrt(disc)

    exp_term = _series_exp(_series_scale(Δ, -τ))
    numerator = _series_scale(_series_sub(one_series, exp_term), 2pars.λ)
    denominator = _series_add(_series_sub(Δ, b), _series_mul(_series_add(Δ, b), exp_term))
    γs = τ == zero(T) ? zero_series : _series_div(numerator, denominator)
    αs = _series_scale(_series_mul(a, γs), inv(pars.λ))
    βs = _series_add(one_series,
        _series_add(
            _series_scale(_series_mul(b, γs), inv(pars.λ)),
            _series_scale(_series_mul(a, _series_mul(γs, γs)), inv(pars.λ)),
        ),
    )
    return αs, βs, γs
end

function constant_rate_pgf_series(smax::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    T = promote_type(typeof(tᵢ), typeof(tⱼ), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{T}(T(pars.λ), T(pars.μ), T(pars.ψ), T(pars.r), T(pars.ρ₀))
    return _constant_rate_series(smax, T(tᵢ), T(tⱼ), pT)
end

function constant_rate_pgf_series(smax::Integer, tᵢ::Real, tⱼ::Real, λ::Real, μ::Real, ψ::Real, r::Real)
    return constant_rate_pgf_series(smax, tᵢ, tⱼ, ConstantRateBDParameters(λ, μ, ψ, r))
end

function _s_marginal_series(smax::Integer, tᵢ::Real, tⱼ::Real, pars::ConstantRateBDParameters)
    _check_series_order(smax)
    αs, βs, γs = constant_rate_pgf_series(smax, tᵢ, tⱼ, pars)
    return _pgf_sampling_marginal_series(αs, βs, γs)
end

function _pgf_sampling_marginal_series(αs::AbstractVector{T}, βs::AbstractVector{T}, γs::AbstractVector{T}) where {T<:AbstractFloat}
    smax = length(αs) - 1
    return _series_add(αs, _series_div(βs, _series_sub(_series_constant(one(T), smax), γs)))
end
