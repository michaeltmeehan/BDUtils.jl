function _check_multitype_analytical_time(t::Real)
    isfinite(t) || throw(ArgumentError("times must be finite."))
    t >= 0 || throw(ArgumentError("times must be non-negative."))
    return Float64(t)
end

function _check_multitype_analytical_steps(steps_per_unit::Integer, min_steps::Integer)
    steps_per_unit >= 1 || throw(ArgumentError("steps_per_unit must be positive."))
    min_steps >= 1 || throw(ArgumentError("min_steps must be positive."))
    return nothing
end

function _multitype_transition_rates(pars::MultitypeBDParameters)
    transition = copy(pars.transition)
    for k in 1:length(pars)
        transition[k, k] = zero(eltype(transition))
    end
    return transition
end

function _multitype_E_derivative!(dE::AbstractVector{T}, E::AbstractVector{T},
                                  pars::MultitypeBDParameters{T}, transition::AbstractMatrix{T}) where {T<:AbstractFloat}
    K = length(pars)
    fill!(dE, zero(T))
    for a in 1:K
        birth_out = zero(T)
        hidden_birth = zero(T)
        transition_out = zero(T)
        hidden_transition = zero(T)
        for b in 1:K
            λab = pars.birth[a, b]
            γab = transition[a, b]
            birth_out += λab
            hidden_birth += λab * E[a] * E[b]
            transition_out += γab
            hidden_transition += γab * E[b]
        end
        total = birth_out + pars.death[a] + pars.sampling[a] + transition_out
        dE[a] = -total * E[a] + hidden_birth + pars.death[a] + hidden_transition
    end
    return dE
end

function _multitype_flow_rate!(rate::AbstractVector{T}, E::AbstractVector{T},
                               pars::MultitypeBDParameters{T}, transition::AbstractMatrix{T}) where {T<:AbstractFloat}
    K = length(pars)
    fill!(rate, zero(T))
    for a in 1:K
        birth_out = zero(T)
        hidden_birth = zero(T)
        transition_out = zero(T)
        for b in 1:K
            birth_out += pars.birth[a, b]
            hidden_birth += pars.birth[a, b] * E[b]
            transition_out += transition[a, b]
        end
        rate[a] = hidden_birth - (birth_out + pars.death[a] + pars.sampling[a] + transition_out)
    end
    return rate
end

function _rk4_multitype_E_step!(E::Vector{T}, dt::T, pars::MultitypeBDParameters{T}, transition::Matrix{T}) where {T<:AbstractFloat}
    K = length(E)
    k1 = similar(E)
    k2 = similar(E)
    k3 = similar(E)
    k4 = similar(E)
    tmp = similar(E)

    _multitype_E_derivative!(k1, E, pars, transition)
    @inbounds for i in 1:K
        tmp[i] = E[i] + dt * k1[i] / 2
    end
    _multitype_E_derivative!(k2, tmp, pars, transition)
    @inbounds for i in 1:K
        tmp[i] = E[i] + dt * k2[i] / 2
    end
    _multitype_E_derivative!(k3, tmp, pars, transition)
    @inbounds for i in 1:K
        tmp[i] = E[i] + dt * k3[i]
    end
    _multitype_E_derivative!(k4, tmp, pars, transition)

    @inbounds for i in 1:K
        E[i] += dt * (k1[i] + 2k2[i] + 2k3[i] + k4[i]) / 6
        E[i] = clamp(E[i], zero(T), one(T))
    end
    return E
end

"""
    multitype_E(t, pars; steps_per_unit=256, min_steps=16)

Evaluate the multitype no-observation probabilities `E_a(t)`.

`E_a(t)` is the probability that one type-`a` lineage alive `t` time units
before the observation horizon leaves no observed descendants. Under the
current simulator semantics, a typed birth `a => b` keeps the parent lineage
at type `a` and creates one type-`b` child; a hidden birth therefore contributes
`birth[a,b] * E_a * E_b`. An anagenetic transition `a => b` changes the lineage
type and contributes `transition[a,b] * E_b`. Sampling is always an observation,
so `sampling[a]` enters as loss in the ODE; `removal_probability[a]` only affects
what happens after an observed sample and does not enter `E_a`.
"""
function multitype_E(t::Real, pars::MultitypeBDParameters; steps_per_unit::Integer=256, min_steps::Integer=16)
    tq = _check_multitype_analytical_time(t)
    _check_multitype_analytical_steps(steps_per_unit, min_steps)
    T = eltype(pars.death)
    E = one(T) .- pars.ρ₀
    tq == 0 && return Vector{Float64}(E)

    nsteps = max(min_steps, ceil(Int, tq * steps_per_unit))
    dt = T(tq / nsteps)
    transition = _multitype_transition_rates(pars)
    for _ in 1:nsteps
        _rk4_multitype_E_step!(E, dt, pars, transition)
    end
    return Vector{Float64}(E)
end

function multitype_E_over_time(times::AbstractVector{<:Real}, pars::MultitypeBDParameters; steps_per_unit::Integer=256, min_steps::Integer=16)
    isempty(times) && return Vector{Float64}[]
    checked = [_check_multitype_analytical_time(t) for t in times]
    _check_multitype_analytical_steps(steps_per_unit, min_steps)
    return [multitype_E(t, pars; steps_per_unit=steps_per_unit, min_steps=min_steps) for t in checked]
end

function _multitype_E_and_log_flow(t::Real, pars::MultitypeBDParameters; steps_per_unit::Integer=256, min_steps::Integer=16)
    tq = _check_multitype_analytical_time(t)
    _check_multitype_analytical_steps(steps_per_unit, min_steps)
    T = eltype(pars.death)
    E = one(T) .- pars.ρ₀
    logflow = zeros(T, length(pars))
    tq == 0 && return (E=Vector{Float64}(E), logflow=Vector{Float64}(logflow))

    nsteps = max(min_steps, ceil(Int, tq * steps_per_unit))
    dt = T(tq / nsteps)
    transition = _multitype_transition_rates(pars)
    r1 = similar(E)
    r2 = similar(E)
    r3 = similar(E)
    r4 = similar(E)
    E0 = similar(E)
    Emid = similar(E)
    Eend = similar(E)

    for _ in 1:nsteps
        copyto!(E0, E)
        _multitype_flow_rate!(r1, E0, pars, transition)

        _rk4_multitype_E_step!(E, dt / 2, pars, transition)
        copyto!(Emid, E)
        _multitype_flow_rate!(r2, Emid, pars, transition)
        _multitype_flow_rate!(r3, Emid, pars, transition)

        copyto!(E, E0)
        _rk4_multitype_E_step!(E, dt, pars, transition)
        copyto!(Eend, E)
        _multitype_flow_rate!(r4, Eend, pars, transition)

        @inbounds for i in eachindex(logflow)
            logflow[i] += dt * (r1[i] + 2r2[i] + 2r3[i] + r4[i]) / 6
        end
    end
    return (E=Vector{Float64}(E), logflow=Vector{Float64}(logflow))
end

"""
    multitype_log_flow(t, pars; steps_per_unit=256, min_steps=16)

Return per-type log transport factors for a lineage segment of duration `t`.

This is the package-semantics analogue of the linear `g_{e,a}` flow with no
observed event on the segment. For a type `a` segment, the instantaneous log
rate is `sum_b birth[a,b] * E_b(t) - (sum_b birth[a,b] + death[a] +
sampling[a] + sum_{b != a} transition[a,b])`. The flow is diagonal because
observed anagenetic transitions terminate the current fully typed segment.
"""
function multitype_log_flow(t::Real, pars::MultitypeBDParameters; steps_per_unit::Integer=256, min_steps::Integer=16)
    return _multitype_E_and_log_flow(t, pars; steps_per_unit=steps_per_unit, min_steps=min_steps).logflow
end

function multitype_flow(t::Real, pars::MultitypeBDParameters; steps_per_unit::Integer=256, min_steps::Integer=16)
    return exp.(multitype_log_flow(t, pars; steps_per_unit=steps_per_unit, min_steps=min_steps))
end
