struct MultitypeColoredSegment
    type::Int
    start_time::Float64
    end_time::Float64
end

MultitypeColoredSegment(type::Integer, start_time::Real, end_time::Real) =
    MultitypeColoredSegment(Int(type), Float64(start_time), Float64(end_time))

struct MultitypeColoredBirth
    time::Float64
    parent_type::Int
    child_type::Int
end

MultitypeColoredBirth(time::Real, parent_type::Integer, child_type::Integer) =
    MultitypeColoredBirth(Float64(time), Int(parent_type), Int(child_type))

struct MultitypeColoredTransition
    time::Float64
    from_type::Int
    to_type::Int
end

MultitypeColoredTransition(time::Real, from_type::Integer, to_type::Integer) =
    MultitypeColoredTransition(Float64(time), Int(from_type), Int(to_type))

struct MultitypeColoredSampling
    time::Float64
    type::Int
end

MultitypeColoredSampling(time::Real, type::Integer) =
    MultitypeColoredSampling(Float64(time), Int(type))

struct MultitypeColoredTree
    origin_time::Float64
    root_type::Int
    segments::Vector{MultitypeColoredSegment}
    births::Vector{MultitypeColoredBirth}
    transitions::Vector{MultitypeColoredTransition}
    terminal_samples::Vector{MultitypeColoredSampling}
    ancestral_samples::Vector{MultitypeColoredSampling}
    present_samples::Vector{Int}
end

function MultitypeColoredTree(origin_time::Real,
                              root_type::Integer;
                              segments::AbstractVector{<:MultitypeColoredSegment}=MultitypeColoredSegment[],
                              births::AbstractVector{<:MultitypeColoredBirth}=MultitypeColoredBirth[],
                              transitions::AbstractVector{<:MultitypeColoredTransition}=MultitypeColoredTransition[],
                              terminal_samples::AbstractVector{<:MultitypeColoredSampling}=MultitypeColoredSampling[],
                              ancestral_samples::AbstractVector{<:MultitypeColoredSampling}=MultitypeColoredSampling[],
                              present_samples::AbstractVector{<:Integer}=Int[])
    return MultitypeColoredTree(Float64(origin_time), Int(root_type),
                                collect(segments), collect(births), collect(transitions),
                                collect(terminal_samples), collect(ancestral_samples),
                                Int.(present_samples))
end

function _check_colored_type(type::Integer, K::Integer, name::AbstractString)
    1 <= type <= K || throw(ArgumentError("$name must be in 1:K."))
    return nothing
end

function _check_colored_time(t::Real, origin_time::Real, name::AbstractString)
    isfinite(t) || throw(ArgumentError("$name must be finite."))
    0 <= t <= origin_time || throw(ArgumentError("$name must lie in [0, origin_time]."))
    return nothing
end

function validate_multitype_colored_tree(tree::MultitypeColoredTree, pars::MultitypeBDParameters)
    K = length(pars)
    isfinite(tree.origin_time) || throw(ArgumentError("origin_time must be finite."))
    tree.origin_time >= 0 || throw(ArgumentError("origin_time must be non-negative."))
    _check_colored_type(tree.root_type, K, "root_type")

    for segment in tree.segments
        _check_colored_type(segment.type, K, "segment type")
        _check_colored_time(segment.start_time, tree.origin_time, "segment start_time")
        _check_colored_time(segment.end_time, tree.origin_time, "segment end_time")
        segment.start_time <= segment.end_time ||
            throw(ArgumentError("segment times must satisfy start_time <= end_time."))
    end
    any(segment -> segment.type == tree.root_type && segment.end_time == tree.origin_time, tree.segments) ||
        throw(ArgumentError("at least one segment with root_type must end at origin_time."))

    for event in tree.births
        _check_colored_time(event.time, tree.origin_time, "birth time")
        _check_colored_type(event.parent_type, K, "birth parent_type")
        _check_colored_type(event.child_type, K, "birth child_type")
    end
    for event in tree.transitions
        _check_colored_time(event.time, tree.origin_time, "transition time")
        _check_colored_type(event.from_type, K, "transition from_type")
        _check_colored_type(event.to_type, K, "transition to_type")
        event.from_type != event.to_type || throw(ArgumentError("transition types must differ."))
    end
    for event in tree.terminal_samples
        _check_colored_time(event.time, tree.origin_time, "terminal sample time")
        _check_colored_type(event.type, K, "terminal sample type")
    end
    for event in tree.ancestral_samples
        _check_colored_time(event.time, tree.origin_time, "ancestral sample time")
        _check_colored_type(event.type, K, "ancestral sample type")
    end
    for type in tree.present_samples
        _check_colored_type(type, K, "present sample type")
    end
    return true
end

@inline function _log_positive_rate(x::Real, label::AbstractString)
    x > 0 || throw(ArgumentError("$label must be positive for this colored tree."))
    return log(x)
end

"""
    multitype_colored_loglikelihood(tree, pars; steps_per_unit=256, min_steps=16)

Log-likelihood for a manually specified fully colored multitype tree.

The tree representation is intentionally critical-event based. `segments`
encode fully typed no-observed-event intervals and contribute the diagonal
`multitype_log_flow` transport over `[start_time,end_time]`. Observed typed
births contribute `birth[parent_type, child_type]`; observed anagenetic
transitions contribute `transition[from_type, to_type]`; present-day samples
contribute `ρ₀[type]`. A terminal serial sample at age `t` contributes
`sampling[type] * (removal_probability[type] + (1-removal_probability[type]) *
E_type(t))`, because a non-removing sampled lineage must have no later observed
descendants to be terminal. An ancestral sample contributes
`sampling[type] * (1-removal_probability[type])`.
"""
function multitype_colored_loglikelihood(tree::MultitypeColoredTree,
                                         pars::MultitypeBDParameters;
                                         steps_per_unit::Integer=256,
                                         min_steps::Integer=16)
    validate_multitype_colored_tree(tree, pars)
    _check_multitype_analytical_steps(steps_per_unit, min_steps)

    times = Float64[0.0, tree.origin_time]
    append!(times, (segment.start_time for segment in tree.segments))
    append!(times, (segment.end_time for segment in tree.segments))
    append!(times, (event.time for event in tree.terminal_samples))
    unique_times = sort!(unique(times))

    logflows = Dict{Float64,Vector{Float64}}()
    Evals = Dict{Float64,Vector{Float64}}()
    for t in unique_times
        result = _multitype_E_and_log_flow(t, pars; steps_per_unit=steps_per_unit, min_steps=min_steps)
        logflows[t] = result.logflow
        Evals[t] = result.E
    end

    ll = 0.0
    for segment in tree.segments
        ll += logflows[segment.end_time][segment.type] - logflows[segment.start_time][segment.type]
    end
    for event in tree.births
        ll += _log_positive_rate(pars.birth[event.parent_type, event.child_type], "birth rate")
    end
    for event in tree.transitions
        ll += _log_positive_rate(pars.transition[event.from_type, event.to_type], "transition rate")
    end
    for event in tree.terminal_samples
        type = event.type
        E = Evals[event.time][type]
        factor = pars.sampling[type] * (pars.removal_probability[type] +
                 (1 - pars.removal_probability[type]) * E)
        ll += _log_positive_rate(factor, "terminal sampling factor")
    end
    for event in tree.ancestral_samples
        type = event.type
        factor = pars.sampling[type] * (1 - pars.removal_probability[type])
        ll += _log_positive_rate(factor, "ancestral sampling factor")
    end
    for type in tree.present_samples
        ll += _log_positive_rate(pars.ρ₀[type], "present sampling probability")
    end
    return ll
end

function multitype_colored_likelihood(tree::MultitypeColoredTree,
                                      pars::MultitypeBDParameters; kwargs...)
    return exp(multitype_colored_loglikelihood(tree, pars; kwargs...))
end
