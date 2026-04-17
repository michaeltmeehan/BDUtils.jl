function multitype_NS_at(log::MultitypeBDEventLog, t::Real)
    tq = _check_query_time(t)
    K = _multitype_log_K(log)
    N = zeros(Int, K)
    S = zeros(Int, K)
    for typ in log.initial_types
        N[typ] += 1
    end

    for i in eachindex(log.time)
        log.time[i] <= tq || break
        before = log.type_before[i]
        after = log.type_after[i]
        kind = log.kind[i]
        if kind == MultitypeBirth
            N[after] += 1
        elseif kind == MultitypeDeath
            N[before] -= 1
        elseif kind == MultitypeSerialSampling
            N[before] -= 1
            S[before] += 1
        elseif kind == MultitypeFossilizedSampling
            S[before] += 1
        elseif kind == MultitypeTransition
            N[before] -= 1
            N[after] += 1
        end
    end
    return (N=N, S=S)
end

multitype_N_at(log::MultitypeBDEventLog, t::Real) = multitype_NS_at(log, t).N
multitype_S_at(log::MultitypeBDEventLog, t::Real) = multitype_NS_at(log, t).S

function multitype_NS_over_time(log::MultitypeBDEventLog, times::AbstractVector{<:Real})
    return [multitype_NS_at(log, t) for t in times]
end

multitype_N_over_time(log::MultitypeBDEventLog, times::AbstractVector{<:Real}) =
    [x.N for x in multitype_NS_over_time(log, times)]

multitype_S_over_time(log::MultitypeBDEventLog, times::AbstractVector{<:Real}) =
    [x.S for x in multitype_NS_over_time(log, times)]

function multitype_mean_N(logs::AbstractVector{<:MultitypeBDEventLog}, t::Real)
    isempty(logs) && throw(ArgumentError("logs must be non-empty."))
    K = maximum(_multitype_log_K(log) for log in logs)
    total = zeros(Float64, K)
    for log in logs
        N = multitype_N_at(log, t)
        total[1:length(N)] .+= N
    end
    return total ./ length(logs)
end

function validate_multitype_eventlog(log::MultitypeBDEventLog)
    n = length(log.time)
    lengths = (length(log.lineage), length(log.parent), length(log.kind),
               length(log.type_before), length(log.type_after))
    all(==(n), lengths) || throw(ArgumentError("event log arrays must have equal length."))
    issorted(log.time) || throw(ArgumentError("event times must be sorted."))
    all(t -> isfinite(t) && 0 <= t <= log.tmax, log.time) ||
        throw(ArgumentError("event times must be finite and inside [0, tmax]."))

    active = Set(1:length(log.initial_types))
    lineage_type = Dict(i => log.initial_types[i] for i in eachindex(log.initial_types))
    for i in eachindex(log.time)
        lineage = log.lineage[i]
        before = log.type_before[i]
        after = log.type_after[i]
        kind = log.kind[i]
        if kind == MultitypeBirth
            (log.parent[i] in active) || throw(ArgumentError("birth parent must be active."))
            lineage in active && throw(ArgumentError("birth child lineage already active."))
            lineage_type[log.parent[i]] == before || throw(ArgumentError("birth parent type mismatch."))
            push!(active, lineage)
            lineage_type[lineage] = after
        elseif kind == MultitypeDeath || kind == MultitypeSerialSampling
            lineage in active || throw(ArgumentError("removal lineage must be active."))
            lineage_type[lineage] == before == after || throw(ArgumentError("removal type mismatch."))
            delete!(active, lineage)
        elseif kind == MultitypeFossilizedSampling
            lineage in active || throw(ArgumentError("fossilized sampling lineage must be active."))
            lineage_type[lineage] == before == after || throw(ArgumentError("sampling type mismatch."))
        elseif kind == MultitypeTransition
            lineage in active || throw(ArgumentError("transition lineage must be active."))
            lineage_type[lineage] == before || throw(ArgumentError("transition type mismatch."))
            lineage_type[lineage] = after
        end
    end
    return true
end
