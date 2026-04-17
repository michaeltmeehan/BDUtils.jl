@enum BDEventKind::UInt8 begin
    Birth = 1
    Death = 2
    FossilizedSampling = 3
    SerialSampling = 4
end

struct BDEventLog
    time::Vector{Float64}
    lineage::Vector{Int}
    parent::Vector{Int}
    kind::Vector{BDEventKind}
    initial_lineages::Int
    tmax::Float64
end

struct BDEventRecord
    time::Float64
    lineage::Int
    parent::Int
    kind::BDEventKind
end

Base.length(log::BDEventLog) = length(log.time)
Base.isempty(log::BDEventLog) = isempty(log.time)
Base.firstindex(::BDEventLog) = 1
Base.lastindex(log::BDEventLog) = length(log)
Base.eltype(::Type{BDEventLog}) = BDEventRecord

function Base.getindex(log::BDEventLog, i::Integer)
    return BDEventRecord(log.time[i], log.lineage[i], log.parent[i], log.kind[i])
end

function Base.iterate(log::BDEventLog, state::Int=1)
    state > length(log) && return nothing
    return (log[state], state + 1)
end

function Base.show(io::IO, log::BDEventLog)
    print(io, "BDEventLog(", length(log), " events, initial_lineages=",
          log.initial_lineages, ", tmax=", log.tmax, ")")
end

function _check_simulation_inputs(tmax::Real, initial_lineages::Integer)
    isfinite(tmax) || throw(ArgumentError("tmax must be finite."))
    tmax >= 0 || throw(ArgumentError("tmax must be non-negative."))
    initial_lineages >= 0 || throw(ArgumentError("initial_lineages must be non-negative."))
    return nothing
end

function _push_bd_event!(times, lineages, parents, kinds, t, lineage, parent, kind)
    push!(times, t)
    push!(lineages, lineage)
    push!(parents, parent)
    push!(kinds, kind)
    return nothing
end

function _remove_active!(rng::AbstractRNG, active::Vector{Int})
    i = rand(rng, eachindex(active))
    lineage = active[i]
    active[i] = active[end]
    pop!(active)
    return lineage
end

"""
    simulate_bd([rng], pars, tmax; initial_lineages=1, apply_ρ₀=true)

Simulate a constant-rate generalized birth-death-sampling process up to `tmax`.

The returned [`BDEventLog`](@ref) records only process events. Birth events add
one active lineage, death and serial-sampling events remove one active lineage,
and fossilized-sampling events leave the active population unchanged. When
`apply_ρ₀=true`, each lineage active at `tmax` is independently sampled with
probability `pars.ρ₀` and recorded as a `SerialSampling` event at `tmax`.
"""
function simulate_bd(rng::AbstractRNG,
                     pars::ConstantRateBDParameters,
                     tmax::Real;
                     initial_lineages::Integer=1,
                     apply_ρ₀::Bool=true)
    _check_simulation_inputs(tmax, initial_lineages)

    tstop = Float64(tmax)
    t = 0.0
    next_lineage = Int(initial_lineages) + 1
    active = collect(1:Int(initial_lineages))

    times = Float64[]
    lineages = Int[]
    parents = Int[]
    kinds = BDEventKind[]

    while !isempty(active)
        n = length(active)
        birth_rate = pars.λ * n
        death_rate = pars.μ * n
        sampling_rate = pars.ψ * n
        total_rate = birth_rate + death_rate + sampling_rate
        total_rate > 0 || break

        t += randexp(rng) / total_rate
        t <= tstop || break

        u = rand(rng) * total_rate
        if u <= birth_rate
            parent = rand(rng, active)
            child = next_lineage
            next_lineage += 1
            push!(active, child)
            _push_bd_event!(times, lineages, parents, kinds, t, child, parent, Birth)
        elseif u <= birth_rate + death_rate
            lineage = _remove_active!(rng, active)
            _push_bd_event!(times, lineages, parents, kinds, t, lineage, 0, Death)
        else
            if rand(rng) < pars.r
                lineage = _remove_active!(rng, active)
                _push_bd_event!(times, lineages, parents, kinds, t, lineage, 0, SerialSampling)
            else
                lineage = rand(rng, active)
                _push_bd_event!(times, lineages, parents, kinds, t, lineage, 0, FossilizedSampling)
            end
        end
    end

    if apply_ρ₀ && pars.ρ₀ > 0 && !isempty(active)
        survivors = copy(active)
        for lineage in survivors
            if rand(rng) < pars.ρ₀
                _push_bd_event!(times, lineages, parents, kinds, tstop, lineage, 0, SerialSampling)
            end
        end
    end

    return BDEventLog(times, lineages, parents, kinds, Int(initial_lineages), tstop)
end

simulate_bd(pars::ConstantRateBDParameters, tmax::Real; kwargs...) =
    simulate_bd(Random.default_rng(), pars, tmax; kwargs...)

function _check_query_time(t::Real)
    isfinite(t) || throw(ArgumentError("query times must be finite."))
    t >= 0 || throw(ArgumentError("query times must be non-negative."))
    return Float64(t)
end

function _check_reconstructed_times(log::BDEventLog, tⱼ::Real, tₖ::Real)
    tj = _check_query_time(tⱼ)
    tk = _check_query_time(tₖ)
    tj <= tk || throw(ArgumentError("reconstructed times must satisfy tⱼ <= tₖ."))
    tk <= log.tmax || throw(ArgumentError("reconstructed truncation time tₖ must be <= log.tmax."))
    return tj, tk
end

function NS_at(log::BDEventLog, t::Real)
    tq = _check_query_time(t)
    N = log.initial_lineages
    S = 0
    for i in eachindex(log.time)
        log.time[i] <= tq || break
        kind = log.kind[i]
        if kind == Birth
            N += 1
        elseif kind == Death
            N -= 1
        elseif kind == SerialSampling
            N -= 1
            S += 1
        elseif kind == FossilizedSampling
            S += 1
        end
    end
    return (N=N, S=S)
end

N_at(log::BDEventLog, t::Real) = NS_at(log, t).N
S_at(log::BDEventLog, t::Real) = NS_at(log, t).S

function NS_over_time(log::BDEventLog, times::AbstractVector{<:Real})
    return [NS_at(log, t) for t in times]
end

N_over_time(log::BDEventLog, times::AbstractVector{<:Real}) =
    [x.N for x in NS_over_time(log, times)]

S_over_time(log::BDEventLog, times::AbstractVector{<:Real}) =
    [x.S for x in NS_over_time(log, times)]

function _lineage_history(log::BDEventLog)
    born = Dict{Int,Float64}(i => 0.0 for i in 1:log.initial_lineages)
    removed = Dict{Int,Float64}()
    parent = Dict{Int,Int}(i => 0 for i in 1:log.initial_lineages)

    for i in eachindex(log.time)
        kind = log.kind[i]
        lineage = log.lineage[i]
        if kind == Birth
            born[lineage] = log.time[i]
            parent[lineage] = log.parent[i]
        elseif kind == Death || kind == SerialSampling
            removed[lineage] = log.time[i]
        elseif kind == FossilizedSampling
            haskey(born, lineage) || (born[lineage] = 0.0)
        end
    end

    return (born=born, removed=removed, parent=parent)
end

function _extant_lineages_at(history, t::Float64)
    return sort!([lineage for (lineage, born_time) in history.born
                  if born_time <= t && get(history.removed, lineage, Inf) > t])
end

"""
    extant_lineages_at(log, t)

Lineage identifiers active immediately after all events recorded at time `t`.
Death and serial-sampling events at `t` are therefore removals, while births
and fossilized samples at `t` are retained as active lineages.
"""
function extant_lineages_at(log::BDEventLog, t::Real)
    tq = _check_query_time(t)
    history = _lineage_history(log)
    return _extant_lineages_at(history, tq)
end

function _is_extant_lineage_at(history, lineage::Int, t::Float64)
    return get(history.born, lineage, Inf) <= t && get(history.removed, lineage, Inf) > t
end

function _retained_lineage_set(log::BDEventLog, history, tⱼ::Float64, tₖ::Float64)
    retained = Set{Int}()

    for i in eachindex(log.time)
        tⱼ < log.time[i] <= tₖ || continue
        kind = log.kind[i]
        (kind == SerialSampling || kind == FossilizedSampling) || continue

        lineage = log.lineage[i]
        while lineage != 0
            if _is_extant_lineage_at(history, lineage, tⱼ)
                push!(retained, lineage)
                break
            end
            lineage = get(history.parent, lineage, 0)
        end
    end

    return retained
end

"""
    retained_lineages_at(log, tⱼ, tₖ=log.tmax)

Lineage identifiers active immediately after events at `tⱼ` that have at least
one sampled descendant in `(tⱼ, tₖ]`. Descendant marking is computed directly
from `BDEventLog` parent links and includes serial, fossilized, and terminal
`ρ₀` samples when they fall inside the strict-future window.
"""
function retained_lineages_at(log::BDEventLog, tⱼ::Real, tₖ::Real=log.tmax)
    tj, tk = _check_reconstructed_times(log, tⱼ, tₖ)
    history = _lineage_history(log)
    active = _extant_lineages_at(history, tj)
    retained = _retained_lineage_set(log, history, tj, tk)
    return [lineage for lineage in active if lineage in retained]
end

"""
    A_at(log, tⱼ, tₖ=log.tmax)

Number of reconstructed lineages active at `tⱼ`, i.e. extant lineages at `tⱼ`
with a sampled descendant in `(tⱼ, tₖ]`.
"""
A_at(log::BDEventLog, tⱼ::Real, tₖ::Real=log.tmax) = length(retained_lineages_at(log, tⱼ, tₖ))

function A_over_time(log::BDEventLog, times::AbstractVector{<:Real}; tₖ::Real=log.tmax, tk::Real=tₖ)
    return [A_at(log, t, tk) for t in times]
end

function joint_counts_NS(logs, t::Real)
    counts = Dict{Tuple{Int,Int},Int}()
    for log in logs
        ns = NS_at(log, t)
        key = (ns.N, ns.S)
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

function joint_counts_NS(logs, times::AbstractVector{<:Real})
    return [joint_counts_NS(logs, t) for t in times]
end

function joint_pmf_NS(counts::Dict{Tuple{Int,Int},Int})
    total = sum(values(counts))
    total > 0 || throw(ArgumentError("cannot normalize an empty joint count table."))
    return Dict(k => v / total for (k, v) in counts)
end

joint_pmf_NS(logs, t::Real) = joint_pmf_NS(joint_counts_NS(logs, t))
joint_pmf_NS(logs, times::AbstractVector{<:Real}) = [joint_pmf_NS(joint_counts_NS(logs, t)) for t in times]

function reconstructed_counts_A(logs, tⱼ::Real, tₖ_by_log=log -> log.tmax)
    counts = Dict{Int,Int}()
    for log in logs
        a = A_at(log, tⱼ, tₖ_by_log(log))
        counts[a] = get(counts, a, 0) + 1
    end
    return counts
end

reconstructed_counts_A(logs, tⱼ::Real, tₖ::Real) =
    reconstructed_counts_A(logs, tⱼ, Returns(tₖ))

function reconstructed_counts_A(logs, times::AbstractVector{<:Real}; tₖ_by_log=log -> log.tmax, tk_by_log=tₖ_by_log)
    return [reconstructed_counts_A(logs, t, tk_by_log) for t in times]
end

function reconstructed_pmf_A(counts::Dict{Int,Int})
    total = sum(values(counts))
    total > 0 || throw(ArgumentError("cannot normalize an empty reconstructed count table."))
    return Dict(k => v / total for (k, v) in counts)
end

reconstructed_pmf_A(logs, tⱼ::Real, tₖ_by_log=log -> log.tmax) =
    reconstructed_pmf_A(reconstructed_counts_A(logs, tⱼ, tₖ_by_log))
reconstructed_pmf_A(logs, tⱼ::Real, tₖ::Real) =
    reconstructed_pmf_A(reconstructed_counts_A(logs, tⱼ, tₖ))
reconstructed_pmf_A(logs, times::AbstractVector{<:Real}; tₖ_by_log=log -> log.tmax, tk_by_log=tₖ_by_log) =
    [reconstructed_pmf_A(logs, t, tk_by_log) for t in times]

function empirical_retention_probability(logs, tⱼ::Real, tₖ_by_log=log -> log.tmax)
    total_extant = 0
    total_retained = 0
    for log in logs
        total_extant += length(extant_lineages_at(log, tⱼ))
        total_retained += A_at(log, tⱼ, tₖ_by_log(log))
    end
    total_extant > 0 || throw(ArgumentError("cannot estimate retention probability with no extant lineages."))
    return total_retained / total_extant
end

empirical_retention_probability(logs, tⱼ::Real, tₖ::Real) =
    empirical_retention_probability(logs, tⱼ, Returns(tₖ))

function reconstructed_joint_counts_AS(logs, tⱼ::Real, tₖ_by_log=log -> log.tmax)
    counts = Dict{Tuple{Int,Int},Int}()
    for log in logs
        key = (A_at(log, tⱼ, tₖ_by_log(log)), S_at(log, tⱼ))
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

reconstructed_joint_counts_AS(logs, tⱼ::Real, tₖ::Real) =
    reconstructed_joint_counts_AS(logs, tⱼ, Returns(tₖ))

function reconstructed_joint_counts_AS(logs, times::AbstractVector{<:Real}; tₖ_by_log=log -> log.tmax, tk_by_log=tₖ_by_log)
    return [reconstructed_joint_counts_AS(logs, t, tk_by_log) for t in times]
end

function reconstructed_joint_pmf_AS(counts::Dict{Tuple{Int,Int},Int})
    total = sum(values(counts))
    total > 0 || throw(ArgumentError("cannot normalize an empty reconstructed joint count table."))
    return Dict(k => v / total for (k, v) in counts)
end

reconstructed_joint_pmf_AS(logs, tⱼ::Real, tₖ_by_log=log -> log.tmax) =
    reconstructed_joint_pmf_AS(reconstructed_joint_counts_AS(logs, tⱼ, tₖ_by_log))
reconstructed_joint_pmf_AS(logs, tⱼ::Real, tₖ::Real) =
    reconstructed_joint_pmf_AS(reconstructed_joint_counts_AS(logs, tⱼ, tₖ))
reconstructed_joint_pmf_AS(logs, times::AbstractVector{<:Real}; tₖ_by_log=log -> log.tmax, tk_by_log=tₖ_by_log) =
    [reconstructed_joint_pmf_AS(logs, t, tk_by_log) for t in times]

function marginal_counts_NS(counts::Dict{Tuple{Int,Int},Int})
    n_counts = Dict{Int,Int}()
    s_counts = Dict{Int,Int}()
    for ((n, s), count) in counts
        n_counts[n] = get(n_counts, n, 0) + count
        s_counts[s] = get(s_counts, s, 0) + count
    end
    return (N=n_counts, S=s_counts)
end

function marginal_pmf_NS(counts::Dict{Tuple{Int,Int},Int})
    total = sum(values(counts))
    total > 0 || throw(ArgumentError("cannot normalize an empty joint count table."))
    marg = marginal_counts_NS(counts)
    return (N=Dict(k => v / total for (k, v) in marg.N),
            S=Dict(k => v / total for (k, v) in marg.S))
end
