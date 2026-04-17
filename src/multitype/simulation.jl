function _push_multitype_event!(times, lineages, parents, kinds, type_before, type_after,
                                t, lineage, parent, kind, before, after)
    push!(times, t)
    push!(lineages, lineage)
    push!(parents, parent)
    push!(kinds, kind)
    push!(type_before, before)
    push!(type_after, after)
    return nothing
end

function _sample_type_weighted_lineage(rng::AbstractRNG, active::Vector{Int}, lineage_type::Vector{Int}, rates::AbstractVector)
    total = zero(eltype(rates))
    for lineage in active
        total += rates[lineage_type[lineage]]
    end
    total > 0 || return 0

    u = rand(rng) * total
    acc = zero(total)
    for lineage in active
        acc += rates[lineage_type[lineage]]
        u <= acc && return lineage
    end
    return active[end]
end

function _sample_matrix_event(rng::AbstractRNG, active::Vector{Int}, lineage_type::Vector{Int}, rates::AbstractMatrix)
    total = zero(eltype(rates))
    K = size(rates, 2)
    for lineage in active
        a = lineage_type[lineage]
        for b in 1:K
            total += rates[a, b]
        end
    end
    total > 0 || return (lineage=0, from=0, to=0)

    u = rand(rng) * total
    acc = zero(total)
    for lineage in active
        a = lineage_type[lineage]
        for b in 1:K
            acc += rates[a, b]
            u <= acc && return (lineage=lineage, from=a, to=b)
        end
    end
    lineage = active[end]
    return (lineage=lineage, from=lineage_type[lineage], to=K)
end

function _remove_active_lineage!(active::Vector{Int}, lineage::Int)
    i = findfirst(==(lineage), active)
    i === nothing && throw(ArgumentError("lineage is not active."))
    active[i] = active[end]
    pop!(active)
    return lineage
end

"""
    simulate_multitype_bd([rng], pars, tmax; initial_types=[1], apply_ρ₀=true)

Simulate a finite-type constant-rate birth-death-sampling process.

`pars.birth[a,b]` is the rate at which an active type-`a` lineage produces a
new type-`b` child lineage. `pars.transition[a,b]` is the anagenetic rate for
an active lineage changing from type `a` to type `b`; diagonal entries are
ignored. Serial sampling removes a lineage with its type-specific removal
probability, otherwise it is recorded as fossilized sampling.
"""
function simulate_multitype_bd(rng::AbstractRNG,
                               pars::MultitypeBDParameters,
                               tmax::Real;
                               initial_types::AbstractVector{<:Integer}=[1],
                               apply_ρ₀::Bool=true)
    _check_simulation_inputs(tmax, length(initial_types))
    K = length(pars)
    initial = _check_initial_types(initial_types, K)

    tstop = Float64(tmax)
    t = 0.0
    next_lineage = length(initial) + 1
    active = collect(1:length(initial))
    lineage_type = copy(initial)

    times = Float64[]
    lineages = Int[]
    parents = Int[]
    kinds = MultitypeBDEventKind[]
    type_before = Int[]
    type_after = Int[]

    transition_rates = copy(pars.transition)
    for k in 1:K
        transition_rates[k, k] = zero(eltype(transition_rates))
    end

    birth_by_type = vec(sum(pars.birth; dims=2))
    transition_by_type = vec(sum(transition_rates; dims=2))

    while !isempty(active)
        birth_rate = sum(birth_by_type[lineage_type[lineage]] for lineage in active)
        death_rate = sum(pars.death[lineage_type[lineage]] for lineage in active)
        sampling_rate = sum(pars.sampling[lineage_type[lineage]] for lineage in active)
        transition_rate = sum(transition_by_type[lineage_type[lineage]] for lineage in active)
        total_rate = birth_rate + death_rate + sampling_rate + transition_rate
        total_rate > 0 || break

        t += randexp(rng) / total_rate
        t <= tstop || break

        u = rand(rng) * total_rate
        if u <= birth_rate
            event = _sample_matrix_event(rng, active, lineage_type, pars.birth)
            child = next_lineage
            next_lineage += 1
            push!(active, child)
            push!(lineage_type, event.to)
            _push_multitype_event!(times, lineages, parents, kinds, type_before, type_after,
                                   t, child, event.lineage, MultitypeBirth, event.from, event.to)
        elseif u <= birth_rate + death_rate
            lineage = _sample_type_weighted_lineage(rng, active, lineage_type, pars.death)
            before = lineage_type[lineage]
            _remove_active_lineage!(active, lineage)
            _push_multitype_event!(times, lineages, parents, kinds, type_before, type_after,
                                   t, lineage, 0, MultitypeDeath, before, before)
        elseif u <= birth_rate + death_rate + sampling_rate
            lineage = _sample_type_weighted_lineage(rng, active, lineage_type, pars.sampling)
            before = lineage_type[lineage]
            if rand(rng) < pars.removal_probability[before]
                _remove_active_lineage!(active, lineage)
                kind = MultitypeSerialSampling
            else
                kind = MultitypeFossilizedSampling
            end
            _push_multitype_event!(times, lineages, parents, kinds, type_before, type_after,
                                   t, lineage, 0, kind, before, before)
        else
            event = _sample_matrix_event(rng, active, lineage_type, transition_rates)
            lineage_type[event.lineage] = event.to
            _push_multitype_event!(times, lineages, parents, kinds, type_before, type_after,
                                   t, event.lineage, 0, MultitypeTransition, event.from, event.to)
        end
    end

    if apply_ρ₀ && !isempty(active)
        for lineage in copy(active)
            before = lineage_type[lineage]
            if rand(rng) < pars.ρ₀[before]
                _push_multitype_event!(times, lineages, parents, kinds, type_before, type_after,
                                       tstop, lineage, 0, MultitypeSerialSampling, before, before)
            end
        end
    end

    return MultitypeBDEventLog(times, lineages, parents, kinds, type_before, type_after, initial, tstop)
end

simulate_multitype_bd(pars::MultitypeBDParameters, tmax::Real; kwargs...) =
    simulate_multitype_bd(Random.default_rng(), pars, tmax; kwargs...)
