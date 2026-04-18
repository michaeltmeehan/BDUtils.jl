function _multitype_age(log::MultitypeBDEventLog, t::Real)
    return log.tmax - Float64(t)
end

function _push_multitype_segment_from_forward!(segments::Vector{MultitypeColoredSegment},
                                               log::MultitypeBDEventLog,
                                               type::Integer,
                                               start_time::Real,
                                               end_time::Real)
    start = Float64(start_time)
    stop = Float64(end_time)
    stop >= start || throw(ArgumentError("lineage segment end time precedes start time."))
    stop == start && return nothing
    push!(segments, MultitypeColoredSegment(type, _multitype_age(log, stop), _multitype_age(log, start)))
    return nothing
end

"""
    multitype_colored_tree_from_eventlog(log; serial_at_tmax=:present)

Convert a clean, fully observed `MultitypeBDEventLog` into a
`MultitypeColoredTree` for the current fully colored likelihood API.

Forward simulation times are converted to likelihood ages by `age = log.tmax -
time`, so present-day samples occur at age zero and the origin is
`log.tmax`. Typed births, anagenetic transitions, fossilized samples, and
serial samples are mapped to the corresponding colored event records. Serial
samples at exactly `log.tmax` are treated as present-day samples when
`serial_at_tmax=:present`; set `serial_at_tmax=:terminal` to treat them as
terminal serial samples at age zero.

This bridge is deliberately conservative. It supports a single initial
lineage and rejects event logs containing `MultitypeDeath` or active lineages
that survive to `tmax` without a terminal/present sample, because those cases
require pruning or hidden-history integration rather than direct conversion to
the observed colored-tree representation.
"""
function multitype_colored_tree_from_eventlog(log::MultitypeBDEventLog; serial_at_tmax::Symbol=:present)
    serial_at_tmax in (:present, :terminal) ||
        throw(ArgumentError("serial_at_tmax must be :present or :terminal."))
    validate_multitype_eventlog(log)
    length(log.initial_types) == 1 ||
        throw(ArgumentError("conversion currently supports exactly one initial lineage."))
    any(==(MultitypeDeath), log.kind) &&
        throw(ArgumentError("event logs with unobserved death events are not directly convertible to a fully colored tree."))

    active_type = Dict{Int,Int}(1 => log.initial_types[1])
    segment_start = Dict{Int,Float64}(1 => 0.0)
    segments = MultitypeColoredSegment[]
    births = MultitypeColoredBirth[]
    transitions = MultitypeColoredTransition[]
    hidden_births = MultitypeColoredHiddenBirth[]
    terminal_samples = MultitypeColoredSampling[]
    ancestral_samples = MultitypeColoredSampling[]
    present_samples = Int[]

    for i in eachindex(log.time)
        t = log.time[i]
        lineage = log.lineage[i]
        parent = log.parent[i]
        before = log.type_before[i]
        after = log.type_after[i]
        age = _multitype_age(log, t)
        kind = log.kind[i]

        if kind == MultitypeBirth
            haskey(active_type, parent) ||
                throw(ArgumentError("birth parent is not active at event time."))
            active_type[parent] == before ||
                throw(ArgumentError("birth parent type does not match active lineage type."))
            _push_multitype_segment_from_forward!(segments, log, before, segment_start[parent], t)
            segment_start[parent] = t
            active_type[lineage] = after
            segment_start[lineage] = t
            push!(births, MultitypeColoredBirth(age, before, after))
        elseif kind == MultitypeTransition
            haskey(active_type, lineage) ||
                throw(ArgumentError("transition lineage is not active at event time."))
            active_type[lineage] == before ||
                throw(ArgumentError("transition type does not match active lineage type."))
            _push_multitype_segment_from_forward!(segments, log, before, segment_start[lineage], t)
            active_type[lineage] = after
            segment_start[lineage] = t
            push!(transitions, MultitypeColoredTransition(age, before, after))
        elseif kind == MultitypeFossilizedSampling
            haskey(active_type, lineage) ||
                throw(ArgumentError("ancestral sample lineage is not active at event time."))
            active_type[lineage] == before == after ||
                throw(ArgumentError("ancestral sample type does not match active lineage type."))
            _push_multitype_segment_from_forward!(segments, log, before, segment_start[lineage], t)
            segment_start[lineage] = t
            push!(ancestral_samples, MultitypeColoredSampling(age, before))
        elseif kind == MultitypeSerialSampling
            haskey(active_type, lineage) ||
                throw(ArgumentError("serial sample lineage is not active at event time."))
            active_type[lineage] == before == after ||
                throw(ArgumentError("serial sample type does not match active lineage type."))
            _push_multitype_segment_from_forward!(segments, log, before, segment_start[lineage], t)
            if t == log.tmax && serial_at_tmax == :present
                push!(present_samples, before)
            else
                push!(terminal_samples, MultitypeColoredSampling(age, before))
            end
            delete!(active_type, lineage)
            delete!(segment_start, lineage)
        end
    end

    isempty(active_type) ||
        throw(ArgumentError("event log has active lineages at tmax without terminal or present samples."))

    tree = MultitypeColoredTree(log.tmax, log.initial_types[1];
                                segments=segments,
                                births=births,
                                transitions=transitions,
                                hidden_births=hidden_births,
                                terminal_samples=terminal_samples,
                                ancestral_samples=ancestral_samples,
                                present_samples=present_samples)
    return tree
end

function validate_multitype_colored_tree_from_eventlog(log::MultitypeBDEventLog,
                                                       pars::MultitypeBDParameters; kwargs...)
    tree = multitype_colored_tree_from_eventlog(log; kwargs...)
    validate_multitype_colored_tree(tree, pars)
    return tree
end

function _multitype_birth_children(log::MultitypeBDEventLog)
    children = Dict{Int,Vector{Int}}()
    for i in eachindex(log.time)
        log.kind[i] == MultitypeBirth || continue
        push!(get!(children, log.parent[i], Int[]), log.lineage[i])
    end
    return children
end

function _multitype_retained_lineages(log::MultitypeBDEventLog)
    children = _multitype_birth_children(log)
    sample_lineages = Set{Int}()
    for i in eachindex(log.time)
        if log.kind[i] == MultitypeFossilizedSampling || log.kind[i] == MultitypeSerialSampling
            push!(sample_lineages, log.lineage[i])
        end
    end

    retained = Dict{Int,Bool}()
    function is_retained(lineage::Int)
        haskey(retained, lineage) && return retained[lineage]
        keep = lineage in sample_lineages || any(is_retained, get(children, lineage, Int[]))
        retained[lineage] = keep
        return keep
    end

    for lineage in 1:length(log.initial_types)
        is_retained(lineage)
    end
    for child_list in values(children), child in child_list
        is_retained(child)
    end
    return retained
end

function _multitype_lineage_retained_after(log::MultitypeBDEventLog,
                                           retained::Dict{Int,Bool},
                                           lineage::Int,
                                           t::Real)
    for i in eachindex(log.time)
        log.time[i] > t || continue
        if log.lineage[i] == lineage &&
           (log.kind[i] == MultitypeFossilizedSampling || log.kind[i] == MultitypeSerialSampling)
            return true
        end
        if log.kind[i] == MultitypeBirth && log.parent[i] == lineage &&
           get(retained, log.lineage[i], false)
            return true
        end
    end
    return false
end

"""
    pruned_multitype_colored_tree_from_eventlog(log; serial_at_tmax=:present)

Extract an observed/pruned `MultitypeColoredTree` from a general
`MultitypeBDEventLog`.

Unobserved dead branches and unsampled survivors are removed. Observed
lineages are retained when they contain a serial/fossilized sample or lead to a
retained sampled descendant. Birth events are emitted when both the child side
and the continuing parent side remain observed after pruning. If only the child
side is observed, the extractor emits a `MultitypeColoredHiddenBirth`: the
retained observed lineage follows the child side and the continuing parent side
is integrated out through the no-observation factor `E_parent_type(t)`.
"""
function pruned_multitype_colored_tree_from_eventlog(log::MultitypeBDEventLog; serial_at_tmax::Symbol=:present)
    serial_at_tmax in (:present, :terminal) ||
        throw(ArgumentError("serial_at_tmax must be :present or :terminal."))
    validate_multitype_eventlog(log)
    length(log.initial_types) == 1 ||
        throw(ArgumentError("pruned conversion currently supports exactly one initial lineage."))

    retained = _multitype_retained_lineages(log)
    get(retained, 1, false) ||
        throw(ArgumentError("event log contains no observed samples reachable from the initial lineage."))

    active_type = Dict{Int,Int}(1 => log.initial_types[1])
    segment_start = Dict{Int,Float64}(1 => 0.0)
    segments = MultitypeColoredSegment[]
    births = MultitypeColoredBirth[]
    transitions = MultitypeColoredTransition[]
    hidden_births = MultitypeColoredHiddenBirth[]
    terminal_samples = MultitypeColoredSampling[]
    ancestral_samples = MultitypeColoredSampling[]
    present_samples = Int[]

    for i in eachindex(log.time)
        t = log.time[i]
        lineage = log.lineage[i]
        parent = log.parent[i]
        before = log.type_before[i]
        after = log.type_after[i]
        age = _multitype_age(log, t)
        kind = log.kind[i]

        if kind == MultitypeBirth
            child_retained = get(retained, lineage, false)
            parent_active = haskey(active_type, parent)
            parent_future_retained = parent_active && _multitype_lineage_retained_after(log, retained, parent, t)

            if child_retained && parent_future_retained
                active_type[parent] == before ||
                    throw(ArgumentError("birth parent type does not match active retained lineage type."))
                _push_multitype_segment_from_forward!(segments, log, before, segment_start[parent], t)
                segment_start[parent] = t
                active_type[lineage] = after
                segment_start[lineage] = t
                push!(births, MultitypeColoredBirth(age, before, after))
            elseif child_retained && !parent_future_retained
                parent_active ||
                    throw(ArgumentError("hidden birth parent is not active at event time."))
                active_type[parent] == before ||
                    throw(ArgumentError("hidden birth parent type does not match active retained lineage type."))
                _push_multitype_segment_from_forward!(segments, log, before, segment_start[parent], t)
                delete!(active_type, parent)
                delete!(segment_start, parent)
                active_type[lineage] = after
                segment_start[lineage] = t
                push!(hidden_births, MultitypeColoredHiddenBirth(age, before, after))
            end
        elseif kind == MultitypeTransition
            haskey(active_type, lineage) || continue
            active_type[lineage] == before ||
                throw(ArgumentError("transition type does not match active retained lineage type."))
            _push_multitype_segment_from_forward!(segments, log, before, segment_start[lineage], t)
            active_type[lineage] = after
            segment_start[lineage] = t
            push!(transitions, MultitypeColoredTransition(age, before, after))
        elseif kind == MultitypeFossilizedSampling
            haskey(active_type, lineage) || continue
            active_type[lineage] == before == after ||
                throw(ArgumentError("fossilized sample type does not match active retained lineage type."))
            _push_multitype_segment_from_forward!(segments, log, before, segment_start[lineage], t)
            if _multitype_lineage_retained_after(log, retained, lineage, t)
                segment_start[lineage] = t
                push!(ancestral_samples, MultitypeColoredSampling(age, before))
            else
                push!(terminal_samples, MultitypeColoredSampling(age, before))
                delete!(active_type, lineage)
                delete!(segment_start, lineage)
            end
        elseif kind == MultitypeSerialSampling
            haskey(active_type, lineage) || continue
            active_type[lineage] == before == after ||
                throw(ArgumentError("serial sample type does not match active retained lineage type."))
            _push_multitype_segment_from_forward!(segments, log, before, segment_start[lineage], t)
            if t == log.tmax && serial_at_tmax == :present
                push!(present_samples, before)
            else
                push!(terminal_samples, MultitypeColoredSampling(age, before))
            end
            delete!(active_type, lineage)
            delete!(segment_start, lineage)
        elseif kind == MultitypeDeath
            if haskey(active_type, lineage)
                throw(ArgumentError("retained lineage reached an unobserved death; pruning state is inconsistent."))
            end
        end
    end

    isempty(active_type) ||
        throw(ArgumentError("pruned event log has retained active lineages without terminal or present samples."))

    return MultitypeColoredTree(log.tmax, log.initial_types[1];
                                segments=segments,
                                births=births,
                                transitions=transitions,
                                hidden_births=hidden_births,
                                terminal_samples=terminal_samples,
                                ancestral_samples=ancestral_samples,
                                present_samples=present_samples)
end

function validate_pruned_multitype_colored_tree_from_eventlog(log::MultitypeBDEventLog,
                                                              pars::MultitypeBDParameters; kwargs...)
    tree = pruned_multitype_colored_tree_from_eventlog(log; kwargs...)
    validate_multitype_colored_tree(tree, pars)
    return tree
end
