@enum MultitypeBDEventKind::UInt8 begin
    MultitypeBirth = 1
    MultitypeDeath = 2
    MultitypeFossilizedSampling = 3
    MultitypeSerialSampling = 4
    MultitypeTransition = 5
end

struct MultitypeBDParameters{T<:AbstractFloat}
    birth::Matrix{T}
    death::Vector{T}
    sampling::Vector{T}
    removal_probability::Vector{T}
    transition::Matrix{T}
    ρ₀::Vector{T}

    function MultitypeBDParameters{T}(birth::AbstractMatrix{T},
                                      death::AbstractVector{T},
                                      sampling::AbstractVector{T},
                                      removal_probability::AbstractVector{T},
                                      transition::AbstractMatrix{T},
                                      ρ₀::AbstractVector{T}=zeros(T, length(death))) where {T<:AbstractFloat}
        _check_multitype_parameters(birth, death, sampling, removal_probability, transition, ρ₀)
        return new{T}(Matrix{T}(birth), Vector{T}(death), Vector{T}(sampling),
                      Vector{T}(removal_probability), Matrix{T}(transition), Vector{T}(ρ₀))
    end
end

function MultitypeBDParameters(birth::AbstractMatrix{<:Real},
                               death::AbstractVector{<:Real},
                               sampling::AbstractVector{<:Real},
                               removal_probability::AbstractVector{<:Real},
                               transition::AbstractMatrix{<:Real},
                               ρ₀::AbstractVector{<:Real}=zeros(length(death)))
    T = promote_type(eltype(birth), eltype(death), eltype(sampling),
                     eltype(removal_probability), eltype(transition), eltype(ρ₀), Float64)
    return MultitypeBDParameters{T}(T.(birth), T.(death), T.(sampling),
                                   T.(removal_probability), T.(transition), T.(ρ₀))
end

Base.length(pars::MultitypeBDParameters) = length(pars.death)

function Base.show(io::IO, pars::MultitypeBDParameters)
    print(io, "MultitypeBDParameters(K=", length(pars), ")")
end

struct MultitypeBDEventLog
    time::Vector{Float64}
    lineage::Vector{Int}
    parent::Vector{Int}
    kind::Vector{MultitypeBDEventKind}
    type_before::Vector{Int}
    type_after::Vector{Int}
    initial_types::Vector{Int}
    tmax::Float64
end

struct MultitypeBDEventRecord
    time::Float64
    lineage::Int
    parent::Int
    kind::MultitypeBDEventKind
    type_before::Int
    type_after::Int
end

Base.length(log::MultitypeBDEventLog) = length(log.time)
Base.isempty(log::MultitypeBDEventLog) = isempty(log.time)
Base.firstindex(::MultitypeBDEventLog) = 1
Base.lastindex(log::MultitypeBDEventLog) = length(log)
Base.eltype(::Type{MultitypeBDEventLog}) = MultitypeBDEventRecord

function Base.getindex(log::MultitypeBDEventLog, i::Integer)
    return MultitypeBDEventRecord(log.time[i], log.lineage[i], log.parent[i], log.kind[i],
                                  log.type_before[i], log.type_after[i])
end

function Base.iterate(log::MultitypeBDEventLog, state::Int=1)
    state > length(log) && return nothing
    return (log[state], state + 1)
end

function Base.show(io::IO, log::MultitypeBDEventLog)
    print(io, "MultitypeBDEventLog(", length(log), " events, K=",
          _multitype_log_K(log), ", initial_lineages=",
          length(log.initial_types), ", tmax=", log.tmax, ")")
end

function _multitype_log_K(log::MultitypeBDEventLog)
    return max(maximum(log.initial_types; init=0),
               maximum(log.type_before; init=0),
               maximum(log.type_after; init=0))
end

function _check_multitype_parameters(birth, death, sampling, removal_probability, transition, ρ₀)
    K = length(death)
    K >= 1 || throw(ArgumentError("at least one type is required."))
    size(birth) == (K, K) || throw(ArgumentError("birth must be a K x K matrix."))
    size(transition) == (K, K) || throw(ArgumentError("transition must be a K x K matrix."))
    length(sampling) == K || throw(ArgumentError("sampling must have length K."))
    length(removal_probability) == K || throw(ArgumentError("removal_probability must have length K."))
    length(ρ₀) == K || throw(ArgumentError("ρ₀ must have length K."))

    for x in birth
        isfinite(x) || throw(ArgumentError("birth rates must be finite."))
        x >= zero(x) || throw(ArgumentError("birth rates must be non-negative."))
    end
    for x in death
        isfinite(x) || throw(ArgumentError("death rates must be finite."))
        x >= zero(x) || throw(ArgumentError("death rates must be non-negative."))
    end
    for x in sampling
        isfinite(x) || throw(ArgumentError("sampling rates must be finite."))
        x >= zero(x) || throw(ArgumentError("sampling rates must be non-negative."))
    end
    for x in transition
        isfinite(x) || throw(ArgumentError("transition rates must be finite."))
        x >= zero(x) || throw(ArgumentError("transition rates must be non-negative."))
    end
    for x in removal_probability
        _check_probability("removal_probability", x)
    end
    for x in ρ₀
        _check_probability("ρ₀", x)
    end
    return nothing
end

function _check_initial_types(initial_types::AbstractVector{<:Integer}, K::Integer)
    all(t -> 1 <= t <= K, initial_types) || throw(ArgumentError("initial_types must be in 1:K."))
    return Vector{Int}(initial_types)
end
