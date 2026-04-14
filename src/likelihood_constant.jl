@inline function E_constant(t::T, λ::T, μ::T, ψ::T; ρ₀::T=zero(T)) where {T<:AbstractFloat}
    isfinite(t) || throw(ArgumentError("t must be finite."))
    _check_constant_bd_parameters(λ, μ, ψ, zero(T); ρ₀=ρ₀)
    t <= zero(T) && return one(T) - ρ₀

    δ = λ + μ + ψ
    disc = muladd(δ, δ, -4λ * μ)
    disc = ifelse(disc < zero(T), zero(T), disc)
    Δ = sqrt(disc)

    E_plus = (δ + Δ) / (2λ)
    E_minus = (δ - Δ) / (2λ)
    E0 = one(T) - ρ₀

    C = (E0 - E_minus) / _stabilize_denominator(E0 - E_plus)
    x = Δ * t
    em = abs(x) <= sqrt(eps(T)) ? one(T) - x + x * x / 2 : exp(-x)

    num = E_minus - C * E_plus * em
    den = one(T) - C * em
    return num / _stabilize_denominator(den)
end

@inline function E_constant(t::Real, λ::Real, μ::Real, ψ::Real; ρ₀::Real=0.0)
    T = promote_type(typeof(t), typeof(λ), typeof(μ), typeof(ψ), typeof(ρ₀), Float64)
    return E_constant(T(t), T(λ), T(μ), T(ψ); ρ₀=T(ρ₀))
end

@inline function g_constant(t::T, λ::T, μ::T, ψ::T; ρ₀::T=zero(T)) where {T<:AbstractFloat}
    isfinite(t) || throw(ArgumentError("t must be finite."))
    _check_constant_bd_parameters(λ, μ, ψ, zero(T); ρ₀=ρ₀)
    t <= zero(T) && return zero(T)

    δ = λ + μ + ψ
    disc = muladd(δ, δ, -4λ * μ)
    disc = ifelse(disc < zero(T), zero(T), disc)
    Δ = sqrt(disc)

    E_plus = (δ + Δ) / (2λ)
    E_minus = (δ - Δ) / (2λ)
    E0 = one(T) - ρ₀

    C = (E0 - E_minus) / _stabilize_denominator(E0 - E_plus)
    x = Δ * t
    em = abs(x) <= sqrt(eps(T)) ? one(T) - x + x * x / 2 : exp(-x)

    num = one(T) - C * em
    den = one(T) - C
    return -x - 2 * log(num / _stabilize_denominator(den))
end

@inline function g_constant(t::Real, λ::Real, μ::Real, ψ::Real; ρ₀::Real=0.0)
    T = promote_type(typeof(t), typeof(λ), typeof(μ), typeof(ψ), typeof(ρ₀), Float64)
    return g_constant(T(t), T(λ), T(μ), T(ψ); ρ₀=T(ρ₀))
end

@inline logaddexp(a::T, b::T) where {T<:AbstractFloat} = max(a, b) + log1p(exp(-abs(a - b)))

@inline function logaddexp(a::Real, b::Real)
    T = promote_type(typeof(a), typeof(b), Float64)
    return logaddexp(T(a), T(b))
end

"""
    bd_loglikelihood_constant(tree::TreeSim.Tree, λ, μ, ψ, r; ρ₀=0)

Compute the log likelihood for the currently supported constant-rate sampled
birth-death core.

Supported model assumptions:

- constant birth rate `λ > 0`, death rate `μ >= 0`, and sampling rate `ψ > 0`
- sampled individuals are removed with probability `r in [0, 1]`
- optional contemporaneous sampling probability `ρ₀ in [0, 1]`
- the root contribution is conditioned through `log(1 - E(T))`, where `T` is
  the maximum node time in the tree and `E` is `E_constant`

`TreeSim.validate_tree` defines structural tree validity. This likelihood adds
stricter analytical admissibility rules: the tree must be non-empty, have a
single reachable root, finite times, at least one sampled node, and contain only
`Root`, `Binary`, `SampledLeaf`, and `SampledUnary` nodes. In particular,
`TreeSim.UnsampledUnary` is structurally valid in `TreeSim.jl` but unsupported
by this analytical likelihood.
"""
function bd_loglikelihood_constant(tree::TreeSim.Tree, λ::T, μ::T, ψ::T, r::T; ρ₀::T=zero(T)) where {T<:AbstractFloat}
    _check_constant_bd_parameters(λ, μ, ψ, r; ρ₀=ρ₀)
    ψ > zero(T) || throw(ArgumentError("ψ must be positive for sampled-tree likelihood evaluation."))
    _check_constant_likelihood_tree(tree)

    Tfinal = maximum(tree.time)

    log_λ = log(λ)
    log_ψ = log(ψ)
    log_r = log(r)
    log_1mr = log1p(-r)

    E_T = clamp(E_constant(Tfinal, λ, μ, ψ; ρ₀=ρ₀), zero(T), one(T))
    ll = log1p(-E_T)

    for node in tree
        τ = Tfinal - node.time
        gτ = g_constant(τ, λ, μ, ψ; ρ₀=ρ₀)
        Eτ = clamp(E_constant(τ, λ, μ, ψ; ρ₀=ρ₀), zero(T), one(T))

        if node.kind == TreeSim.Binary
            ll += log_λ + gτ
        elseif node.kind == TreeSim.SampledLeaf
            log_term = if r == zero(T)
                log_1mr + log(Eτ)
            elseif r == one(T)
                log_r
            else
                logaddexp(log_r, log_1mr + log(Eτ))
            end
            ll += log_ψ + log_term - gτ
        elseif node.kind == TreeSim.SampledUnary
            ll += log_ψ + log_1mr
        end
    end

    return ll
end

function bd_loglikelihood_constant(tree::TreeSim.Tree, λ::Real, μ::Real, ψ::Real, r::Real; ρ₀::Real=0.0)
    T = promote_type(typeof(λ), typeof(μ), typeof(ψ), typeof(r), typeof(ρ₀), Float64)
    return bd_loglikelihood_constant(tree, T(λ), T(μ), T(ψ), T(r); ρ₀=T(ρ₀))
end

function _check_constant_likelihood_tree(tree::TreeSim.Tree)
    isempty(tree.time) && throw(ArgumentError("tree must be non-empty."))

    try
        TreeSim.validate_tree(tree; require_single_root=true, require_reachable=true)
    catch err
        throw(ArgumentError("tree is not structurally valid for likelihood evaluation: $(sprint(showerror, err))"))
    end

    all(isfinite, tree.time) || throw(ArgumentError("tree times must be finite."))

    supported = (TreeSim.Root, TreeSim.Binary, TreeSim.SampledLeaf, TreeSim.SampledUnary)
    for (i, kind) in pairs(tree.kind)
        kind in supported || throw(ArgumentError("node $i has kind $kind, which is not supported by the constant-rate analytical likelihood."))
    end

    any(k -> k == TreeSim.SampledLeaf || k == TreeSim.SampledUnary, tree.kind) ||
        throw(ArgumentError("tree must contain at least one sampled node."))

    return nothing
end
