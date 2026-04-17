@inline function _promote_tree_stat_inputs(t₀::Real, T::Real, pars::ConstantRateBDParameters)
    FT = promote_type(typeof(t₀), typeof(T), typeof(pars.λ), Float64)
    pT = ConstantRateBDParameters{FT}(FT(pars.λ), FT(pars.μ), FT(pars.ψ), FT(pars.r), FT(pars.ρ₀))
    t0T = FT(t₀)
    TT = FT(T)
    t0T <= TT || throw(ArgumentError("times must satisfy t₀ <= T."))
    return t0T, TT, pT
end

function _quad_simpson(f, a::T, b::T; n::Integer=512) where {T<:AbstractFloat}
    n > 0 || throw(ArgumentError("n must be positive."))
    a == b && return zero(T)
    n_even = iseven(n) ? n : n + 1
    h = (b - a) / n_even
    acc = f(a) + f(b)
    @inbounds for i in 1:(n_even - 1)
        x = a + i * h
        acc += (isodd(i) ? 4 : 2) * f(x)
    end
    return acc * h / 3
end

"""
    reconstructed_survival_kernel(t, s, T, pars; n=512)

No-event survival kernel `𝒮(t,s)=exp(-∫ₜˢ (b(u)+d(u))du)` for one
constant-rate reconstructed lineage with sampling horizon `T`.
"""
function reconstructed_survival_kernel(t::Real, s::Real, T::Real, pars::ConstantRateBDParameters; n::Integer=512)
    tT, sT, pT = _promote_tree_stat_inputs(t, s, pars)
    _, TT, _ = _promote_tree_stat_inputs(s, T, pT)
    sT <= TT || throw(ArgumentError("times must satisfy t <= s <= T."))
    rate(u) = transformed_birth_rate(u, TT, pT) + transformed_death_rate(u, TT, pT)
    return exp(-_quad_simpson(rate, tT, sT; n))
end

"""
    reconstructed_effective_rates(t, T, pars)

Return `(b, d, R)` for the reconstructed process at time `t` with horizon `T`,
where `b` is the effective split rate and `d` is the tip-termination rate.
"""
function reconstructed_effective_rates(t::Real, T::Real, pars::ConstantRateBDParameters)
    tT, TT, pT = _promote_tree_stat_inputs(t, T, pars)
    b = transformed_birth_rate(tT, TT, pT)
    d = transformed_death_rate(tT, TT, pT)
    return (b=b, d=d, R=b + d)
end

"""
    reconstructed_y(t, T, pars)

Sample-retention probability `y(t)=1-p(t,T)` for one lineage over `(t,T]`.
"""
function reconstructed_y(t::Real, T::Real, pars::ConstantRateBDParameters)
    tT, TT, pT = _promote_tree_stat_inputs(t, T, pars)
    return one(typeof(tT)) - unsampled_probability(tT, TT, pT)
end

_reconstructed_split_rate(t::Real, T::Real, pars::ConstantRateBDParameters) =
    pars.λ * reconstructed_y(t, T, pars)

"""
    reconstructed_mean_lineages(t, t₀, T, pars)

Expected number `m(t)` of reconstructed lineages at `t`, starting from one
reconstructed lineage at `t₀`, using the constant-rate closed form from the
reconstructed-tree statistics derivation.
"""
function reconstructed_mean_lineages(t::Real, t₀::Real, T::Real, pars::ConstantRateBDParameters)
    t0T, TT, pT = _promote_tree_stat_inputs(t₀, T, pars)
    tT = typeof(t0T)(t)
    t0T <= tT <= TT || throw(ArgumentError("times must satisfy t₀ <= t <= T."))
    y0 = reconstructed_y(t0T, TT, pT)
    yt = reconstructed_y(tT, TT, pT)
    ρ = pT.λ - pT.μ - pT.r * pT.ψ
    return exp(ρ * (tT - t0T)) * yt / y0
end

"""
    reconstructed_internal_branch_density(ell, s, T, pars)

Conditional density that a reconstructed branch born at `s` has length `ell`
and ends at a split, under the constant-rate closed form.
"""
function reconstructed_internal_branch_density(ell::Real, s::Real, T::Real, pars::ConstantRateBDParameters)
    sT, TT, pT = _promote_tree_stat_inputs(s, T, pars)
    ℓ = typeof(sT)(ell)
    zero(ℓ) <= ℓ <= TT - sT || throw(ArgumentError("ell must satisfy 0 <= ell <= T-s."))
    _, _, Δ = bd_coefficients(zero(typeof(sT)), pT)
    κ = Δ - pT.ψ * (one(typeof(sT)) - pT.r)
    return pT.λ * reconstructed_y(sT, TT, pT) * exp(-κ * ℓ)
end

"""
    reconstructed_external_branch_density(ell, s, T, pars)

Conditional density that a reconstructed branch born at `s` has length `ell`
and ends as a terminating sampled tip, under the constant-rate reconstructed
process.
"""
function reconstructed_external_branch_density(ell::Real, s::Real, T::Real, pars::ConstantRateBDParameters)
    sT, TT, pT = _promote_tree_stat_inputs(s, T, pars)
    ℓ = typeof(sT)(ell)
    zero(ℓ) <= ℓ <= TT - sT || throw(ArgumentError("ell must satisfy 0 <= ell <= T-s."))
    _, _, Δ = bd_coefficients(zero(typeof(sT)), pT)
    κ = Δ - pT.ψ * (one(typeof(sT)) - pT.r)
    y_start = reconstructed_y(sT, TT, pT)
    y_end = reconstructed_y(sT + ℓ, TT, pT)
    y_end <= sqrt(eps(typeof(sT))) && return zero(typeof(sT))
    d_end = pT.ψ / y_end - pT.ψ * (one(typeof(sT)) - pT.r)
    return exp(-κ * ℓ) * y_start * d_end / y_end
end

"""
    reconstructed_node_depth_intensity(x, t₀, T, pars)

Expected split-count density at node depth `x=T-s` for a reconstructed tree
started from one reconstructed lineage at `t₀`.
"""
function reconstructed_node_depth_intensity(x::Real, t₀::Real, T::Real, pars::ConstantRateBDParameters)
    t0T, TT, pT = _promote_tree_stat_inputs(t₀, T, pars)
    xT = typeof(t0T)(x)
    zero(xT) <= xT <= TT - t0T || throw(ArgumentError("x must satisfy 0 <= x <= T-t₀."))
    s = TT - xT
    return reconstructed_mean_lineages(s, t0T, TT, pT) * _reconstructed_split_rate(s, TT, pT)
end

function reconstructed_node_depth_density(x::Real, t₀::Real, T::Real, pars::ConstantRateBDParameters; n::Integer=512)
    t0T, TT, pT = _promote_tree_stat_inputs(t₀, T, pars)
    τ = TT - t0T
    normalizer = _quad_simpson(u -> reconstructed_node_depth_intensity(u, t0T, TT, pT), zero(t0T), τ; n)
    normalizer > zero(normalizer) || return NaN
    return reconstructed_node_depth_intensity(x, t0T, TT, pT) / normalizer
end

"""
    reconstructed_one_tip_probability(t, T, pars)

Probability `q(t)` that a subtree from one reconstructed lineage at `t`
contains exactly one terminating sampled tip by horizon `T`.
"""
function reconstructed_one_tip_probability(t::Real, T::Real, pars::ConstantRateBDParameters)
    tT, TT, pT = _promote_tree_stat_inputs(t, T, pars)
    τ = TT - tT
    τ == zero(τ) && return one(τ)
    _, _, Δ = bd_coefficients(zero(typeof(tT)), pT)
    κ = Δ - pT.ψ * (one(typeof(tT)) - pT.r)
    y = reconstructed_y(tT, TT, pT)
    if abs(κ) <= sqrt(eps(typeof(tT)))
        return one(typeof(tT)) - pT.λ * y * τ
    end
    return one(typeof(tT)) - pT.λ * y / κ * (-expm1(-κ * τ))
end

"""
    expected_reconstructed_cherries(t₀, T, pars; n=1024)

Expected number of cherries in the reconstructed tree started from one
reconstructed lineage at `t₀`, using the one-dimensional constant-rate
integral in the TeX derivation.
"""
function expected_reconstructed_cherries(t₀::Real, T::Real, pars::ConstantRateBDParameters; n::Integer=1024)
    t0T, TT, pT = _promote_tree_stat_inputs(t₀, T, pars)
    integrand(s) = begin
        q = reconstructed_one_tip_probability(s, TT, pT)
        reconstructed_mean_lineages(s, t0T, TT, pT) * _reconstructed_split_rate(s, TT, pT) * q * q
    end
    return _quad_simpson(integrand, t0T, TT; n)
end

function reconstructed_branch_length_intensity(kind::Symbol, ell::Real, t₀::Real, T::Real, pars::ConstantRateBDParameters; n::Integer=512)
    kind in (:internal, :external) || throw(ArgumentError("kind must be :internal or :external."))
    t0T, TT, pT = _promote_tree_stat_inputs(t₀, T, pars)
    ℓ = typeof(t0T)(ell)
    zero(ℓ) <= ℓ <= TT - t0T || throw(ArgumentError("ell must satisfy 0 <= ell <= T-t₀."))
    density = kind == :internal ? reconstructed_internal_branch_density : reconstructed_external_branch_density
    stem = density(ℓ, t0T, TT, pT)
    upper = TT - ℓ
    born = _quad_simpson(s -> 2 * reconstructed_mean_lineages(s, t0T, TT, pT) *
                              _reconstructed_split_rate(s, TT, pT) *
                              density(ℓ, s, TT, pT),
                         t0T, upper; n)
    return stem + born
end

function reconstructed_tree_stat_counts(tree::TreeSim.Tree)
    isempty(tree) && return (internal_branches=0, external_branches=0, sampled_unary_branches=0, node_count=0, cherries=0)
    TreeSim.validate_tree(tree; require_single_root=true, require_reachable=true)
    internal = 0
    external = 0
    sampled_unary = 0
    nodes = 0
    cherries = 0
    for i in eachindex(tree)
        tree.kind[i] == TreeSim.Binary && (nodes += 1)
        l = tree.left[i]
        r = tree.right[i]
        if l != 0 && r != 0 && TreeSim.isleaf(tree, l) && TreeSim.isleaf(tree, r)
            cherries += 1
        end
        if tree.parent[i] != 0
            if tree.kind[i] == TreeSim.Binary
                internal += 1
            elseif tree.kind[i] == TreeSim.SampledLeaf
                external += 1
            elseif tree.kind[i] == TreeSim.SampledUnary
                sampled_unary += 1
            end
        end
    end
    return (internal_branches=internal, external_branches=external, sampled_unary_branches=sampled_unary, node_count=nodes, cherries=cherries)
end

function reconstructed_forest_stat_counts(forest::AbstractVector{<:TreeSim.Tree})
    internal = 0
    external = 0
    sampled_unary = 0
    nodes = 0
    cherries = 0
    for tree in forest
        stats = reconstructed_tree_stat_counts(tree)
        internal += stats.internal_branches
        external += stats.external_branches
        sampled_unary += stats.sampled_unary_branches
        nodes += stats.node_count
        cherries += stats.cherries
    end
    return (components=length(forest), internal_branches=internal, external_branches=external,
            sampled_unary_branches=sampled_unary, node_count=nodes, cherries=cherries)
end
