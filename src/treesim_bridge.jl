"""
    TreeSim.tree_from_eventlog(log::BDEventLog; validate=true, tj=0.0, tk=log.tmax)

Extract the strict single reconstructed sampled tree from a `BDEventLog`.

The extraction window is `(tj, tk]`: samples at exactly `tj` are excluded and
samples at exactly `tk` are included. Roots are the retained lineages extant
immediately after events at `tj`, matching [`A_at`](@ref).
"""
function TreeSim.tree_from_eventlog(log::BDEventLog; validate::Bool=true, tⱼ::Real=0.0, tₖ::Real=log.tmax, tj::Real=tⱼ, tk::Real=tₖ)
    return TreeSim.reconstructed_tree_from_eventlog(log; validate, tj, tk)
end

function TreeSim.reconstructed_tree_from_eventlog(log::BDEventLog; validate::Bool=true, tⱼ::Real=0.0, tₖ::Real=log.tmax, tj::Real=tⱼ, tk::Real=tₖ)
    forest = TreeSim.forest_from_eventlog(log; validate, tj, tk)
    isempty(forest) && return TreeSim.Tree()
    length(forest) == 1 ||
        error("reconstructed_tree_from_eventlog found $(length(forest)) retained sampled components. It is a strict single-tree API; use forest_from_eventlog to extract the reconstructed forest.")
    return only(forest)
end

function TreeSim.full_tree_from_eventlog(log::BDEventLog; validate::Bool=true, tⱼ::Real=0.0, tₖ::Real=log.tmax, tj::Real=tⱼ, tk::Real=tₖ)
    forest = full_forest_from_eventlog(log; validate, tj, tk)
    isempty(forest) && return TreeSim.Tree()
    length(forest) == 1 ||
        error("full_tree_from_eventlog found $(length(forest)) retained sampled components. It is a strict single-tree API.")
    return only(forest)
end

function TreeSim.forest_from_eventlog(log::BDEventLog; validate::Bool=true, tⱼ::Real=0.0, tₖ::Real=log.tmax, tj::Real=tⱼ, tk::Real=tₖ)
    tj_checked, tk_checked = _check_reconstructed_times(log, tj, tk)
    forest = _bd_sampled_ancestry_forest_from_eventlog(log, tj_checked, tk_checked; keep_unsampled_unary=false)
    validate && _validate_bd_sampled_ancestry_forest(log, forest, tj_checked, tk_checked; allow_unsampled_unary=false)
    return forest
end

"""
    full_forest_from_eventlog(log::BDEventLog; validate=true, tj=0.0, tk=log.tmax)

Extract all full sampled-ancestry components from a `BDEventLog`.

The retention window is `(tj, tk]`, matching reconstructed extraction. Unlike
`forest_from_eventlog`, retained birth events with only the child side sampled
are kept as `TreeSim.UnsampledUnary` nodes instead of being collapsed.
"""
function full_forest_from_eventlog(log::BDEventLog; validate::Bool=true, tⱼ::Real=0.0, tₖ::Real=log.tmax, tj::Real=tⱼ, tk::Real=tₖ)
    tj_checked, tk_checked = _check_reconstructed_times(log, tj, tk)
    forest = _bd_sampled_ancestry_forest_from_eventlog(log, tj_checked, tk_checked; keep_unsampled_unary=true)
    validate && _validate_bd_sampled_ancestry_forest(log, forest, tj_checked, tk_checked; allow_unsampled_unary=true)
    return forest
end

function _bd_sampled_ancestry_forest_from_eventlog(log::BDEventLog, tj::Float64, tk::Float64; keep_unsampled_unary::Bool)
    retained_roots = retained_lineages_at(log, tj, tk)
    isempty(retained_roots) && return TreeSim.Tree[]

    tree = TreeSim.Tree()
    active = Dict{Int,Int}()

    for i in length(log):-1:1
        time = log.time[i]
        tj < time <= tk || continue

        kind = log.kind[i]
        lineage = log.lineage[i]

        if kind == SerialSampling || kind == FossilizedSampling
            child = get(active, lineage, 0)
            if child == 0
                active[lineage] = _bd_push_tree_node!(tree, time, lineage, 0, 0, TreeSim.SampledLeaf)
            else
                node = _bd_push_tree_node!(tree, time, lineage, child, 0, TreeSim.SampledUnary)
                tree.parent[child] = node
                active[lineage] = node
            end
        elseif kind == Birth
            child_lineage = lineage
            parent_lineage = log.parent[i]
            child_node = get(active, child_lineage, 0)
            child_node == 0 && continue

            parent_node = get(active, parent_lineage, 0)
            if parent_node == 0
                if keep_unsampled_unary
                    node = _bd_push_tree_node!(tree, time, parent_lineage, child_node, 0, TreeSim.UnsampledUnary)
                    tree.parent[child_node] = node
                    active[parent_lineage] = node
                else
                    active[parent_lineage] = child_node
                end
            else
                node = _bd_push_tree_node!(tree, time, parent_lineage, parent_node, child_node, TreeSim.Binary)
                tree.parent[parent_node] = node
                tree.parent[child_node] = node
                active[parent_lineage] = node
            end
            delete!(active, child_lineage)
        end
    end

    for lineage in retained_roots
        child = get(active, lineage, 0)
        child == 0 && error("Retained lineage $lineage at tj=$tj has no reconstructed sampled descendant in the extracted tree.")
        root = _bd_push_tree_node!(tree, tj, lineage, child, 0, TreeSim.Root)
        tree.parent[child] = root
        active[lineage] = root
    end

    tree = _bd_canonicalize_tree(tree)
    roots = sort(TreeSim.roots(tree); by=i -> (tree.time[i], tree.host[i], i))
    forest = [_bd_subtree_as_tree(tree, root) for root in roots]
    sort!(forest; by=t -> (t.time[TreeSim.root(t)], t.host[TreeSim.root(t)]))
    return forest
end

function _validate_bd_sampled_ancestry_forest(log::BDEventLog, forest::AbstractVector{<:TreeSim.Tree}, tj::Float64, tk::Float64; allow_unsampled_unary::Bool)
    length(forest) == A_at(log, tj, tk) ||
        error("Extracted sampled-ancestry forest has $(length(forest)) components, but A_at(log, $tj, $tk) is $(A_at(log, tj, tk)).")

    expected_roots = retained_lineages_at(log, tj, tk)
    observed_roots = Int[]

    for tree in forest
        TreeSim.validate_tree(tree; require_single_root=true, require_reachable=true)
        root = TreeSim.root(tree)
        tree.time[root] == tj ||
            error("Sampled-ancestry tree root time $(tree.time[root]) does not match tj=$tj.")
        push!(observed_roots, tree.host[root])

        for i in eachindex(tree)
            kind = tree.kind[i]
            kind == TreeSim.UnsampledUnary && !allow_unsampled_unary &&
                error("UnsampledUnary node $i is not part of BDUtils reconstructed extraction.")
            kind in (TreeSim.Root, TreeSim.Binary, TreeSim.SampledLeaf, TreeSim.SampledUnary, TreeSim.UnsampledUnary) ||
                error("Unsupported TreeSim node kind $kind at node $i.")
            _bd_has_sampled_descendant(tree, i) ||
                error("Node $i has no sampled descendant.")
        end
    end

    sort!(observed_roots)
    observed_roots == expected_roots ||
        error("Extracted reconstructed roots $observed_roots do not match retained_lineages_at(log, $tj, $tk) = $expected_roots.")

    return true
end

function _bd_collapse_unsampled_unary(tree::TreeSim.Tree)
    drop = tree.kind .== TreeSim.UnsampledUnary
    any(drop) || return tree

    old_to_new = zeros(Int, length(tree))
    next = 0
    for i in eachindex(tree)
        drop[i] && continue
        next += 1
        old_to_new[i] = next
    end

    kept = findall(!, drop)
    return TreeSim.Tree(
        tree.time[kept],
        [_bd_remap(_bd_nearest_retained_descendant(tree, tree.left[old], drop), old_to_new) for old in kept],
        [_bd_remap(_bd_nearest_retained_descendant(tree, tree.right[old], drop), old_to_new) for old in kept],
        [_bd_remap(_bd_nearest_retained_parent(tree, old, drop), old_to_new) for old in kept],
        tree.kind[kept],
        tree.host[kept],
        tree.label[kept],
    )
end

function _bd_push_tree_node!(tree::TreeSim.Tree, time::Float64, host::Int, left::Int, right::Int, kind::TreeSim.NodeKind)
    push!(tree.time, time)
    push!(tree.left, left)
    push!(tree.right, right)
    push!(tree.parent, 0)
    push!(tree.kind, kind)
    push!(tree.host, host)
    push!(tree.label, kind == TreeSim.SampledLeaf ? host : 0)
    return length(tree)
end

function _bd_canonicalize_tree(tree::TreeSim.Tree)
    order = sortperm(eachindex(tree); by=i -> (tree.time[i], _bd_kind_order(tree.kind[i]), i))
    old_to_new = zeros(Int, length(tree))
    for (new, old) in pairs(order)
        old_to_new[old] = new
    end

    return TreeSim.Tree(
        tree.time[order],
        [_bd_remap(tree.left[old], old_to_new) for old in order],
        [_bd_remap(tree.right[old], old_to_new) for old in order],
        [_bd_remap(tree.parent[old], old_to_new) for old in order],
        tree.kind[order],
        tree.host[order],
        tree.label[order],
    )
end

function _bd_subtree_as_tree(tree::TreeSim.Tree, root_id::Int)
    old_nodes = sort(collect(TreeSim.preorder(tree, root_id)))
    old_to_new = Dict{Int,Int}(old => new for (new, old) in pairs(old_nodes))

    return TreeSim.Tree(
        tree.time[old_nodes],
        [_bd_remap_component(tree.left[old], old_to_new) for old in old_nodes],
        [_bd_remap_component(tree.right[old], old_to_new) for old in old_nodes],
        [_bd_remap_component(tree.parent[old], old_to_new) for old in old_nodes],
        tree.kind[old_nodes],
        tree.host[old_nodes],
        tree.label[old_nodes],
    )
end

_bd_remap(i::Int, old_to_new::Vector{Int}) = i == 0 ? 0 : old_to_new[i]
_bd_remap_component(i::Int, old_to_new::Dict{Int,Int}) = get(old_to_new, i, 0)
_bd_kind_order(kind::TreeSim.NodeKind) = kind == TreeSim.Root ? 1 : kind == TreeSim.Binary || kind == TreeSim.UnsampledUnary || kind == TreeSim.SampledUnary ? 2 : 3

function _bd_nearest_retained_descendant(tree::TreeSim.Tree, node::Int, drop::AbstractVector{Bool})
    node == 0 && return 0
    while drop[node]
        children = TreeSim.children(tree, node)
        isempty(children) && return 0
        node = only(children)
    end
    return node
end

function _bd_nearest_retained_parent(tree::TreeSim.Tree, node::Int, drop::AbstractVector{Bool})
    parent = tree.parent[node]
    while parent != 0 && drop[parent]
        parent = tree.parent[parent]
    end
    return parent
end

function _bd_has_sampled_descendant(tree::TreeSim.Tree, i::Int)
    stack = [i]
    while !isempty(stack)
        node = pop!(stack)
        tree.kind[node] == TreeSim.SampledLeaf && return true
        tree.left[node] != 0 && push!(stack, tree.left[node])
        tree.right[node] != 0 && push!(stack, tree.right[node])
    end
    return false
end
