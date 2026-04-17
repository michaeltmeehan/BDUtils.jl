#!/usr/bin/env julia

using BDUtils
using Printf
using TreeSim

import Base: +

const R_WRAPPER = joinpath(@__DIR__, "treepar_compare.R")

struct ComparisonResult
    checked::Int
    mismatches::Int
    skipped::Int
end

abstract type AbstractConditioning end

struct CurrentBDUtilsConditioning <: AbstractConditioning end

struct NoConditioning <: AbstractConditioning end

# Developer note: future public support for multiple conditioning conventions
# should grow from a structured conditioning abstraction like this, not from
# ad hoc root/survival booleans threaded through the likelihood API.
struct TreeParLikShiftsSTTConditioning <: AbstractConditioning
    root::Int
    survival::Int
    include_root_edge::Bool
end

function +(a::ComparisonResult, b::ComparisonResult)
    return ComparisonResult(a.checked + b.checked,
                            a.mismatches + b.mismatches,
                            a.skipped + b.skipped)
end

function run_r(args::Vector{String})
    cmd = Cmd(vcat(["Rscript", R_WRAPPER], args))
    try
        return parse(Float64, strip(read(cmd, String)))
    catch err
        if err isa Base.IOError || err isa Base.ProcessFailedException
            error("R TreePar wrapper failed for arguments $(join(args, " ")). Ensure Rscript is on PATH and TreePar is installed.")
        end
        rethrow()
    end
end

function approx_equal(x, y; atol=1e-10, rtol=1e-8)
    isfinite(x) && isfinite(y) || return false
    return abs(x - y) <= atol + rtol * max(abs(x), abs(y))
end

function report(label, x, y; atol=1e-10, rtol=1e-8)
    ok = approx_equal(x, y; atol=atol, rtol=rtol)
    delta = x - y
    println(rpad(label, 54),
            " julia=", @sprintf("%.12g", x),
            "  ref=", @sprintf("%.12g", y),
            "  diff=", @sprintf("%.4e", delta),
            ok ? "  [OK]" : "  [MISMATCH]")
    return ok
end

bd_params(lambda, mu, psi, rho0; r = 1.0) = ConstantRateBDParameters(lambda, mu, psi, r, rho0)

bd_p0(t, lambda, mu, psi, rho0) = E_constant(t, bd_params(lambda, mu, psi, rho0))
bd_p1(t, lambda, mu, psi, rho0) = rho0 * exp(g_constant(t, bd_params(lambda, mu, psi, rho0)))

function treepar_grid()
    return [
        (0.0, 1.2, 0.3, 0.8),
        (0.1, 1.2, 0.3, 0.8),
        (1.0, 1.2, 0.3, 0.8),
        (2.5, 2.0, 0.4, 0.5),
        (5.0, 0.9, 0.2, 0.95),
        (0.01, 3.0, 1.0, 0.2),
        (10.0, 1.01, 1.0, 0.7),
        (3.0, 0.6, 0.9, 0.4),
        (8.0, 1.5, 0.0, 0.6),
    ]
end

function ode_grid()
    return [
        (1e-4, 1.2, 0.3, 0.0, 0.8),
        (0.1, 1.2, 0.3, 1e-10, 0.8),
        (1.0, 1.2, 0.3, 0.05, 0.8),
        (2.5, 2.0, 0.4, 0.2, 0.5),
        (5.0, 0.9, 0.2, 0.1, 0.95),
        (0.01, 3.0, 1.0, 1e-8, 0.2),
        (10.0, 1.01, 1.0, 0.01, 0.7),
        (3.0, 0.6, 0.9, 1e-6, 0.4),
        (8.0, 1.5, 0.0, 0.05, 0.6),
        (2.0, 1.2, 0.3, 0.1, 0.0),
    ]
end

function compare_treepar_helpers()
    println("== TreePar helper comparisons ==")
    checked = 0
    mismatches = 0
    skipped = 0

    for (t, lambda, mu, rho0) in treepar_grid()
        jp0 = bd_p0(t, lambda, mu, 0.0, rho0)
        rp0 = run_r(["p0", string(t), string(lambda), string(mu), string(rho0)])
        checked += 1
        mismatches += report("p0 t=$t lambda=$lambda mu=$mu rho=$rho0", jp0, rp0) ? 0 : 1

        if rho0 == 0.0
            println(rpad("p1 t=$t lambda=$lambda mu=$mu rho=$rho0", 54),
                    " skipped: rho=0 is degenerate for p1 comparison")
            skipped += 1
        else
            jp1 = bd_p1(t, lambda, mu, 0.0, rho0)
            rp1 = run_r(["p1", string(t), string(lambda), string(mu), string(rho0)])
            checked += 1
            mismatches += report("p1 t=$t lambda=$lambda mu=$mu rho=$rho0", jp1, rp1) ? 0 : 1
        end
    end

    println(mismatches == 0 ? "\nTreePar helper comparisons passed.\n" :
                              "\nTreePar helper comparisons had mismatches.\n")
    return ComparisonResult(checked, mismatches, skipped)
end

function check_boundaries()
    println("== Boundary conditions ==")
    checked = 0
    mismatches = 0

    for (_, lambda, mu, psi, rho0) in ode_grid()
        e0 = bd_p0(0.0, lambda, mu, psi, rho0)
        checked += 1
        mismatches += report("E(0) lambda=$lambda mu=$mu psi=$psi rho=$rho0",
                             e0, 1.0 - rho0; atol=5e-14, rtol=1e-12) ? 0 : 1

        p10 = bd_p1(0.0, lambda, mu, psi, rho0)
        checked += 1
        mismatches += report("rho*exp(g(0)) lambda=$lambda mu=$mu psi=$psi rho=$rho0",
                             p10, rho0; atol=5e-14, rtol=1e-12) ? 0 : 1
    end

    println(mismatches == 0 ? "\nBoundary checks passed.\n" : "\nBoundary checks had mismatches.\n")
    return ComparisonResult(checked, mismatches, 0)
end

function central_difference(f, t)
    h = cbrt(eps(Float64)) * max(1.0, abs(t))
    if t > h
        return (f(t + h) - f(t - h)) / (2h)
    end
    return (f(t + h) - f(t)) / h
end

function check_ode_identities()
    println("== ODE identity checks ==")
    checked = 0
    mismatches = 0

    for (t, lambda, mu, psi, rho0) in ode_grid()
        pars = bd_params(lambda, mu, psi, rho0)
        E(tval) = E_constant(tval, pars)
        g(tval) = g_constant(tval, pars)

        e = E(t)
        dE_num = central_difference(E, t)
        dE_rhs = -(lambda + mu + psi) * e + lambda * e * e + mu
        checked += 1
        mismatches += report("dE/dt t=$t lambda=$lambda mu=$mu psi=$psi rho=$rho0",
                             dE_num, dE_rhs; atol=2e-6, rtol=2e-5) ? 0 : 1

        dg_num = central_difference(g, t)
        dg_rhs = -(lambda + mu + psi) + 2lambda * e
        checked += 1
        mismatches += report("dg/dt t=$t lambda=$lambda mu=$mu psi=$psi rho=$rho0",
                             dg_num, dg_rhs; atol=2e-6, rtol=2e-5) ? 0 : 1
    end

    println(mismatches == 0 ? "\nODE identity checks passed.\n" : "\nODE identity checks had mismatches.\n")
    return ComparisonResult(checked, mismatches, 0)
end

function note_likelihoods_not_compared()
    println("== Likelihood comparison assessment ==")
    println("LikConstant remains out of scope for this pass.")
    println("LikShiftsSTT is probed below only in a restricted r=1/no-sampled-unary regime.")
    println("Raw agreement is not assumed because TreePar's root/survival and 2λ branching conventions")
    println("are not yet proven to match BDUtils' current stem/origin-conditioned assembly.\n")
    return ComparisonResult(0, 0, 0)
end

function tiny_serial_tree_balanced()
    return Tree(
        [0.0, 0.6, 1.0, 1.4],
        [2, 3, 0, 0],
        [0, 4, 0, 0],
        [0, 1, 2, 2],
        [Root, Binary, SampledLeaf, SampledLeaf],
        [0, 0, 0, 0],
        [0, 0, 101, 102],
    )
end

function tiny_serial_tree_ladder()
    return Tree(
        [0.0, 0.4, 0.9, 1.2, 1.7, 2.0],
        [2, 3, 0, 5, 0, 0],
        [0, 4, 0, 6, 0, 0],
        [0, 1, 2, 2, 4, 4],
        [Root, Binary, SampledLeaf, Binary, SampledLeaf, SampledLeaf],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 101, 0, 102, 103],
    )
end

function treepar_events(tree::TreeSim.Tree; include_root_edge::Bool)
    Tfinal = maximum(tree.time)
    events = Tuple{Float64,Int,String}[]

    for node in tree
        if node.kind == Root
            include_root_edge && push!(events, (Tfinal - node.time, 1, "RootAsTransmission"))
        elseif node.kind == Binary
            push!(events, (Tfinal - node.time, 1, "Binary"))
        elseif node.kind == SampledLeaf
            push!(events, (Tfinal - node.time, 0, "SampledLeaf"))
        else
            error("Restricted LikShiftsSTT comparison only supports Root, Binary, and SampledLeaf nodes; saw $(node.kind).")
        end
    end

    sort!(events; by = x -> (-x[1], -x[2]))
    times = [e[1] for e in events]
    ttype = [e[2] for e in events]
    labels = [e[3] for e in events]
    return times, ttype, labels
end

function restricted_events(tree::TreeSim.Tree, conditioning::TreeParLikShiftsSTTConditioning)
    times, ttype, labels = treepar_events(tree; include_root_edge = conditioning.include_root_edge)
    return times, ttype, labels
end

function csv(xs)
    return join(xs, ",")
end

function treepar_likshift_loglik(lambda, mu, psi, rho0, times, ttype; root, survival)
    total_removal = mu + psi
    sprob = psi / total_removal
    nll = run_r([
        "LikShiftsSTT",
        string(lambda),
        string(total_removal),
        csv(times),
        csv(ttype),
        string(rho0),
        string(sprob),
        string(root),
        string(survival),
    ])
    return -nll
end

function validation_loglikelihood(tree::TreeSim.Tree, lambda, mu, psi, ::CurrentBDUtilsConditioning; rho0 = 0.0)
    return bd_loglikelihood_constant(tree, bd_params(lambda, mu, psi, rho0))
end

function validation_loglikelihood(tree::TreeSim.Tree, lambda, mu, psi, ::NoConditioning; rho0 = 0.0)
    terms = bd_likelihood_terms(tree, bd_params(lambda, mu, psi, rho0))
    return diagnostic_total(terms; root = :root_none, branch = :lambda)
end

function validation_loglikelihood(tree::TreeSim.Tree, lambda, mu, psi,
                                  conditioning::TreeParLikShiftsSTTConditioning; rho0 = 0.0)
    pars = bd_params(lambda, mu, psi, rho0)
    times, ttype, _ = restricted_events(tree, conditioning)
    transmission = times[ttype .== 1]
    sampling = times[ttype .== 0]

    if conditioning.root == 1
        push!(transmission, maximum(transmission))
    elseif conditioning.root != 0
        error("TreePar root flag must be 0 or 1.")
    end

    # TreePar's LikShiftsSTT implements conditioning by additive log terms in
    # this likelihood accumulator. Longer term, the MacPherson appendix points
    # toward modelling conditioning as a multiplicative factor S applied to the
    # base tree likelihood, rather than baking root/survival booleans into the
    # assembly. This validation-only type keeps that future split visible.
    oldest_transmission = maximum(transmission)
    loglik = -(conditioning.root + 1) * log(2lambda)
    if conditioning.survival == 1
        loglik -= (conditioning.root + 1) * log1p(-E_constant(oldest_transmission, pars))
    elseif conditioning.survival != 0
        error("TreePar survival flag must be 0 or 1.")
    end

    for t in transmission
        loglik += g_constant(t, pars) + log(2lambda)
    end

    for t in sampling
        loglik += -g_constant(t, pars) + log(psi)
    end

    loglik -= (length(transmission) - 1 - conditioning.root) * log(2)
    return loglik
end

function bd_likelihood_terms(tree::TreeSim.Tree, pars::ConstantRateBDParameters)
    Tfinal = maximum(tree.time)
    root_term = log1p(-E_constant(Tfinal, pars))
    binary_terms = NamedTuple[]
    leaf_terms = NamedTuple[]

    for node in tree
        tau = Tfinal - node.time
        if node.kind == Binary
            g = g_constant(tau, pars)
            push!(binary_terms, (
                id = node.id,
                time = node.time,
                tau = tau,
                base = log(pars.λ) + g,
                doubled = log(2pars.λ) + g,
            ))
        elseif node.kind == SampledLeaf
            g = g_constant(tau, pars)
            push!(leaf_terms, (
                id = node.id,
                time = node.time,
                tau = tau,
                base = log(pars.ψ) - g,
            ))
        elseif node.kind != Root
            error("Restricted diagnostic does not support $(node.kind).")
        end
    end

    return (
        root_current = root_term,
        root_none = 0.0,
        root_plus_loglambda = root_term + log(pars.λ),
        root_double_survival = 2root_term,
        root_treepar_survival0 = 0.0,
        root_treepar_survival1 = root_term,
        binary_terms = binary_terms,
        leaf_terms = leaf_terms,
    )
end

function sum_field(xs, field)
    return sum(getfield(x, field) for x in xs; init = 0.0)
end

function diagnostic_total(terms; root = :root_current, branch = :lambda, root_extra_log2 = false)
    binary_sum = branch == :lambda ? sum_field(terms.binary_terms, :base) :
                 branch == :two_lambda ? sum_field(terms.binary_terms, :doubled) :
                 error("unknown branch variant $branch")
    total = getfield(terms, root) + binary_sum + sum_field(terms.leaf_terms, :base)
    return root_extra_log2 ? total + log(2) : total
end

function print_bd_term_breakdown(terms)
    println("  BDUtils components:")
    println("    root current log(1-E(T)) = ", @sprintf("%.12g", terms.root_current))
    println("    binary sum log(lambda)+g = ", @sprintf("%.12g", sum_field(terms.binary_terms, :base)))
    for term in terms.binary_terms
        println("      binary node=$(term.id) time=", @sprintf("%.12g", term.time),
                " tau=", @sprintf("%.12g", term.tau),
                " log(lambda)+g=", @sprintf("%.12g", term.base),
                " log(2lambda)+g=", @sprintf("%.12g", term.doubled))
    end
    println("    sampled-leaf sum log(psi)-g = ", @sprintf("%.12g", sum_field(terms.leaf_terms, :base)))
    for term in terms.leaf_terms
        println("      leaf node=$(term.id) time=", @sprintf("%.12g", term.time),
                " tau=", @sprintf("%.12g", term.tau),
                " log(psi)-g=", @sprintf("%.12g", term.base))
    end
    println("    total current = ", @sprintf("%.12g", diagnostic_total(terms)))
end

function print_variant_table(terms, treepar_value)
    variants = [
        ("current", :root_current, :lambda, false),
        ("branch log(2lambda)", :root_current, :two_lambda, false),
        ("current + root log2", :root_current, :lambda, true),
        ("branch log(2lambda) + root log2", :root_current, :two_lambda, true),
        ("root none", :root_none, :lambda, false),
        ("root none + branch log(2lambda)", :root_none, :two_lambda, false),
        ("root log(1-E)+log(lambda)", :root_plus_loglambda, :lambda, false),
        ("root 2log(1-E)", :root_double_survival, :lambda, false),
        ("TreePar-like survival=0 root", :root_treepar_survival0, :lambda, false),
        ("TreePar-like survival=1 root", :root_treepar_survival1, :lambda, false),
    ]

    println("    diagnostic assembly vs this TreePar total:")
    best_name = ""
    best_abs = Inf
    for (name, root, branch, extra_log2) in variants
        val = diagnostic_total(terms; root = root, branch = branch, root_extra_log2 = extra_log2)
        diff = val - treepar_value
        absdiff = abs(diff)
        if absdiff < best_abs
            best_abs = absdiff
            best_name = name
        end
        println("      ", rpad(name, 34),
                " total=", @sprintf("% .12g", val),
                " diff=", @sprintf("% .4e", diff))
    end
    println("      closest variant: ", best_name, " absdiff=", @sprintf("%.4e", best_abs))
end

function compare_restricted_likshiftstt()
    println("== Restricted LikShiftsSTT probes ==")
    println("Regime: r=1, no SampledUnary nodes, hand-built trees, explicit τ = Tfinal - node.time conversion.")

    cases = [
        ("balanced-2-tip", tiny_serial_tree_balanced(), 2.0, 0.5, 0.4, 0.0),
        ("ladder-3-tip", tiny_serial_tree_ladder(), 1.4, 0.3, 0.2, 0.0),
    ]

    checked = 0
    original_close = 0
    aligned_close = 0

    for (name, tree, lambda, mu, psi, rho0) in cases
        pars = bd_params(lambda, mu, psi, rho0)
        terms = bd_likelihood_terms(tree, pars)
        bd = bd_loglikelihood_constant(tree, pars)
        println("\n$name BDUtils logL=", @sprintf("%.12g", bd),
                "  lambda=$lambda mu=$mu psi=$psi rho0=$rho0")
        print_bd_term_breakdown(terms)

        for config in ((root=1, survival=0, include_root_edge=false),
                       (root=1, survival=1, include_root_edge=false),
                       (root=0, survival=0, include_root_edge=true),
                       (root=0, survival=1, include_root_edge=true))
            conditioning = TreeParLikShiftsSTTConditioning(config.root, config.survival, config.include_root_edge)
            times, ttype, labels = treepar_events(tree; include_root_edge=config.include_root_edge)
            tp = treepar_likshift_loglik(lambda, mu, psi, rho0, times, ttype;
                                         root=config.root, survival=config.survival)
            aligned = validation_loglikelihood(tree, lambda, mu, psi, conditioning; rho0 = rho0)
            checked += 1
            ok = approx_equal(bd, tp; atol=1e-8, rtol=1e-7)
            aligned_ok = approx_equal(aligned, tp; atol=1e-8, rtol=1e-7)
            original_close += ok ? 1 : 0
            aligned_close += aligned_ok ? 1 : 0
            println("  TreePar root=$(config.root) survival=$(config.survival)",
                    " include_root_edge=$(config.include_root_edge)")
            println("    times=", times, " ttype=", ttype, " labels=", labels)
            println("    original BDUtils logL=", @sprintf("%.12g", bd),
                    " diff(original-TreePar)=", @sprintf("%.4e", bd - tp),
                    ok ? " [RAW MATCH]" : " [raw mismatch]")
            println("    TreePar-aligned validation logL=", @sprintf("%.12g", aligned),
                    " TreePar logL=", @sprintf("%.12g", tp),
                    " diff(aligned-TreePar)=", @sprintf("%.4e", aligned - tp),
                    aligned_ok ? " [ALIGNED MATCH]" : " [aligned mismatch]")
            print_variant_table(terms, tp)
        end
    end

    aligned_mismatches = checked - aligned_close
    unexpected_original_matches = original_close

    println("\nRestricted LikShiftsSTT original raw matches: $original_close / $checked")
    println("Restricted LikShiftsSTT TreePar-aligned matches: $aligned_close / $checked")
    println("Interpretation: original raw mismatches are expected here and preserve the distinction between")
    println("BDUtils' current convention and TreePar's root/survival/event-counting convention.")
    println("The validation-only TreePar-aligned assembly is expected to match TreePar exactly in this restricted regime.\n")

    if unexpected_original_matches > 0
        println("WARNING: original BDUtils unexpectedly matched TreePar in $unexpected_original_matches restricted cases.")
    end
    if aligned_mismatches > 0
        println("WARNING: TreePar-aligned validation assembly mismatched TreePar in $aligned_mismatches restricted cases.")
    end

    return ComparisonResult(checked, aligned_mismatches + unexpected_original_matches, 0)
end

function main()
    result = compare_treepar_helpers()
    result += check_boundaries()
    result += check_ode_identities()
    result += note_likelihoods_not_compared()
    result += compare_restricted_likshiftstt()

    println("Summary: checked=$(result.checked) mismatches=$(result.mismatches) skipped=$(result.skipped)")
    return result.mismatches == 0
end

try
    ok = main()
    exit(ok ? 0 : 1)
catch err
    println(stderr, "Validation failed: ", sprint(showerror, err))
    exit(2)
end
