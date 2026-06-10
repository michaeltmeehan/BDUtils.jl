#!/usr/bin/env julia

# Manual validation for cached origin-time sampling-time likelihood evaluation.
#
# Run from the package root:
#
#   julia --project=. scripts/validation/origin_time_likelihood_cache_validation.jl

using BDUtils
using Printf

pars = ConstantRateBDParameters(1.35, 0.3, 0.55, 1.0, 0.4)
tℓ = 1.4
sampling_times = [0.35, 0.8]
sample_counts = [1, 2]
terminal_count = 1
t0_grid = collect(range(-0.4, stop=0.3, length=8))

function main()
    cache = cache_sampling_time_likelihood(
        sampling_times,
        sample_counts,
        terminal_count,
        pars;
        tℓ=tℓ,
        labelled_samples=false,
    )

    println("origin-time likelihood cache validation")
    println("parameters = ", pars)
    println("sampling_times = ", sampling_times)
    println("sample_counts = ", sample_counts)
    println("terminal_count = ", terminal_count)
    println()
    @printf("%10s  %16s  %16s  %12s  %s\n", "t0", "full", "cached", "absdiff", "status")

    max_absdiff = 0.0
    all_ok = true
    for t0 in t0_grid
        full = sampling_time_likelihood(t0, sampling_times, sample_counts, terminal_count, pars; tℓ=tℓ)
        cached = sampling_time_likelihood(cache, t0)
        diff = abs(full - cached)
        ok = isapprox(cached, full; rtol=5e-12, atol=5e-14)
        max_absdiff = max(max_absdiff, diff)
        all_ok &= ok
        @printf("%10.5f  %16.8e  %16.8e  %12.4e  %s\n", t0, full, cached, diff, ok ? "ok" : "FAIL")
    end

    println()
    @printf("maximum absolute difference = %.6e\n", max_absdiff)
    println("status = ", all_ok ? "ok" : "FAIL")
end

main()
