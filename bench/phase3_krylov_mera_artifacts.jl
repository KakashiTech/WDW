#!/usr/bin/env julia
using WDW
using LinearAlgebra
using Random

const Kry = WDW.Krylov
const Tn = WDW.Tensor

# Krylov/Lanczos metrics
Random.seed!(0)
n = 20
H0 = randn(n, n)
H = Symmetric((H0 + H0')/2)
v0 = randn(n)

ms = 3:10
complexities = Float64[]
for m in ms
    T, α, β = Kry.lanczos_tridiagonal(Matrix(H), v0, m)
    c = Kry.krylov_spread_complexity(T)
    push!(complexities, c)
end
increasing = all(complexities[i] <= complexities[i+1] + 1e-12 for i in 1:length(complexities)-1)

# MERA metrics
x = randn(32) # power of 2 length not strictly required by our ops, but typical
levels = 5
errs = [ Tn.multiscale_error(x, levels, k) for k in 0:levels ]
zeros_thetas = zeros(Float64, levels)
keep = min(2, levels)
θopt, eopt = Tn.optimize_thetas(x, levels, keep; iters=30, step=0.15)
e0 = Tn.param_multiscale_error(x, levels, keep, zeros_thetas)
improves = (eopt <= e0 + 1e-9)

# Write artifacts
isdir("bench") || mkpath("bench")
open("bench/phase3_krylov_metrics.csv", "w") do io
    println(io, "m,complexity")
    for (i,m) in enumerate(ms)
        println(io, string(m, ",", round(complexities[i], digits=8)))
    end
end

open("bench/phase3_mera_metrics.csv", "w") do io
    println(io, "keep_levels,error")
    for k in 0:levels
        println(io, string(k, ",", round(errs[k+1], digits=8)))
    end
    println(io, string("opt_keep_", keep, ",", round(eopt, digits=8)))
    println(io, string("base_keep_", keep, ",", round(e0, digits=8)))
end

open("bench/phase3_krylov_mera_certificate.txt", "w") do io
    println(io, "WDW++ Phase 3′ Certificate: Krylov & MERA")
    println(io, "krylov_complexity_increasing=$(increasing)")
    println(io, "mera_opt_improves=$(improves)")
    println(io, "krylov_last_complexity=$(round(last(complexities), digits=8))")
    println(io, "mera_error_keep$(keep)_opt=$(round(eopt, digits=8))")
    println(io, "mera_error_keep$(keep)_base=$(round(e0, digits=8))")
end
