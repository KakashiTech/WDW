#!/usr/bin/env julia
# Minimal reproducible demo: Equivariance detection and recovery via projection

using WDW
using LinearAlgebra
using Random
using Printf
using DelimitedFiles: writedlm

const Q = WDW.Quantum

function eqerr(W::AbstractMatrix, G; max_perms::Union{Int,Nothing}=nothing)
    n = size(W, 1)
    m = max_perms === nothing ? min(length(G.perms), n) : min(length(G.perms), max_perms)
    acc = 0.0
    for i in 1:m
        p = G.perms[i]
        for t in 1:n
            x = zeros(n); x[t] = 1.0
            acc += norm(Q.act(G, p, W * x) - W * Q.act(G, p, x))
        end
    end
    acc / (m * n)
end

function main()
    n     = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 12
    noise = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.10
    seed  = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 0
    outcsv = length(ARGS) >= 4 ? ARGS[4] : nothing
    outpng = length(ARGS) >= 5 ? ARGS[5] : nothing
    Random.seed!(seed)

    G = Q.dihedral_group(n)

    # 1) Build an exactly equivariant operator
    M = randn(n, n)
    W_eq = Q.project_equivariant(M, G)
    e_eq = eqerr(W_eq, G)

    # 2) Induce a rupture (break equivariance)
    R = randn(n, n)
    W_rupt = W_eq + noise * R
    e_rupt = eqerr(W_rupt, G)

    # 3) Recover via projection
    W_rec = Q.project_equivariant(W_rupt, G)
    e_rec = eqerr(W_rec, G)

    @printf "n = %d, noise = %.3f, seed = %d\n" n noise seed
    @printf "Equivariance error (equivariant base): %.3e\n" e_eq
    @printf "After induced rupture:                %.3e\n" e_rupt
    @printf "After projection (recovery):          %.3e\n" e_rec
    @printf "Recovery factor e_rec/e_rupt:         %.2e\n" e_rec / max(e_rupt, eps())

    ok = (e_eq <= 1e-8) && (e_rec <= 1e-8) && (e_rec <= 1e-2 * e_rupt)
    println(ok ? "STATUS: OK (equivariance recovered)" : "STATUS: CHECK (expected strong reduction)")

    if outcsv !== nothing
        try
            writedlm(outcsv, [
                "n"           n;
                "noise"       noise;
                "seed"        seed;
                "err_base"    e_eq;
                "err_rupt"    e_rupt;
                "err_recover" e_rec;
                "ratio"       e_rec / max(e_rupt, eps())
            ], ',')
            @printf "Wrote CSV: %s\n" outcsv
        catch err
            @printf "WARN: could not write CSV to %s (%s)\n" string(outcsv) string(err)
        end
    end

    if outpng !== nothing
        try
            Base.require(Base.PkgId(Base.UUID("91a5bcdd-55d7-5caf-9e0b-520d859cae80"), "Plots"))
            vals = [e_eq, e_rupt, e_rec]
            let vals = vals, outpng = outpng
                @eval using Plots
                p = Plots.bar(["base","rupt","recover"], $vals; title = "Equivariance Error", ylabel = "‖⋅‖", legend = false)
                Plots.savefig(p, $outpng)
            end
            @printf "Wrote PNG: %s\n" outpng
        catch err
            @printf "WARN: could not write PNG to %s (Plots not available or error: %s)\n" string(outpng) string(err)
        end
    end

    return nothing
end

isinteractive() || main()
