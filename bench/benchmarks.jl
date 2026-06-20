using WDW
using LinearAlgebra
using Random

Random.seed!(0)

const _bench_results = Vector{Tuple{String,Float64}}()

function bench(name, f)
    t = @elapsed f()
    push!(_bench_results, (name, t))
    println(name, ": ", round(t, digits=6), " s")
end

# Compatibility in case do-block is parsed as first argument
bench(f::Function, name::AbstractString) = bench(name, f)

bench("Quiver adjacency stability") do
    q = WDW.Algebra.Quiver(collect(1:100), [(i, i%100+1) for i in 1:100])
    A = WDW.Algebra.normalized_adjacency(q)
    WDW.Algebra.is_spectrally_stable(A)
end

bench("Krylov Lanczos 64x64") do
    H = Symmetric(randn(64,64))
    v0 = randn(64)
    T, α, β = WDW.Krylov.lanczos_tridiagonal(Matrix(H), v0, 16)
    WDW.Krylov.krylov_spread_complexity(T)
end

bench("Time ITE evolve 64 steps") do
    H = Symmetric(randn(32,32))
    psi0 = randn(32)
    psi, energies = WDW.TimeITE.evolve(Matrix(H), psi0, 0.05, 64)
    WDW.TimeITE.monotone_energy(energies)
end

bench("Quantum equivariant projection n=64") do
    g = WDW.Quantum.CyclicGroup(64)
    M = randn(64,64)
    W = WDW.Quantum.project_equivariant(M, g)
    WDW.Quantum.is_equivariant(W, g; trials=1)
end

# Comparative benches
G_dn = WDW.Quantum.dihedral_group(256)
Gc_dn = WDW.Quantum.cache_permutation_matrices(G_dn)

bench("Quantum FinitePerm vs Cached (Dn n=256) — FinitePerm") do
    M = randn(256,256)
    W = WDW.Quantum.project_equivariant(M, G_dn)
    WDW.Quantum.is_equivariant(W, G_dn)
end

bench("Quantum FinitePerm vs Cached (Dn n=256) — Cached") do
    M = randn(256,256)
    W = WDW.Quantum.project_equivariant(M, Gc_dn)
    WDW.Quantum.is_equivariant(W, Gc_dn)
end

bench("Quantum presented vs reduced (Dn n=64) — presented") do
    n = 64
    rot = [2:64; 1]
    refl = collect(n:-1:1)
    Gp = WDW.Quantum.presented_perm_group(n, [rot, refl]; max_size=100000)
    length(Gp.perms)
end

bench("Quantum presented vs reduced (Dn n=64) — reduced") do
    n = 64
    rot = [2:64; 1]
    refl = collect(n:-1:1)
    rels = [fill(1, n), [2,2], [2,1,2,1]]
    Gr = WDW.Quantum.presented_perm_group_reduced(n, [rot, refl]; relations=rels, max_size=100000)
    length(Gr.perms)
end

bench("SO2 project_equivariant n=512 K=16") do
    n = 512
    K = 16
    G = WDW.Quantum.SO2_linear_group(n, K)
    M = randn(n,n)
    W = WDW.Quantum.project_equivariant(M, G)
    WDW.Quantum.is_equivariant(W, G; tol=1e-2)
end

bench("SO2 project_equivariant n=512 K=64") do
    n = 512
    K = 64
    G = WDW.Quantum.SO2_linear_group(n, K)
    M = randn(n,n)
    W = WDW.Quantum.project_equivariant(M, G)
    WDW.Quantum.is_equivariant(W, G; tol=5e-3)
end

bench("MERA optimize_thetas L=6 keep=3") do
    x = collect(1.0:64.0) .- 32.0
    levels = 6
    keep_levels = 3
    e0 = WDW.Tensor.param_multiscale_error(x, levels, keep_levels, zeros(levels))
    θ, e1 = WDW.Tensor.optimize_thetas(x, levels, keep_levels; iters=40, step=0.2)
    # Print errors for manual inspection; timings go to CSV
    println("MERA_error_before=", e0, ", after=", e1)
end

# Persist results for CI artifacts (non-fatal if write fails)
try
    mkpath("bench")
    open(joinpath("bench", "results.csv"), "w") do io
        println(io, "name,seconds")
        for (n, t) in _bench_results
            println(io, string(n, ",", t))
        end
    end
catch e
end
