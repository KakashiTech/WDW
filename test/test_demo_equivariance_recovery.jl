using Test
using WDW
using LinearAlgebra

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

@testset "demo_equivariance_recovery" begin
    n = 12
    G = Q.dihedral_group(n)
    # Deterministic base matrix (no Random dependency)
    M = reshape(collect(1:n*n), n, n) .* 0.01
    W_eq = Q.project_equivariant(M, G)
    e_eq = eqerr(W_eq, G)

    # Deterministic rupture (non-equivariant w.r.t. permutations)
    R = Diagonal(collect(1:n))
    W_rupt = W_eq + 0.10 * R
    e_rupt = eqerr(W_rupt, G)

    W_rec = Q.project_equivariant(W_rupt, G)
    e_rec = eqerr(W_rec, G)

    @test e_eq <= 1e-8
    @test e_rec <= 1e-8
    @test e_rec <= 1e-2 * e_rupt
end
