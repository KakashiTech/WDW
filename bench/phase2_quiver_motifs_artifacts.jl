#!/usr/bin/env julia
using WDW
using LinearAlgebra

const Al = WDW.Algebra
const Mv = WDW.Motives

# Quiver with near-ring layer and multi-walk aggregation
q = Al.Quiver([1,2,3,4], [(1,2), (2,3), (3,4)])
layer = Al.QuiverLayer(q, 2, 2)
X = [1.0 0.0; 0.5 -1.0; -2.0 3.0; 0.1 0.2]
Y1 = Al.apply_quiver(layer, X)
Y2 = Al.apply_quiver_walks(layer, X; depth=2)
A = Al.adjacency_matrix(q)
B = Al.normalized_adjacency(q)
stable = Al.is_spectrally_stable(B)
M = Al.mv_activation(Y1, [identity, Al.relu])

isdir("bench") || mkpath("bench")
open("bench/phase2_quiver_metrics.csv", "w") do io
    println(io, "metric,value")
    println(io, "n_nodes,", size(X,1))
    println(io, "in_dim,", size(X,2))
    println(io, "out_dim,", size(Y1,2))
    println(io, "stable_norm_adj,", stable)
    println(io, "walk_depth2_norm,", round(norm(Y2), digits=6))
    println(io, "mv_activation_cols,", size(M,2))
end

# Motives: correspondences and features over primes
cycles = [(1,1,1.0), (2,2,0.5), (3,2,0.5)]  # from m=3 to n=2
C = Mv.correspondence_matrix(cycles, 3, 2)
x = [2.0, -1.0, 0.5]
y = Mv.apply_correspondence(C, x)

Aint = [1 1 0; 0 1 1]
bint = [1, 1]
primes = [3,5,7,11]
feats = Mv.motivic_features(Aint, bint, primes)

open("bench/phase2_motifs_metrics.csv", "w") do io
    println(io, "metric,value")
    println(io, "corr_rows,", size(C,1))
    println(io, "corr_cols,", size(C,2))
    println(io, "y_norm,", round(norm(y), digits=6))
    for (i,p) in enumerate(primes)
        println(io, "feat_p$(p),", round(feats[i], digits=6))
    end
end

open("bench/phase2_certificate.txt", "w") do io
    println(io, "WDW++ Phase 2′ Certificate: Quivers (near-ring), Motifs (correspondences)")
    println(io, "quiver_stable_norm_adj=$(stable)")
    println(io, "apply_quiver_walks_depth2_norm=$(round(norm(Y2), digits=6))")
    println(io, "mv_activation_cols=$(size(M,2))")
    println(io, "correspondence_shape=$(size(C,1))x$(size(C,2))")
    println(io, "motivic_features_len=$(length(feats)) primes=$(join(primes,","))")
end
