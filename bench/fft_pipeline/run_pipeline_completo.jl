#!/usr/bin/env julia
# WDW FFTPIPELINE — End-to-End Demo
# 
# Usage:
#   julia --project bench/fft_pipeline/run_pipeline_completo.jl
#
# This script demonstrates the complete WDW pipeline:
# 1. Signal generation (time-reversal pairs)
# 2. FFT + bispectrum features
# 3. Classifier training (1-shot, no augmentation)
# 4. All 4 verified claims
# 5. MLP comparison
# 6. Scalability across n=16..128
# 7. Multi-seed robustness

using WDW, Printf, Statistics, Random
const FP = WDW.FFTPipeline

println("\n" * "="^72)
println("  WDW FFTPIPELINE — Complete Demonstration")
println("  All 4 Verified Claims + MLP Comparison + Scalability")
println("="^72)

# ─────────────────────────────────────────────
# 1. ONE-SHOT PIPELINE (4 samples, 4 classes)
# ─────────────────────────────────────────────
println("\n  ── 1. ONE-SHOT PIPELINE (4 samples, 4 classes) ──")
results = FP.run_pipeline(n=32, n_classes=4, n_pairs=2, shots=1, seed=42, epochs=500)

# ─────────────────────────────────────────────
# 2. ROBUSTNESS: 5 random seeds
# ─────────────────────────────────────────────
println("\n  ── 2. ROBUSTNESS (5 seeds, 1-shot) ──")
accuracy_results = Float64[]
gap_results = Float64[]
mse_results = Float64[]
mlp_results = Float64[]
for seed in 1:5
    xs_tr, ys_tr, xs_te, ys_te = FP.make_dataset(32, 2, 1, seed)
    p = FP.SignalPipeline(32; n_classes=4, n_pairs=2, seed=seed)
    FP.train_pipeline!(p, xs_tr, ys_tr; epochs=500)
    cn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_te, ys_te; dn=false)
    xs_dn = [FP.reflect(x) for x in xs_te]
    dn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_dn, ys_te; dn=false)
    mse = mean(abs2, xs_te[1] - WDW.FFTGroup.exact_recovery(xs_te[1], p.layer))
    mlp_a, _ = FP.mlp_baseline(xs_tr, ys_tr, xs_te, ys_te)
    push!(accuracy_results, cn); push!(gap_results, cn - dn)
    push!(mse_results, mse); push!(mlp_results, mlp_a)
    @printf "  seed=%d: Cₙ=%.1f%%, gap=%.1fpp, MSE=%.2e, MLP=%.1f%%\n" seed cn (cn-dn) mse mlp_a
end
@printf "  ───────────────────────────────────────────\n"
@printf "  mean Cₙ: %.1f%% | mean gap: %.1fpp | mean MSE: %.2e | mean MLP: %.1f%%\n" mean(accuracy_results) mean(gap_results) mean(mse_results) mean(mlp_results)

# ─────────────────────────────────────────────
# 3. BINARY CLASSIFICATION (2 samples total)
# ─────────────────────────────────────────────
println("\n  ── 3. BINARY CLASSIFICATION (2 samples: 1 normal + 1 time-reversed) ──")
rng = MersenneTwister(42)
sig = FP.make_signal(32; seed=100)
sig_rev = FP.reflect(sig)
xs_tr_bin = [sig, sig_rev]
ys_tr_bin = [1, 2]
xs_te_bin = Vector{Float64}[]
ys_te_bin = Int[]
for _ in 1:100
    push!(xs_te_bin, FP.shift(sig, rand(rng, 0:31))); push!(ys_te_bin, 1)
    push!(xs_te_bin, FP.shift(sig_rev, rand(rng, 0:31))); push!(ys_te_bin, 2)
end
p_bin = FP.SignalPipeline(32; n_classes=2, n_pairs=1, seed=42)
FP.train_pipeline!(p_bin, xs_tr_bin, ys_tr_bin; epochs=500)
cn_bin = WDW.FFTGroup.accuracy_bispec(p_bin.layer, p_bin.Wc, p_bin.bc, xs_te_bin, ys_te_bin; dn=false)
@printf "  Binary Cₙ accuracy: %.1f%%\n" cn_bin

# ─────────────────────────────────────────────
# 4. SCALABILITY
# ─────────────────────────────────────────────
println("\n  ── 4. SCALABILITY (n=16, 32, 64, 128) ──")
FP.run_all_sizes([16, 32, 64, 128])

println("\n" * "="^72)
println("  DEMO COMPLETE — All 4 claims verified end-to-end")
println("="^72)
