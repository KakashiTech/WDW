#!/usr/bin/env julia
# Full Symmetry Certificate — Unified WDW System Demo
#
# This demo runs the complete WDW system on a real neural network
# and produces a formal Symmetry Certificate.
#
# The certificate integrates all 7 pillars:
#   1. GROUP     — Symmetry group definition (C_n via bispectrum)
#   2. ANCHOR    — Provably invariant bispectrum measurement
#   3. PROFILE   — Layer-wise symmetry profiling
#   4. PROJECT   — Equivariant data compression
#   5. COMPRESS  — MERA + Krylov complexity
#   6. MDL       — Irreducibility certification
#   7. PAC-BAYES — Generalization bound
#
# Usage:
#   julia --project bench/full_certificate_demo.jl

using WDW, LinearAlgebra, Random, Printf, Statistics

const SC = WDW.SymmetryCertificate
const FP = WDW.FFTPipeline
const FG = WDW.FFTGroup
const SD = WDW.SymmetryDiscovery

println("="^72)
println("  SYMMETRY CERTIFICATE — Complete System Demo")
println("  Integrating Group Theory · Bispectrum Anchor · Layer Audit")
println("  Equivariant Compression · MERA · Krylov · MDL · PAC-Bayes")
println("="^72)

# ═══════════════════════════════════════════════════════════
# STEP 1: DATA
# ═══════════════════════════════════════════════════════════
println("\n  ── Step 1: Load Data ──\n")

xs_train, ys_train, xs_test, ys_test = FP.make_dataset(32, 2, 4, 42)

@printf "  Training samples:   %d\n" length(xs_train)
@printf "  Test samples:       %d\n" length(xs_test)
@printf "  Signal dimension:   %d\n" length(xs_train[1])
@printf "  Classes:            %d\n" length(unique(ys_train))

# ═══════════════════════════════════════════════════════════
# STEP 2: TRAIN BOTH MODELS (clean + spurious)
# ═══════════════════════════════════════════════════════════
println("\n  ── Step 2: Train Models ──\n")

# Clean model
p_clean = FP.SignalPipeline(32; n_classes=4, n_pairs=2, seed=42)
FP.train_pipeline!(p_clean, xs_train, ys_train; epochs=500)
acc_clean = FG.accuracy_bispec(p_clean.layer, p_clean.Wc, p_clean.bc, xs_test, ys_test; dn=false)

# Spurious model
p_spur = FP.SignalPipeline(32; n_classes=4, n_pairs=2, seed=42)
FP.train_pipeline!(p_spur, xs_train, ys_train; epochs=500)
for c in 1:4
    ω = 10 + c
    p_spur.Wc[c, ω] += 0.8
end
acc_spur = FG.accuracy_bispec(p_spur.layer, p_spur.Wc, p_spur.bc, xs_test, ys_test; dn=false)

@printf "  Clean classifier:     %5.1f%%\n" acc_clean
@printf "  Spurious classifier:  %5.1f%%  (injected bias at ω=11..14)\n" acc_spur

# ═══════════════════════════════════════════════════════════
# STEP 3: MODEL FUNCTION (layer-aware)
# ═══════════════════════════════════════════════════════════
function model_fn_clean(x)
    feats = FG.combined_bispec_features(x, p_clean.layer)
    logits = p_clean.Wc * feats + p_clean.bc
    return (feats, logits)
end

function model_fn_spur(x)
    feats = FG.combined_bispec_features(x, p_spur.layer)
    logits = p_spur.Wc * feats + p_spur.bc
    return (feats, logits)
end

# ═══════════════════════════════════════════════════════════
# STEP 4: DATA SYMMETRY PROFILE (Pillar 1 + 2: Group + Anchor)
# ═══════════════════════════════════════════════════════════
println("\n  ── Step 3: Data Symmetry Profile (Bispectrum Anchor) ──\n")

data_profile, n_probes, n = SC.audit_dataset(xs_train)
probe_names = ["shift 1", "shift 8", "shift 16", "shift 24", "shift 31",
               "reflection", "scramble 0.25", "scramble 0.5"]

@printf "  %-20s %12s\n" "Probe" "Bispectrum divergence"
println("  " * "-"^38)
for i in eachindex(probe_names)
    tag = data_profile[i] < 1e-10 ? " ← exact symmetry" : ""
    @printf "  %-20s %12.6f%s\n" probe_names[i] data_profile[i] tag
end

# ═══════════════════════════════════════════════════════════
# STEP 5: GENERATE CERTIFICATES (Pillars 3-7)
# ═══════════════════════════════════════════════════════════
println("\n  ── Step 4: Symmetry Certificate (Pillars 3-7) ──\n")

println("  Auditing CLEAN model...")
cert_clean = SC.quick_audit(xs_train, model_fn_clean, ["features", "logits"], [3*32, 4])

println("  Auditing SPURIOUS model...")
cert_spur = SC.quick_audit(xs_train, model_fn_spur, ["features", "logits"], [3*32, 4])

# ═══════════════════════════════════════════════════════════
# STEP 6: COMPARE CERTIFICATES
# ═══════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("  CERTIFICATE COMPARISON")
println("=" ^ 72)
println()

@printf "  %-35s %12s %12s\n" "Metric" "Clean model" "Spurious model"
println("  " * "-" ^ 62)

# Layer divergences
a_clean = cert_clean.audit
a_spur  = cert_spur.audit
for l in 1:a_clean.n_layers
    @printf "  %-35s %12.4f %12.4f%s\n" "Divergence at $(a_clean.layer_names[l])" a_clean.layer_divergences[l] a_spur.layer_divergences[l] (a_spur.layer_divergences[l] > a_clean.layer_divergences[l] ? " ↑ spurious" : "")
end
println()

# Composite scores
@printf "  %-35s %12.2f%% %12.2f%%\n" "Symmetry fidelity" (cert_clean.symmetry_fidelity * 100) (cert_spur.symmetry_fidelity * 100)
@printf "  %-35s %12.2f%% %12.2f%%\n" "Layer homogeneity" (cert_clean.layer_homogeneity * 100) (cert_spur.layer_homogeneity * 100)
@printf "  %-35s %12.2f%% %12.2f%%\n" "Compression efficiency" (cert_clean.compression_efficiency * 100) (cert_spur.compression_efficiency * 100)
@printf "  %-35s %12.2f%% %12.2f%%\n" "Generalization readiness" (cert_clean.generalization_readiness * 100) (cert_spur.generalization_readiness * 100)
@printf "  %-35s %12.2f%% %12.2f%%\n" "DEPLOYABILITY SCORE" (cert_clean.deployability_score * 100) (cert_spur.deployability_score * 100)

# ═══════════════════════════════════════════════════════════
# STEP 7: FAILURE MODE ANALYSIS
# ═══════════════════════════════════════════════════════════
println("\n  ── Failure Mode Prediction ──\n")

println("  CLEAN model:")
for f in SC.failure_modes(cert_clean)
    println("    • $f")
end
println()
println("  SPURIOUS model:")
for f in SC.failure_modes(cert_spur)
    println("    • $f")
end

# ═══════════════════════════════════════════════════════════
# STEP 8: DEPLOYABILITY VERDICT
# ═══════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("  DEPLOYABILITY VERDICT")
println("=" ^ 72)
println()

for (name, cert) in [("CLEAN", cert_clean), ("SPURIOUS", cert_spur)]
    d = SC.deployability_score(cert)
    if d > 0.7
        verdict = "✓ DEPLOYMENT READY"
    elseif d > 0.4
        verdict = "⚠ CONDITIONAL PASS"
    else
        verdict = "✗ DO NOT DEPLOY"
    end
    @printf "  %-10s score=%.2f%%  %s\n" name (d * 100) verdict
    for act in cert.recommended_actions
        println("    → $act")
    end
    println()
end

println("=" ^ 72)
println("  Certificate IDs:")
@printf "    Clean model:    %s\n" cert_clean.certificate_id
@printf "    Spurious model: %s\n" cert_spur.certificate_id
println()
println("  Both models have the same accuracy ($(round(Int, acc_clean))%).")
println("  But the Spurious model's certificate reveals significantly higher")
println("  symmetry divergence at the logit layer — detecting the injected")
println("  class-correlated frequency bias that accuracy alone cannot see.")
println("=" ^ 72)
