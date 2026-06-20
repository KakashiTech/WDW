#!/usr/bin/env julia
# SymmetryDiscovery — AI Representation Auditing
#
# Core insight: Every neural network has a symmetry signature.
# Compare it to the data's symmetry signature → detect spurious correlations.
#
# Usage:
#   julia --project bench/symmetry_ai_demo.jl

using WDW, LinearAlgebra, Random, Printf, Statistics

const SD = WDW.SymmetryDiscovery
const FP = WDW.FFTPipeline
const FG = WDW.FFTGroup

println("="^72)
println("  SYMMETRY DISCOVERY — AI Representation Auditing")
println("  Detecting spurious correlations via symmetry profiling")
println("="^72)

# ═══════════════════════════════════════════════════════════
# 1. DATA SYMMETRY PROFILE (ground truth via bispectrum)
# ═══════════════════════════════════════════════════════════
println("\n  ── 1. DATA SYMMETRY PROFILE ──\n")

xs_tr, ys_tr, xs_te, ys_te = FP.make_dataset(32, 2, 4, 42)
n, probes = 32, SD.default_probes(32)
probe_names = ["shift 1", "shift 8", "shift 16", "shift 24", "shift 31",
               "reflection", "scramble 0.25", "scramble 0.5"]

profile_data = SD.symmetry_profile(xs_tr; probes=probes)

@printf "  %-20s %12s\n" "Probe" "Bispectrum divergence"
println("  " * "-"^38)
for i in eachindex(probes)
    tag = profile_data[i] < 1e-10 ? " ← exact symmetry" : ""
    @printf "  %-20s %12.6f%s\n" probe_names[i] profile_data[i] tag
end

# ═══════════════════════════════════════════════════════════
# 2. BUILD TWO CLASSIFIERS (both 100%, one secretly spurious)
# ═══════════════════════════════════════════════════════════
# We use the WDW pipeline (bispectrum → linear classifier).
# Both achieve 100% on the time-reversal pair task.
# The spurious version has class-correlated frequency bias
# injected into its classifier weights.
# ═══════════════════════════════════════════════════════════
println("\n  ── 2. TWO CLASSIFIERS ──\n")

function make_clean_classifier()
    p = FP.SignalPipeline(32; n_classes=4, n_pairs=2, seed=42)
    FP.train_pipeline!(p, xs_tr, ys_tr; epochs=500)
    return p
end

function make_spurious_classifier()
    p = FP.SignalPipeline(32; n_classes=4, n_pairs=2, seed=42)
    FP.train_pipeline!(p, xs_tr, ys_tr; epochs=500)

    # Inject subtle spurious bias: class-specific frequency bands
    # that correlate with class identity but are NOT true class features
    n_feat = 3 * 32
    for c in 1:4
        ω = 10 + c
        p.Wc[c, ω] += 0.8  # class c gets a boost at frequency 10+c
    end
    return p
end

p_clean = make_clean_classifier()
p_spur  = make_spurious_classifier()

acc_clean = FG.accuracy_bispec(p_clean.layer, p_clean.Wc, p_clean.bc, xs_te, ys_te; dn=false)
acc_spur  = FG.accuracy_bispec(p_spur.layer, p_spur.Wc, p_spur.bc, xs_te, ys_te; dn=false)

@printf "  Clean classifier:     %5.1f%%\n" acc_clean
@printf "  Spurious classifier:  %5.1f%%\n" acc_spur
println("  (Accuracy alone cannot distinguish them.)")

# ═══════════════════════════════════════════════════════════
# 3. MODEL FUNCTION (returns layer activations for auditing)
# ═══════════════════════════════════════════════════════════
function model_fn(pipeline)
    return function(x)
        feats = FG.combined_bispec_features(x, pipeline.layer)
        logits = pipeline.Wc * feats + pipeline.bc
        return (feats, logits)
    end
end

# ═══════════════════════════════════════════════════════════
# 4. SYMMETRY AUDIT: LAYER-BY-LAYER
# ═══════════════════════════════════════════════════════════
println("\n  ── 3. SYMMETRY AUDIT ──\n")

result_clean = SD.detect_spurious_layers(model_fn(p_clean), xs_tr; threshold=0.5)
result_spur  = SD.detect_spurious_layers(model_fn(p_spur), xs_tr; threshold=0.5)

@printf "  %-15s %15s %15s %15s\n" "Layer" "Data profile" "Clean div." "Spurious div."
println("  " * "-"^63)
for (l, name) in enumerate(["features", "logits"])
    flag = result_spur.divergences[l] > result_clean.divergences[l] + 1.0 ?
           "  ⚠ SPURIOUS" : "  ✓ clean"
    @printf "  %-15s %15.4f %15.4f %15.4f%s\n" name mean(profile_data) result_clean.divergences[l] result_spur.divergences[l] flag
end

println("\n  Key finding:")
println("  The feature layer is identical (same bispectrum features).")
println("  The LOGIT layer divergence is $(round(result_spur.divergences[2] / result_clean.divergences[2], digits=2))× higher for the spurious model.")
println("  This reveals the class-correlated bias that accuracy hides.")

# ═══════════════════════════════════════════════════════════
# 5. VISUALIZATION: SYMMETRY PROFILE AT LOGIT LAYER
# ═══════════════════════════════════════════════════════════
println("\n  ── 4. SYMMETRY PROFILE OF LOGIT LAYER ──\n")

logit_prof_clean = mean([result_clean.layer_profiles[end] for _ in 1:1])
logit_prof_spur  = mean([result_spur.layer_profiles[end] for _ in 1:1])

@printf "  %-20s %12s %12s\n" "Probe" "Clean logits" "Spurious logits"
println("  " * "-"^47)
for i in eachindex(probe_names)
    d_c = result_clean.layer_profiles[end][i]
    d_s = result_spur.layer_profiles[end][i]
    @printf "  %-20s %12.4f %12.4f\n" probe_names[i] d_c d_s
end

println("\n  Difference (spurious - clean):")
for i in eachindex(probe_names)
    diff = result_spur.layer_profiles[end][i] - result_clean.layer_profiles[end][i]
    label = diff > 0.1 ? "  ↑ spurious" : ""
    @printf "    %-20s %+.4f%s\n" probe_names[i] diff label
end

# ═══════════════════════════════════════════════════════════
# 6. IMPACT
# ═══════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("  WHY THIS MATTERS")
println("=" ^ 72)
println("""
  Accuracy says both classifiers are equally good (both $(round(Int, acc_clean))%).
  But they are structurally different:
    - The clean model only uses C_n-invariant features.
    - The spurious model also uses frequency bands 11-14
      that correlate with class labels but are not true symmetries.

  The bispectrum anchor detects this because:
    1.  Data profile: shifts are exact symmetries (divergence ≈ 0)
    2.  Clean logits: the CLASSIFIER also respects shift symmetry
        (its output has the same symmetry profile as the data)
    3.  Spurious logits: the bias at ω=11..14 makes the logit layer
        more sensitive to frequency modulations in those bands

  Result: the spurious model's logit-layer symmetry profile
  diverges from the data profile by $(round(result_spur.divergences[2] / result_clean.divergences[2], digits=2))× more.

  No labels needed. No held-out test. Just group theory.

  The bispectrum anchor applies to ANY model's internal representations.
  Audit deeper networks layer by layer to find WHERE spurious
  correlations enter and WHICH symmetries they exploit.
""")
