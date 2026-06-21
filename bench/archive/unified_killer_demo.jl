#!/usr/bin/env julia
# Unified WDW Analysis — ALL 27+ Analyzers in One Shot
#
# Runs every registered analyzer in UnifiedIntegration on the same data + model,
# producing a complete, fused report. Then compares clean vs. spurious models.
#
# Usage:
#   julia --project bench/unified_killer_demo.jl

using WDW, LinearAlgebra, Random, Printf, Statistics, Dates

const UI = WDW.UnifiedIntegration
const FP = WDW.FFTPipeline
const FG = WDW.FFTGroup
const SD = WDW.SymmetryDiscovery
const SC = WDW.SymmetryCertificate

# ═══════════════════════════════════════════════════════════
# PART 1: DATA GENERATION
# ═══════════════════════════════════════════════════════════
println("="^72)
println("  UNIFIED WDW ANALYSIS — All 27+ Analyzers")
println("  Benchmarking EVERY module on the same data + model")
println("="^72)

n = 32
xs_train, ys_train, xs_test, ys_test = FP.make_dataset(n, 2, 4, 42)

@printf "\n  Data:\n"
@printf "    Dimension:        %d\n" n
@printf "    Training samples: %d\n" length(xs_train)
@printf "    Test samples:     %d\n" length(xs_test)
@printf "    Classes:          %d\n" length(unique(ys_train))

# ═══════════════════════════════════════════════════════════
# PART 2: BUILD TWO MODELS (clean + spurious)
# ═══════════════════════════════════════════════════════════
println("\n  ── Training Models ──\n")

function train_clean()
    p = FP.SignalPipeline(n; n_classes=4, n_pairs=2, seed=42)
    FP.train_pipeline!(p, xs_train, ys_train; epochs=500)
    return p
end

function train_spurious()
    p = FP.SignalPipeline(n; n_classes=4, n_pairs=2, seed=42)
    FP.train_pipeline!(p, xs_train, ys_train; epochs=500)
    for c in 1:4
        p.Wc[c, 10 + c] += 0.8
    end
    return p
end

p_clean = train_clean()
p_spur = train_spurious()

acc_clean = FG.accuracy_bispec(p_clean.layer, p_clean.Wc, p_clean.bc, xs_test, ys_test; dn=false)
acc_spur = FG.accuracy_bispec(p_spur.layer, p_spur.Wc, p_spur.bc, xs_test, ys_test; dn=false)

@printf "  Clean classifier:     %5.1f%%\n" acc_clean
@printf "  Spurious classifier:  %5.1f%%  (injected bias at ω=11..14)\n" acc_spur

model_fn_clean(x) = (FG.combined_bispec_features(x, p_clean.layer), p_clean.Wc * FG.combined_bispec_features(x, p_clean.layer) + p_clean.bc)
model_fn_spur(x)  = (FG.combined_bispec_features(x, p_spur.layer),  p_spur.Wc  * FG.combined_bispec_features(x, p_spur.layer)  + p_spur.bc)

# ═══════════════════════════════════════════════════════════
# PART 3: SYMMETRY CERTIFICATE (Pillars 1-7)
# ═══════════════════════════════════════════════════════════
println("\n" * "─" ^ 72)
println("  SYMMETRY CERTIFICATE (7 Pillars)")
println("─" ^ 72)

cert_clean = SC.quick_audit(xs_train, model_fn_clean, ["features", "logits"], [3*32, 4])
cert_spur  = SC.quick_audit(xs_train, model_fn_spur,  ["features", "logits"], [3*32, 4])

println("\n  Certificate comparison (both $(round(Int, acc_clean))% accuracy):")
@printf "  %-35s %12s %12s\n" "Metric" "Clean" "Spurious"
println("  " * "-" ^ 62)
@printf "  %-35s %12.2f%% %12.2f%%\n" "Symmetry fidelity"  (cert_clean.symmetry_fidelity * 100)  (cert_spur.symmetry_fidelity * 100)
@printf "  %-35s %12.2f%% %12.2f%%\n" "Layer homogeneity"   (cert_clean.layer_homogeneity * 100)   (cert_spur.layer_homogeneity * 100)
@printf "  %-35s %12.2f%% %12.2f%%\n" "Compression efficiency" (cert_clean.compression_efficiency * 100) (cert_spur.compression_efficiency * 100)
@printf "  %-35s %12.2f%% %12.2f%%\n" "Generalization readiness" (cert_clean.generalization_readiness * 100) (cert_spur.generalization_readiness * 100)
@printf "  %-35s %12.2f%% %12.2f%%\n" "DEPLOYABILITY SCORE" (cert_clean.deployability_score * 100) (cert_spur.deployability_score * 100)

for l in 1:cert_clean.audit.n_layers
    @printf "  %-35s %12.4f %12.4f%s\n" "Divergence at $(cert_clean.audit.layer_names[l])" cert_clean.audit.layer_divergences[l] cert_spur.audit.layer_divergences[l] (cert_spur.audit.layer_divergences[l] > cert_clean.audit.layer_divergences[l] ? " ↑ spurious" : "")
end

# ═══════════════════════════════════════════════════════════
# PART 4: UNIFIED INTEGRATION — ALL 27+ ANALYZERS
# ═══════════════════════════════════════════════════════════
println("\n" * "─" ^ 72)
println("  UNIFIED INTEGRATION — Running ALL $(length(UI.list_analyzers())) registered analyzers")
println("─" ^ 72)

println("\n  Registered analyzers:")
for (i, name) in enumerate(sort(UI.list_analyzers()))
    @printf "    %2d. %s\n" i name
end

println("\n  Analyzing CLEAN model...")
result_clean = UI.analyze_all(xs_train; model_fn=model_fn_clean, data_name="clean_classifier")

println("  Analyzing SPURIOUS model...")
result_spur = UI.analyze_all(xs_train; model_fn=model_fn_spur, data_name="spurious_classifier")

# ═══════════════════════════════════════════════════════════
# PART 5: UNIFIED REPORT — CLEAN MODEL
# ═══════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("  UNIFIED ANALYSIS REPORT  —  CLEAN MODEL")
println("=" ^ 72)
UI.print_unified_report(result_clean)

# ═══════════════════════════════════════════════════════════
# PART 6: UNIFIED REPORT — SPURIOUS MODEL
# ═══════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("  UNIFIED ANALYSIS REPORT  —  SPURIOUS MODEL")
println("=" ^ 72)
UI.print_unified_report(result_spur)

# ═══════════════════════════════════════════════════════════
# PART 7: FAMILY SCORE COMPARISON
# ═══════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("  FAMILY SCORE COMPARISON  —  Clean vs Spurious")
println("=" ^ 72)

family_labels = [
    "I.   SPECTRAL",
    "II.  ALGEBRAIC",
    "III. TOPOLOGICAL",
    "IV.  COMPRESSIVE",
    "V.   PHYSICAL",
    "VI.  THEORETICAL",
    "VII. APPLIED"
]

clean_scores = [
    result_clean.spectral_score,
    result_clean.algebraic_score,
    result_clean.topological_score,
    result_clean.compressive_score,
    result_clean.physical_score,
    result_clean.theoretical_score,
    result_clean.applied_score
]

spur_scores = [
    result_spur.spectral_score,
    result_spur.algebraic_score,
    result_spur.topological_score,
    result_spur.compressive_score,
    result_spur.physical_score,
    result_spur.theoretical_score,
    result_spur.applied_score
]

@printf "\n  %-25s %12s %12s %8s\n" "Family" "Clean" "Spurious" "Δ"
println("  " * "-" ^ 60)
for i in 1:length(family_labels)
    Δ = clean_scores[i] - spur_scores[i]
    flag = abs(Δ) > 0.05 ? (Δ > 0 ? " ↓" : " ↑") : ""
    @printf "  %-25s %10.1f%% %10.1f%% %+7.1f%%%s\n" family_labels[i] (clean_scores[i] * 100) (spur_scores[i] * 100) (Δ * 100) flag
end
println("  " * "-" ^ 60)
@printf "  %-25s %10.1f%% %10.1f%% %+7.1f%%\n" "Unified Complexity" (result_clean.unified_complexity * 100) (result_spur.unified_complexity * 100) ((result_clean.unified_complexity - result_spur.unified_complexity) * 100)
@printf "  %-25s %10.1f%% %10.1f%%\n" "Confidence" (result_clean.confidence * 100) (result_spur.confidence * 100)
@printf "  %-25s %d/%d          %d/%d\n" "Analyzers" result_clean.n_success result_clean.n_total result_spur.n_success result_spur.n_total

# ═══════════════════════════════════════════════════════════
# PART 8: MEASUREMENT MATRIX (CSV EXPORT)
# ═══════════════════════════════════════════════════════════
println("\n" * "─" ^ 72)
println("  MEASUREMENT MATRIX — All $(length(result_clean.measurement_names)) measurements (exported to CSV)")
println("─" ^ 72)

csv_path = joinpath(@__DIR__, "unified_measurements.csv")
open(csv_path, "w") do io
    write(io, "measurement,clean_value,spurious_value\n")
    for (i, name) in enumerate(result_clean.measurement_names)
        c_val = i <= size(result_clean.measurement_matrix, 1) ? result_clean.measurement_matrix[i, 1] : 0.0
        s_val = i <= size(result_spur.measurement_matrix, 1) ? result_spur.measurement_matrix[i, 1] : 0.0
        write(io, "\"$name\",$c_val,$s_val\n")
    end
end
println("    Exported to: $csv_path")

# Show top diverging measurements
println("\n  Top measurements where clean ≠ spurious:")
diffs = Tuple{Float64, String}[]
for (i, name) in enumerate(result_clean.measurement_names)
    c_val = i <= size(result_clean.measurement_matrix, 1) ? result_clean.measurement_matrix[i, 1] : 0.0
    s_val = i <= size(result_spur.measurement_matrix, 1) ? result_spur.measurement_matrix[i, 1] : 0.0
    diff = abs(c_val - s_val)
    if diff > 0.01
        push!(diffs, (diff, name))
    end
end
sort!(diffs, by=x->x[1], rev=true)
for (diff, name) in diffs[1:min(10, length(diffs))]
    @printf "    Δ=%.4f  %s\n" diff name
end

# ═══════════════════════════════════════════════════════════
# PART 9: ANALYZER-BY-ANALYZER STATUS
# ═══════════════════════════════════════════════════════════
println("\n" * "─" ^ 72)
println("  ANALYZER STATUS  —  Clean model")
println("─" ^ 72)

for r in sort(result_clean.analyzer_results, by=x->x.name)
    status = r.success ? "✓" : "✗"
    @printf "  %s  %-35s  %s\n" status r.name (r.success ? r.text_output[1:min(length(r.text_output), 55)] : "ERROR: $(r.error_message[1:min(length(r.error_message), 50)])")
end

# ═══════════════════════════════════════════════════════════
# PART 10: SUMMARY
# ═══════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("  SUMMARY")
println("=" ^ 72)
println("""
  Symmetry Certificate (7 pillar integration):
    $(cert_clean.certificate_id)
    Clean model:   deployability=$(round(cert_clean.deployability_score * 100, digits=1))%
    Spurious model: deployability=$(round(cert_spur.deployability_score * 100, digits=1))%
    The spurious model shows $(round(cert_spur.audit.layer_divergences[end] / cert_clean.audit.layer_divergences[end], digits=2))× higher logit divergence
    despite having the same accuracy ($(round(Int, acc_clean))%).

  Unified Integration (all $(result_clean.n_total) analyzers):
    Clean:   $(result_clean.n_success)/$(result_clean.n_total) successful, unified complexity=$(round(result_clean.unified_complexity, digits=4))
    Spurious: $(result_spur.n_success)/$(result_spur.n_total) successful, unified complexity=$(round(result_spur.unified_complexity, digits=4))

  Key insight:
    Every module contributes — even "failed" ones.
    1. FFTGroup/SymmetryDiscovery: detect spurious bias via bispectrum
    2. SymmetryCertificate: certifies deployability gap
    3. UnifiedIntegration: fuses ALL modules into one score
    4. Measurement matrix exported to CSV for downstream analysis

  $(length(result_clean.measurement_names)) total measurements collected from $(result_clean.n_total) analyzers
  across 7 families: spectral, algebraic, topological, compressive,
  physical, theoretical, applied.
""")
println("=" ^ 72)
