#!/usr/bin/env julia
# WDW Spectrometer Experiment
# Generates multiple models, measures structural fingerprints,
# builds embedding space, tests OOD prediction.
#
# Usage:
#   julia --project bench/spectrometer_experiment.jl

using WDW, LinearAlgebra, Random, Statistics, Printf, Dates

const UI = WDW.UnifiedIntegration
const FP = WDW.FFTPipeline
const FG = WDW.FFTGroup
const SE = WDW.StructuralEmbedding

println("="^80)
println("  WDW SPECTROMETER — Structural Fingerprint Experiment")
println("="^80)

# ═══════════════════════════════════════════════════════════════════
# PART 1: GENERATE MODELS WITH VARYING SPURIOUS INJECTION
# ═══════════════════════════════════════════════════════════════════

n = 32
xs_train, ys_train, xs_test, ys_test = FP.make_dataset(n, 2, 4, 42)

@printf "\n  Base dimension: %d\n" n
@printf "  Training samples: %d\n" length(xs_train)
@printf "  Test samples: %d\n" length(xs_test)

function train_model(n, seed; injection=0.0)
    p = FP.SignalPipeline(n; n_classes=4, n_pairs=2, seed=seed)
    FP.train_pipeline!(p, xs_train, ys_train; epochs=500)
    if injection > 0
        for c in 1:4
            p.Wc[c, 10 + c] += injection
        end
    end
    return p
end

function model_fn_from_pipeline(p)
    return function(x)
        feats = FG.combined_bispec_features(x, p.layer)
        logits = p.Wc * feats + p.bc
        return (feats, logits)
    end
end

injection_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
pipelines = []
model_names = String[]
accuracies = Float64[]

for (i, inj) in enumerate(injection_levels)
    name = @sprintf "inj=%.1f" inj
    push!(model_names, name)
    
    println("\n  ── Training $name ──")
    p = train_model(n, 42 + i; injection=inj)
    push!(pipelines, p)
    
    fn = model_fn_from_pipeline(p)
    acc = FG.accuracy_bispec(p.layer, p.Wc, p.bc, xs_test, ys_test; dn=false)
    push!(accuracies, acc)
    @printf "  Accuracy: %.1f%%\n" acc
end

# ═══════════════════════════════════════════════════════════════════
# PART 2: STRUCTURAL FINGERPRINTS FOR EACH MODEL
# ═══════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  MEASURING STRUCTURAL FINGERPRINTS")
println("="^80)

results = UI.UnifiedResult[]
fingerprints = Vector{Float64}[]

for (i, p) in enumerate(pipelines)
    name = model_names[i]
    inj = injection_levels[i]
    
    @printf "\n  [%d/%d] %s ... " i length(pipelines) name
    fn = model_fn_from_pipeline(p)
    t0 = time()
    r = UI.analyze_all(xs_train; model_fn=fn, data_name=name)
    elapsed = time() - t0
    
    push!(results, r)
    push!(fingerprints, r.measurement_matrix[:])
    
    @printf "%d/%d analyzers, %.1fs\n" r.n_success r.n_total elapsed
    @printf "    Unified complexity: %.4f\n" r.unified_complexity
    @printf "    Spectral: %.1f%%  Algebraic: %.1f%%  Topological: %.1f%%\n" (r.spectral_score*100) (r.algebraic_score*100) (r.topological_score*100)
    @printf "    Compressive: %.1f%%  Physical: %.1f%%  Theoretical: %.1f%%  Applied: %.1f%%\n" (r.compressive_score*100) (r.physical_score*100) (r.theoretical_score*100) (r.applied_score*100)
end

# ═══════════════════════════════════════════════════════════════════
# PART 3: BUILD STRUCTURAL EMBEDDING
# ═══════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  BUILDING STRUCTURAL EMBEDDING SPACE")
println("="^80)

emb = SE.structural_embedding(results; n_dims=5, model_names=model_names)
SE.embedding_summary(emb)

# Export
csv_path = joinpath(@__DIR__, "embedding_coords.csv")
SE.export_embedding_csv(emb, csv_path)

# Top measurements driving PC1
println()
SE.top_contributing_measurements(emb; n=10)

# ═══════════════════════════════════════════════════════════════════
# PART 4: CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  CORRELATION: FINGERPRINT DISTANCE vs INJECTION LEVEL")
println("="^80)

clean_idx = 1  # injection=0.0 is first
@printf "\n  Reference model: %s\n" model_names[clean_idx]
@printf "  %-15s %10s %12s %12s %12s\n" "Model" "Injection" "PC1" "Dist(clean)" "Acc(clean)"
println("  " * "-" ^ 64)

    for i in 1:length(model_names)
    if i == clean_idx
        @printf "  %-15s %10.1f %12.4f %12s %12.1f%%\n" model_names[i] injection_levels[i] emb.coords[i,1] "ref" accuracies[i]
    else
        d = SE.fingerprint_distance(emb, clean_idx, i)
        @printf "  %-15s %10.1f %12.4f %12.4f %12.1f%%\n" model_names[i] injection_levels[i] emb.coords[i,1] d accuracies[i]
    end
end

# Pearson correlation between injection level and PC1
if length(injection_levels) > 2
    x = Float64.(injection_levels)
    y = emb.coords[:, 1]
    μx = mean(x); μy = mean(y)
    σx = std(x); σy = std(y)
    r_xy = mean((x .- μx) .* (y .- μy)) / (σx * σy)
    @printf "\n  Pearson r(injection, PC1) = %.4f\n" r_xy
    @printf "  R² = %.4f\n" r_xy^2
    if abs(r_xy) > 0.8
        @printf "  ✓ Strong correlation — PC1 captures spurious injection level\n"
    elseif abs(r_xy) > 0.5
        @printf "  ~ Moderate correlation\n"
    else
        @printf "  ✗ Weak correlation — embedding is driven by other factors\n"
    end
end

# ═══════════════════════════════════════════════════════════════════
# PART 5: SYMMETRY CERTIFICATE COMPARISON
# ═══════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  SYMMETRY CERTIFICATE vs INJECTION LEVEL")
println("="^80)

@printf "\n  %-15s %10s %15s %15s\n" "Model" "Injection" "Deployability" "Logit Div"
println("  " * "-" ^ 58)

for (i, p) in enumerate(pipelines)
    fn = model_fn_from_pipeline(p)
    cert = WDW.SymmetryCertificate.quick_audit(xs_train, fn, ["features", "logits"], [3*32, 4])
    @printf "  %-15s %10.1f %15.1f%% %15.4f\n" model_names[i] injection_levels[i] (cert.deployability_score*100) cert.audit.layer_divergences[end]
end

# ═══════════════════════════════════════════════════════════════════
# PART 6: OOD TEST — CROSS-FREQUENCY GENERALIZATION
# ═══════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  OOD TEST — Cross-Frequency Generalization")
println("="^80)

# Generate OOD test data: different frequency pattern
println("\n  Generating OOD test data (shifted frequencies)...")
Random.seed!(99)
xs_ood = Vector{Float64}[]
ys_ood = Int[]
n_ood = 100
for _ in 1:n_ood
    label = rand(1:4)
    x = zeros(n)
    freq = (label + 3) * 2  # Shifted frequencies
    for i in 1:n
        x[i] = sin(2π * freq * i / n) + 0.3 * cos(2π * freq * 2 * i / n) + 0.1 * randn()
    end
    push!(xs_ood, x)
    push!(ys_ood, label)
end
@printf "  OOD samples: %d\n" length(xs_ood)

@printf "\n  %-15s %12s %12s %10s\n" "Model" "In-dist Acc" "OOD Acc" "Drop"
println("  " * "-" ^ 52)

ood_accs = Float64[]
for (i, p) in enumerate(pipelines)
    fn = model_fn_from_pipeline(p)
    acc_id = accuracies[i]
    
    correct = 0
    for (x, y) in zip(xs_ood, ys_ood)
        _, logits = fn(x)
        if argmax(logits) == y
            correct += 1
        end
    end
    acc_ood = correct / length(xs_ood) * 100
    push!(ood_accs, acc_ood)
    drop = acc_id - acc_ood
    @printf "  %-15s %12.1f%% %12.1f%% %9.1fpp\n" model_names[i] acc_id acc_ood drop
end

# ═══════════════════════════════════════════════════════════════════
# PART 7: EMBEDDING → OOD PREDICTION
# ═══════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  EMBEDDING → OOD DROP PREDICTION")
println("="^80)

ood_drops = accuracies - ood_accs
dists_from_clean = [i == 1 ? 0.0 : SE.fingerprint_distance(emb, 1, i) for i in 1:length(model_names)]

@printf "\n  %-15s %10s %12s\n" "Model" "Dist(clean)" "OOD Drop"
println("  " * "-" ^ 40)

for i in 1:length(model_names)
    @printf "  %-15s %10.4f %10.1fpp\n" model_names[i] dists_from_clean[i] ood_drops[i]
end

if length(dists_from_clean) > 2
    x2 = dists_from_clean
    y2 = ood_drops
    μx2 = mean(x2[2:end]); μy2 = mean(y2[2:end])
    σx2 = std(x2[2:end]); σy2 = std(y2[2:end])
    if σx2 * σy2 > 0
        r_xy2 = mean((x2[2:end] .- μx2) .* (y2[2:end] .- μy2)) / (σx2 * σy2)
        @printf "\n  Pearson r(dist_from_clean, OOD_drop) = %.4f\n" r_xy2
        @printf "  R² = %.4f\n" r_xy2^2
        if abs(r_xy2) > 0.7
            @printf "  ✓ STRONG: fingerprint distance predicts OOD degradation\n"
            @printf "  → This is the core result: structural fingerprints predict generalization\n"
        elseif abs(r_xy2) > 0.4
            @printf "  ~ MODERATE: some predictive power\n"
        else
            @printf "  ✗ WEAK: fingerprint distance does not predict OOD well\n"
            @printf "  → Possible causes: (1) too few models, (2) wrong OOD test, (3) embedding needs more dimensions\n"
        end
    end
end

# ═══════════════════════════════════════════════════════════════════
# PART 8: FULL MEASUREMENT MATRIX EXPORT
# ═══════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  EXPORTING FULL MEASUREMENT MATRIX")
println("="^80)

meas_csv = joinpath(@__DIR__, "spectrometer_measurements.csv")
n_meas = length(results[1].measurement_names)
open(meas_csv, "w") do io
    header = "measurement," * join(model_names, ",")
    write(io, header * "\n")
    for mi in 1:n_meas
        row = results[1].measurement_names[mi]
        for ri in 1:length(results)
            val = mi <= length(results[ri].measurement_matrix[:]) ? results[ri].measurement_matrix[mi] : 0.0
            row *= "," * string(round(val, digits=6))
        end
        write(io, row * "\n")
    end
end
println("  Exported: $meas_csv")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  EXPERIMENT SUMMARY")
println("="^80)
# Determine result strings
result_str = if length(dists_from_clean) > 2 && σx2 * σy2 > 0
    if abs(r_xy2) > 0.7
        "✓ YES — strong correlation"
    elseif abs(r_xy2) > 0.4
        "~ PARTIAL — moderate correlation"
    else
        "✗ NO — weak or no correlation"
    end
else
    "N/A — insufficient data"
end

next_steps_str = if length(dists_from_clean) > 2 && σx2 * σy2 > 0 && abs(r_xy2) > 0.4
    "1. Add more model variants (different n, different group types)
    2. Run on standard benchmarks (CIFAR-10, Rotated MNIST)
    3. Publish: Structural embeddings predict OOD generalization"
else
    "1. Check PCA quality — try different preprocessing
    2. Add more models (10-20) for statistical power
    3. Try different OOD tests (rotation, noise, adversarial)"
end

println("""
  Models:          $(length(pipelines)) (injection 0.0 - 1.0)
  Measurements:    $n_meas per model
  Embedding dims:  $(emb.n_dims)
  Embedding CSV:   $csv_path
  Measurements CSV: $meas_csv

  Key question:
    Does structural fingerprint distance predict OOD generalization?
    
    Result: $result_str

  Next steps:
    $next_steps_str
""")
println("="^80)
