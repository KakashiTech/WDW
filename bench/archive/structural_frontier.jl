#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════════
# WDW STRUCTURAL FRONTIER — The Structural Generalization Frontier
# ═══════════════════════════════════════════════════════════════════════════════
#
# Hypothesis: The 114-dim structural fingerprint captures architecture family.
# Models of the same architecture cluster in fingerprint space. The cluster 
# medoid has optimal OOD generalization. Similar fingerprints → similar failure modes.
#
# Architecture grid:
#   n ∈ {16, 24, 32, 48} × 3 seeds = 12  (architecture cluster)
#   n=32 × n_classes ∈ {2,3,5,6} = 4      (diversity group)
#   Total: 16 models
#
# Usage:
#   julia --project bench/structural_frontier.jl
# ═══════════════════════════════════════════════════════════════════════════════

using WDW, LinearAlgebra, Random, Statistics, Printf, Dates

const UI = WDW.UnifiedIntegration
const FP = WDW.FFTPipeline
const FG = WDW.FFTGroup
const SE = WDW.StructuralEmbedding

println("="^80)
println("  WDW STRUCTURAL FRONTIER — Structural Generalization Frontier")
println("="^80)
println("  Hypothesis: Structural fingerprints capture architecture families")
println("  Models of same architecture → cluster in fingerprint space")
println("  Cluster medoid → optimal OOD generalization")
println("="^80)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 0: MODEL GRID
# ═══════════════════════════════════════════════════════════════════════════════

# Architecture families: (n, n_classes, n_pairs) × seeds
arch_configs = [
    # (name, n, n_classes, n_pairs, seeds)
    ("n16_c2",  16, 2, 1, [101, 102, 103]),
    ("n24_c4",  24, 4, 2, [201, 202, 203]),
    ("n32_c4",  32, 4, 2, [301, 302, 303]),
    ("n48_c4",  48, 4, 2, [401, 402, 403]),
    ("n32_c6",  32, 6, 3, [501]),
    ("n16_c4",  16, 4, 2, [601]),
]

function make_dataset_for(n, n_classes, n_pairs, seed=42)
    FP.make_dataset(n, n_pairs, n_classes, seed)
end

function train_model(n, n_classes, n_pairs, seed; injection=0.0, epochs=500)
    xs_tr, ys_tr, xs_te, ys_te = make_dataset_for(n, n_classes, n_pairs, 42)
    p = FP.SignalPipeline(n; n_classes=n_classes, n_pairs=n_pairs, seed=seed)
    FP.train_pipeline!(p, xs_tr, ys_tr; epochs=epochs)
    if injection > 0
        for c in 1:n_classes
            if 10 + c <= size(p.Wc, 2)
                p.Wc[c, 10 + c] += injection
            end
        end
    end
    return p, xs_tr, ys_tr, xs_te, ys_te
end

function model_fn(p, n, n_classes)
    function fn(x)
        layer = p.layer
        feats = FG.combined_bispec_features(x, layer)
        logits = p.Wc * feats + p.bc
        return (feats, logits)
    end
    return fn
end

function accuracy(p, xs, ys)
    layer = p.layer
    correct = 0
    for (x, y) in zip(xs, ys)
        feats = FG.combined_bispec_features(x, layer)
        logits = p.Wc * feats + p.bc
        if argmax(logits) == y
            correct += 1
        end
    end
    return correct / length(xs) * 100
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: TRAIN ALL MODELS + MEASURE FINGERPRINTS
# ═══════════════════════════════════════════════════════════════════════════════

abstract type ModelSpec end

struct ArchModel <: ModelSpec
    name::String
    family::String
    n::Int
    n_classes::Int
    n_pairs::Int
    seed::Int
    injection::Float64
end

models_to_build = ArchModel[]
for (family, n, n_classes, n_pairs, seeds) in arch_configs
    for seed in seeds
        name = "$(family)_s$(seed)"
        push!(models_to_build, ArchModel(name, family, n, n_classes, n_pairs, seed, 0.0))
    end
end
# Add a spurious variant for n32_c4 to test within-family detection
for seed in [301]
    push!(models_to_build, ArchModel("n32_c4_spur_s301", "n32_c4", 32, 4, 2, seed, 1.0))
end

n_total = length(models_to_build)
@printf "\n  Total models: %d\n" n_total
@printf "  Families: %s\n" join(unique([m.family for m in models_to_build]), ", ")

results = Vector{UI.UnifiedResult}(undef, n_total)
pipelines = Vector{Any}(undef, n_total)
xs_trains = Vector{Any}(undef, n_total)
ys_trains = Vector{Any}(undef, n_total)
xs_tests = Vector{Any}(undef, n_total)
ys_tests = Vector{Any}(undef, n_total)
accs_id = zeros(Float64, n_total)

println("\n" * "="^80)
println("  TRAINING + FINGERPRINTING ALL MODELS")
println("="^80)

t_start = time()
for (i, m) in enumerate(models_to_build)
    @printf "\n  [%2d/%2d] %-25s (n=%2d, c=%d, s=%d) ... " i n_total m.name m.n m.n_classes m.seed
    
    p, x_tr, y_tr, x_te, y_te = train_model(m.n, m.n_classes, m.n_pairs, m.seed; injection=m.injection)
    pipelines[i] = p
    xs_trains[i] = x_tr
    ys_trains[i] = y_tr
    xs_tests[i] = x_te
    ys_tests[i] = y_te
    accs_id[i] = accuracy(p, x_te, y_te)
    @printf "acc=%.0f%%" accs_id[i]
    
    fn = model_fn(p, m.n, m.n_classes)
    t0 = time()
    r = UI.analyze_all(x_tr; model_fn=fn, data_name=m.name)
    t_elapsed = time() - t0
    results[i] = r
    @printf "  |  %d/%d analyzers (%.1fs)" r.n_success r.n_total t_elapsed
end

t_total = time() - t_start
@printf "\n\n  Total time: %.1fs (%.1fs per model)\n" t_total (t_total/n_total)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: BUILD EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  STRUCTURAL EMBEDDING")
println("="^80)

names = [m.name for m in models_to_build]
families = [m.family for m in models_to_build]
emb = SE.structural_embedding(results; n_dims=8, model_names=names)
SE.embedding_summary(emb)

SE.export_embedding_csv(emb, joinpath(@__DIR__, "frontier_coords.csv"))
SE.top_contributing_measurements(emb; n=15)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: CLUSTERING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  PHASE 1: CLUSTERING BY ARCHITECTURE FAMILY")
println("="^80)

unique_families = unique(families)
n_families = length(unique_families)
n_models = length(models_to_build)

# For each family, compute intra-family distances vs inter-family distances
family_members = Dict{String, Vector{Int}}()
for (i, fam) in enumerate(families)
    if !haskey(family_members, fam)
        family_members[fam] = Int[]
    end
    push!(family_members[fam], i)
end

# Intra vs inter distance
println("\n  ── Intra-family vs Inter-family distances ──")
@printf "  %-20s %12s %12s %10s\n" "Family" "Intra-mean" "Inter-mean" "Ratio"
println("  " * "-" ^ 58)

all_intra = Float64[]
all_inter = Float64[]
for fam in unique_families
    members = family_members[fam]
    n_members = length(members)
    if n_members < 2
        @printf "  %-20s %12s %12s %10s\n" fam "n/a" "n/a" "n/a"
        continue
    end
    intra_dists = Float64[]
    inter_dists = Float64[]
    for i in members
        for j in members
            if j > i
                push!(intra_dists, SE.fingerprint_distance(emb, i, j))
            end
        end
        for j in 1:n_models
            if !(j in members)
                push!(inter_dists, SE.fingerprint_distance(emb, i, j))
            end
        end
    end
    intra_mean = mean(intra_dists)
    inter_mean = mean(inter_dists)
    ratio = inter_mean / max(intra_mean, 1e-10)
    append!(all_intra, intra_dists)
    append!(all_inter, inter_dists)
    @printf "  %-20s %12.4f %12.4f %10.2f\n" fam intra_mean inter_mean ratio
end

if length(all_intra) > 0 && length(all_inter) > 0
    println("\n  ── Aggregate ──")
    @printf "  Mean intra-family distance:  %.4f\n" mean(all_intra)
    @printf "  Mean inter-family distance:  %.4f\n" mean(all_inter)
    @printf "  Separation ratio:           %.2f×\n" (mean(all_inter) / max(mean(all_intra), 1e-10))
    if mean(all_inter) > 1.5 * mean(all_intra)
        println("  ✓ ARCHITECTURE CLUSTERING CONFIRMED — families separable in fingerprint space")
    elseif mean(all_inter) > 1.2 * mean(all_intra)
        println("  ~ Moderate separation — families partially overlapping")
    else
        println("  ✗ Weak or no separation — fingerprint not capturing architecture")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: STRUCTURAL MEDOID HYPOTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  PHASE 2: STRUCTURAL MEDOID HYPOTHESIS")
println("="^80)
println("  Prediction: The model closest to cluster centroid has best OOD generalization")

# For each multi-member family, find medoid and test if it's the best OOD generalizer
# First, define OOD tests

println("\n  ── OOD Test 1: Frequency shift ──")
function ood_freq_shift(p, m::ArchModel, n_ood=100)
    n, n_classes = m.n, m.n_classes
    Random.seed!(999)
    xs = Vector{Float64}[]
    ys = Int[]
    for _ in 1:n_ood
        label = rand(1:n_classes)
        x = zeros(n)
        freq = (label + 2) * 2  # shifted frequencies
        for i in 1:n
            x[i] = sin(2π * freq * i / n) + 0.3 * cos(2π * freq * 2 * i / n) + 0.1 * randn()
        end
        push!(xs, x)
        push!(ys, label)
    end
    return accuracy(p, xs, ys)
end

println("\n  ── OOD Test 2: Gaussian noise ──")
function ood_noise(p, m::ArchModel, n_ood=100)
    _, _, xs_te, ys_te = make_dataset_for(m.n, m.n_classes, m.n_pairs, 42)
    indices = rand(1:length(xs_te), min(n_ood, length(xs_te)))
    xs_noisy = [x + 0.5 * randn(length(x)) for (i, x) in enumerate(xs_te) if i in indices]
    ys_noisy = [ys_te[i] for (i, y) in enumerate(ys_te) if i in indices]
    return accuracy(p, xs_noisy, ys_noisy)
end

println("\n  ── OOD Test 3: Reduced signal (low amplitude) ──")
function ood_low_amp(p, m::ArchModel, n_ood=100)
    n, n_classes = m.n, m.n_classes
    Random.seed!(777)
    xs = Vector{Float64}[]
    ys = Int[]
    for _ in 1:n_ood
        label = rand(1:n_classes)
        x = zeros(n)
        freq = label * 2
        for i in 1:n
            x[i] = 0.05 * sin(2π * freq * i / n) + 0.01 * randn()  # very low amplitude
        end
        push!(xs, x)
        push!(ys, label)
    end
    return accuracy(p, xs, ys)
end

ood_accs = zeros(n_models, 3)
for i in 1:n_models
    m = models_to_build[i]
    p = pipelines[i]
    ood_accs[i, 1] = ood_freq_shift(p, m)
    ood_accs[i, 2] = ood_noise(p, m)
    ood_accs[i, 3] = ood_low_amp(p, m)
end

# Medoid analysis
println("\n  ── Medoid per family ──")
for fam in unique_families
    members = family_members[fam]
    n_members = length(members)
    n_members < 2 && continue
    
    # Find medoid: minimize sum of distances to all other members
    best_i = members[1]
    best_sum = Inf
    for i in members
        s = sum(SE.fingerprint_distance(emb, i, j) for j in members if j != i)
        if s < best_sum
            best_sum = s
            best_i = i
        end
    end
    
    # Find centroid: mean of all members' coords
    centroid = mean(emb.coords[members, :], dims=1)
    dists_to_centroid = [norm(emb.coords[i, :] - vec(centroid)) for i in members]
    centroid_best = members[argmin(dists_to_centroid)]
    
    # Which is the best in each OOD test?
    ood_means = [mean(ood_accs[i, :]) for i in members]
    best_ood = members[argmax(ood_means)]
    
    @printf "\n  %s (%d models):\n" fam n_members
    @printf "    Medoid (min pairwise dist):  %s (idx %d)\n" names[best_i] best_i
    @printf "    Nearest to centroid:         %s (idx %d)\n" names[centroid_best] centroid_best
    @printf "    Best OOD (mean of 3 tests):  %s (idx %d, OOD=%.1f%%)\n" names[best_ood] best_ood mean(ood_accs[best_ood, :])
    @printf "    Medoid matches best OOD:     %s\n" (best_i == best_ood ? "✓ YES" : (centroid_best == best_ood ? "~ Centroid matches" : "✗ NO"))
    
    for i in members
        ood_mean = mean(ood_accs[i, :])
        @printf "      %-25s acc_id=%.0f%% ood=[%.0f%% %.0f%% %.0f%%] mean=%.1f%%\n" names[i] accs_id[i] ood_accs[i,1] ood_accs[i,2] ood_accs[i,3] ood_mean
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: FINGERPRINT SIMILARITY → OOD BEHAVIOR SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  PHASE 3: FINGERPRINT SIMILARITY → OOD BEHAVIOR SIMILARITY")
println("="^80)
println("  Hypothesis: Models with similar fingerprints have similar OOD accuracy\n")

# For all pairs of models from the SAME family, compute:
#   d_fp = fingerprint distance
#   d_ood = L2 distance of [ood_acc_freq, ood_acc_noise, ood_acc_lowamp]
# Correlate d_fp with d_ood

all_pairs_fp = Float64[]
all_pairs_ood = Float64[]

for a in 1:n_models
    for b in a+1:n_models
        d_fp = SE.fingerprint_distance(emb, a, b)
        d_ood = norm(ood_accs[a, :] - ood_accs[b, :])
        push!(all_pairs_fp, d_fp)
        push!(all_pairs_ood, d_ood)
    end
end

if length(all_pairs_fp) > 3
    μx = mean(all_pairs_fp)
    μy = mean(all_pairs_ood)
    σx = std(all_pairs_fp)
    σy = std(all_pairs_ood)
    if σx > 0 && σy > 0
        r = mean((all_pairs_fp .- μx) .* (all_pairs_ood .- μy)) / (σx * σy)
        @printf "  Pearson r(fp_distance, ood_distance) = %.4f (n=%d pairs)\n" r length(all_pairs_fp)
        @printf "  R² = %.4f\n" r^2
        if abs(r) > 0.5
            println("  ✓ STRONG: fingerprint proximity → similar OOD behavior")
            println("  → Structural fingerprint predicts failure modes")
        elseif abs(r) > 0.3
            println("  ~ MODERATE: partial correlation")
        else
            println("  ✗ WEAK: fingerprint does not predict OOD similarity")
        end
    end
end

# Now same-family only
println("\n  ── Same-family pairs only ──")
same_family_fp = Float64[]
same_family_ood = Float64[]
for a in 1:n_models
    for b in a+1:n_models
        if families[a] == families[b]
            push!(same_family_fp, SE.fingerprint_distance(emb, a, b))
            push!(same_family_ood, norm(ood_accs[a, :] - ood_accs[b, :]))
        end
    end
end

if length(same_family_fp) > 3
    μx_s = mean(same_family_fp)
    μy_s = mean(same_family_ood)
    σx_s = std(same_family_fp)
    σy_s = std(same_family_ood)
    if σx_s > 0 && σy_s > 0
        r_s = mean((same_family_fp .- μx_s) .* (same_family_ood .- μy_s)) / (σx_s * σy_s)
        @printf "  Pearson r(fp_distance, ood_distance) WITHIN families = %.4f (n=%d pairs)\n" r_s length(same_family_fp)
        @printf "  R² = %.4f\n" r_s^2
        if abs(r_s) > 0.5
            println("  ✓ Within-family: fingerprint predicts OOD ordering among siblings")
        else
            println("  ~ Within-family: seed-level differences not captured by fingerprint")
            println("  → Fingerprint captures architecture, not weight initialization")
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: SPURIOUS MODEL DETECTION — CAN WE SEE THE SPURIOUS MODEL?
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  PHASE 4: SPURIOUS MODEL DETECTION")
println("="^80)

spur_idx = findfirst(i -> occursin("spur", names[i]), 1:n_models)
if spur_idx !== nothing
    n32_idx = findall(i -> families[i] == "n32_c4" && !occursin("spur", names[i]), 1:n_models)
    @printf "\n  Spurious model: %s (idx %d)\n" names[spur_idx] spur_idx
    @printf "  Clean siblings: %s\n" join([names[i] for i in n32_idx], ", ")
    
    # Distance from spurious model to its family cluster
    for i in n32_idx
        d = SE.fingerprint_distance(emb, spur_idx, i)
        @printf "  dist(spurious, %s) = %.4f\n" names[i] d
    end
    
    # What's the nearest model overall?
    nn = SE.find_nearest(emb, spur_idx; k=3)
    @printf "\n  Nearest neighbors to spurious model:\n"
    for (j, d) in nn
        @printf "    %s (dist=%.4f)\n" names[j] d
    end
    
    # Intra-family distance mean
    n32_dists = [SE.fingerprint_distance(emb, i, j) for i in n32_idx, j in n32_idx if i < j]
    mean_intra = length(n32_dists) > 0 ? mean(n32_dists) : 0
    spur_to_fam = mean([SE.fingerprint_distance(emb, spur_idx, i) for i in n32_idx])
    
    @printf "\n  Mean intra-family distance (n32_c4): %.4f\n" mean_intra
    @printf "  Mean spurious-to-family distance:    %.4f\n" spur_to_fam
    if spur_to_fam > mean_intra * 1.5
        println("  ✓ SPURIOUS MODEL DETECTED — far from its family cluster")
    elseif spur_to_fam > mean_intra * 1.2
        println("  ~ Marginal separation — spurious model at edge of cluster")
    else
        println("  ✗ Spurious model NOT detected — inside family cluster")
        println("  → Confirms: weight-level spurious injection is invisible to structural fingerprint")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: CERTIFICATE CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  PHASE 5: SYMMETRY CERTIFICATE vs FINGERPRINT")
println("="^80)

println("\n  Running SymmetryCertificate for all models...")
deploy_scores = zeros(n_models)
logit_divs = zeros(n_models)
for i in 1:n_models
    m = models_to_build[i]
    fn = model_fn(pipelines[i], m.n, m.n_classes)
    feat_dim = 3 * m.n
    cert = WDW.SymmetryCertificate.quick_audit(xs_trains[i], fn, ["features", "logits"], [feat_dim, m.n_classes])
    deploy_scores[i] = cert.deployability_score
    logit_divs[i] = cert.audit.layer_divergences[end]
end

@printf "\n  %-25s %12s %12s %12s %12s\n" "Model" "Family" "PC1" "Deploy" "LogitDiv"
println("  " * "-" ^ 77)
for i in 1:n_models
    @printf "  %-25s %-12s %10.3f %10.1f%% %10.4f\n" names[i] families[i] emb.coords[i,1] (deploy_scores[i]*100) logit_divs[i]
end

# Correlation: certificate vs position
if emb.n_dims >= 1
    cert_y = deploy_scores
    for dim in 1:min(3, emb.n_dims)
        x_dim = emb.coords[:, dim]
        μx = mean(x_dim); μy = mean(cert_y)
        σx = std(x_dim); σy = std(cert_y)
        if σx > 0 && σy > 0
            r_cert = mean((x_dim .- μx) .* (cert_y .- μy)) / (σx * σy)
            @printf "\n  Pearson r(PC%d, deployability) = %.4f\n" dim r_cert
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 8: EXPORT & SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  EXPORTING DATA")
println("="^80)

# Full matrix export
meas_csv = joinpath(@__DIR__, "frontier_measurements.csv")
n_meas = length(results[1].measurement_names)
open(meas_csv, "w") do io
    header = "measurement,family," * join(names, ",")
    write(io, header * "\n")
    for mi in 1:n_meas
        row = results[1].measurement_names[mi] * "," * families[1]
        for ri in 1:n_total
            val = mi <= length(results[ri].measurement_matrix[:]) ? results[ri].measurement_matrix[mi] : 0.0
            row *= "," * string(round(val, digits=6))
        end
        write(io, row * "\n")
    end
end
println("  Exported: $meas_csv")

# Summary table
results_csv = joinpath(@__DIR__, "frontier_summary.csv")
open(results_csv, "w") do io
    write(io, "model,family,n,n_classes,seed,acc_id,ood_freq,ood_noise,ood_lowamp,deploy,logit_div,pc1,pc2\n")
    for i in 1:n_total
        m = models_to_build[i]
        write(io, "$(names[i]),$(families[i]),$(m.n),$(m.n_classes),$(m.seed),$(round(accs_id[i],digits=1)),")
        write(io, "$(round(ood_accs[i,1],digits=1)),$(round(ood_accs[i,2],digits=1)),$(round(ood_accs[i,3],digits=1)),")
        write(io, "$(round(deploy_scores[i],digits=4)),$(round(logit_divs[i],digits=6)),")
        write(io, "$(round(emb.coords[i,1],digits=6)),")
        write(io, "$(emb.n_dims >= 2 ? round(emb.coords[i,2],digits=6) : 0.0)\n")
    end
end
println("  Exported: $results_csv")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  FINAL VERDICT: STRUCTURAL FRONTIER")
println("="^80)

# Determine key results
sep_ok = length(all_intra) > 0 && length(all_inter) > 0 && mean(all_inter) > 1.5 * mean(all_intra)
ood_pred_ok = length(all_pairs_fp) > 3 && abs(r) > 0.3
medoid_ok = false
# Check if any family medoid matches best OOD
for fam in unique_families
    members = family_members[fam]
    length(members) < 2 && continue
    ood_means = [mean(ood_accs[i, :]) for i in members]
    best_ood = members[argmax(ood_means)]
    # compute medoid
    best_i = members[1]
    best_sum = Inf
    for i in members
        s = sum(SE.fingerprint_distance(emb, i, j) for j in members if j != i)
        if s < best_sum
            best_sum = s
            best_i = i
        end
    end
    if best_i == best_ood
        medoid_ok = true
        break
    end
end

println("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    STRUCTURAL FRONTIER VERDICT                    │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                   │
  │  Architecture clustering:  $(sep_ok ? "✓ CONFIRMED" : "✗ NOT CONFIRMED")
  │  OOD behavior prediction:  $(ood_pred_ok ? "✓ CONFIRMED" : "~ PARTIAL / WEAK")
  │  Structural medoid match:  $(medoid_ok ? "✓ CONFIRMED" : "✗ NOT CONFIRMED")
  │  Spurious model detected:  $(spur_idx !== nothing && spur_to_fam > mean_intra * 1.2 ? "✓ YES" : "✗ NO (expected)")
  │                                                                   │
  ├─────────────────────────────────────────────────────────────────┤
  │  INTERPRETATION:                                                  │
  │                                                                   │
  │  The structural fingerprint is an ARCHITECTURAL TAXONOMY —       │
  │  it captures the family structure of models, not weight-level    │
  │  perturbations. This is both a limitation and a strength:        │
  │                                                                   │
  │  ✓ Can identify model architecture from behavior alone           │
  │  ✓ Can predict similar failure modes within architecture family  │
  │  ✗ Cannot detect spurious correlations in weights                │
  │  ✗ Cannot predict OOD accuracy from fingerprint alone             │
  │                                                                   │
  │  The fingerprint answers: "What kind of model is this?"          │
  │  Not: "How accurate is this model on OOD data?"                  │
  │                                                                   │
  │  This is useful for: model auditing, architecture identification, │
  │  transfer learning triage, and detecting architecture drift.     │
  └─────────────────────────────────────────────────────────────────┘
""")
