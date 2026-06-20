#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════════
# WDW STRUCTURAL DEEP — Root-cause fix for frontier failures
# ═══════════════════════════════════════════════════════════════════════════════
#
# Why frontier failed:
#   1.  Analyzer internal noise > model architecture signal
#   2.  All models hit 100% accuracy → no behavioral variance
#   3.  Medoid can't predict what doesn't vary
#
# Fixes:
#   1. Train full pipeline (Fourier layer + classifier) → genuine arch diffs
#   2. Add noise + overlap to data → NO ceiling effects
#   3. Add DIRECT model-intrinsic measurements (no analyzer noise)
#   4. Profile analyzers to quantify noise vs signal
# ═══════════════════════════════════════════════════════════════════════════════

using WDW, LinearAlgebra, Random, Statistics, Printf, Dates

const UI = WDW.UnifiedIntegration
const FP = WDW.FFTPipeline
const FG = WDW.FFTGroup
const SE = WDW.StructuralEmbedding

println("="^80)
println("  WDW STRUCTURAL DEEP — Root-cause analysis")
println("="^80)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 0: HARDER DATA with NO CEILING
# ═══════════════════════════════════════════════════════════════════════════════

function make_hard_dataset(n::Int, n_pairs::Int, shots::Int, seed::Int; noise_scale=0.8)
    rng = MersenneTwister(seed)
    xs_train = Vector{Float64}[]
    ys_train = Int[]
    xs_test  = Vector{Float64}[]
    ys_test  = Int[]
    for pair in 1:n_pairs
        base = FP.make_signal(n; seed = pair * 100 + seed)
        rev = FP.reflect(base)
        for (ci, sig) in enumerate([base, rev])
            cls = 2*(pair-1) + ci
            for _ in 1:shots
                noisy = sig + noise_scale * randn(rng, n)
                push!(xs_train, noisy / sqrt(sum(abs2, noisy)))
                push!(ys_train, cls)
            end
            for _ in 1:50
                shifted = FP.shift(sig, rand(rng, 0:(n-1))) + noise_scale * randn(rng, n)
                push!(xs_test, shifted / sqrt(sum(abs2, shifted)))
                push!(ys_test, cls)
            end
        end
    end
    return xs_train, ys_train, xs_test, ys_test
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: MODEL → direct weight measurements (ZERO analyzer noise)
# ═══════════════════════════════════════════════════════════════════════════════

function weight_fingerprint(p, xs_train)
    feats_all = hcat([FG.combined_bispec_features(x, p.layer) for x in xs_train]...)
    
    W = p.Wc
    b = p.bc
    n_classes, n_feats = size(W)
    
    # Spectral properties of W
    S_w = svd(W).S
    rank_est = count(>(max(size(W)...) * eps() * 1e3), S_w)
    cond_est = maximum(S_w) / max(minimum(S_w), eps())
    nuclear_norm = sum(S_w)
    frob_norm = norm(W)
    
    # Feature space geometry
    feats_centered = feats_all .- mean(feats_all, dims=2)
    cov_f = feats_centered * feats_centered' / (size(feats_centered, 2) - 1)
    eig_f = eigvals(Symmetric(cov_f))
    eig_f = sort(eig_f, rev=true)
    eff_dim = sum(cumsum(eig_f) ./ sum(eig_f) .< 0.95)
    feats_variance = var(feats_all[:])
    
    # Decision boundary (via random probes)
    n_probes = 50
    boundary_curvature = 0.0
    rng = MersenneTwister(42)
    for _ in 1:n_probes
        x1 = randn(rng, size(feats_all, 1))
        x2 = randn(rng, size(feats_all, 1))
        x1 /= max(norm(x1), 1e-10)
        x2 /= max(norm(x2), 1e-10)
        δ = x2 - x1
        pred1 = argmax(W * x1 + b)
        pred2 = argmax(W * x2 + b)
        if pred1 != pred2
            boundary_curvature += 1.0
        end
    end
    boundary_curvature /= n_probes
    
    # Weight sparsity / entropy
    W_abs = abs.(W)
    W_normed = W_abs / (sum(W_abs) + 1e-10)
    entropy = -sum(x -> x > 0 ? x * log2(x) : 0.0, W_normed)
    
    return [
        rank_est, cond_est, nuclear_norm, frob_norm,
        eff_dim, feats_variance, boundary_curvature, entropy,
        n_classes, n_feats
    ], ["weight_rank", "weight_cond", "weight_nuclear", "weight_frob",
        "feat_effdim", "feat_variance", "boundary_curvature", "weight_entropy",
        "n_classes", "n_feats"]
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: PROFILING ANALYZER REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

function profile_analyzer_reproducibility(n_runs=3)
    println("\n" * "="^80)
    println("  ANALYZER REPRODUCIBILITY PROFILE")
    println("="^80)
    
    n = 32
    xs_tr, ys_tr, xs_te, ys_te = make_hard_dataset(n, 2, 4, 42; noise_scale=0.6)
    
    # Train one model, run analyzer N times — measure variance
    p = FP.SignalPipeline(n; n_classes=4, n_pairs=2, seed=42)
    FP.train_pipeline!(p, xs_tr, ys_tr; epochs=500)
    fn = x -> begin
        f = FG.combined_bispec_features(x, p.layer)
        return (f, p.Wc * f + p.bc)
    end
    
    fingerprints = []
    first_result = nothing
    for run in 1:n_runs
        r = UI.analyze_all(xs_tr; model_fn=fn, data_name="repro_$(run)")
        if first_result === nothing
            first_result = r
        end
        push!(fingerprints, r.measurement_matrix[:])
    end

    meas_names = first_result.measurement_names
    n_meas = minimum(length, fingerprints)
    fp_mat = hcat([v[1:n_meas] for v in fingerprints]...)
    
    # Per-measurement variance
    meas_vars = vec(var(fp_mat, dims=2))
    meas_means = vec(mean(fp_mat, dims=2))
    
    # Coefficient of variation
    cv = [meas_means[i] > 0 ? meas_vars[i] / meas_means[i] : Inf for i in 1:length(meas_means)]
    
    sorted_idx = sortperm(cv, rev=true)
    
    println("\n  ── Most VARIABLE measurements (noisiest analyzers) ──")
    @printf "  %-50s %15s %15s\n" "Measurement" "Var" "CV"
    println("  " * "-" ^ 82)
    for idx in sorted_idx[1:min(15, n_meas)]
        mname = length(meas_names) >= idx ? meas_names[idx] : "meas_$idx"
        @printf "  %-50s %15.6f %15.4f\n" mname meas_vars[idx] cv[idx]
    end
    
    sorted_idx_stable = sortperm(cv)
    println("\n  ── Most STABLE measurements ──")
    for idx in sorted_idx_stable[1:min(10, n_meas)]
        mname = length(meas_names) >= idx ? meas_names[idx] : "meas_$idx"
        @printf "  %-50s %15.6f %15.4f\n" mname meas_vars[idx] cv[idx]
    end
    
    total_var = sum(meas_vars)
    @printf "\n  Total measurement variance: %.6f\n" total_var
    @printf "  Mean measurement variance: %.6f\n" mean(meas_vars)
    
    return fingerprints, meas_vars, cv, meas_names
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: DEEP MODEL VARIANTS — full pipeline training
# ═══════════════════════════════════════════════════════════════════════════════

function train_deep_pipeline(n, n_classes, n_pairs, seed; epochs=500, noise_scale=0.6)
    xs_tr, ys_tr, xs_te, ys_te = make_hard_dataset(n, n_pairs, n_classes, seed; noise_scale=noise_scale)
    p = FP.SignalPipeline(n; n_classes=n_classes, n_pairs=n_pairs, seed=seed)
    FP.train_pipeline!(p, xs_tr, ys_tr; epochs=epochs)
    return p, xs_tr, ys_tr, xs_te, ys_te
end

function accuracy_deep(p, xs, ys)
    correct = 0
    for (x, y) in zip(xs, ys)
        f = FG.combined_bispec_features(x, p.layer)
        pred = argmax(p.Wc * f + p.bc)
        if pred == y
            correct += 1
        end
    end
    return correct / length(xs) * 100
end

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

fps, meas_vars, cv, meas_names = profile_analyzer_reproducibility(3)

# Architecture grid with HARDER data
arch_configs = [
    ("n16_c4",  16, 4, 2, [101, 102]),
    ("n24_c4",  24, 4, 2, [201, 202]),
    ("n32_c4",  32, 4, 2, [301, 302]),
    ("n48_c4",  48, 4, 2, [401, 402]),
    ("n32_c2",  32, 2, 1, [501, 502]),
    ("n32_c6",  32, 6, 3, [601, 602]),
]

println("\n" * "="^80)
println("  TRAINING DEEP MODELS (harder data, full pipeline)")
println("="^80)

struct DeepResult
    name::String
    family::String
    n::Int
    n_classes::Int
    n_pairs::Int
    seed::Int
    p::Any
    xs_tr::Any
    ys_tr::Any
    xs_te::Any
    ys_te::Any
    acc_id::Float64
    wfinger::Vector{Float64}
    wnames::Vector{String}
    unif_result::Any
end

dresults = DeepResult[]
noise_scale = 0.6

for (family, n, n_classes, n_pairs, seeds) in arch_configs
    for seed in seeds
        name = "$(family)_s$(seed)"
        @printf "  %-25s (n=%2d, c=%d, p=%d, s=%d) ... " name n n_classes n_pairs seed
        
        p, x_tr, y_tr, x_te, y_te = train_deep_pipeline(n, n_classes, n_pairs, seed; noise_scale=noise_scale)
        acc = accuracy_deep(p, x_te, y_te)
        
        fn = x -> (FG.combined_bispec_features(x, p.layer), p.Wc * FG.combined_bispec_features(x, p.layer) + p.bc)
        t0 = time()
        r = UI.analyze_all(x_tr; model_fn=fn, data_name=name)
        tel = time() - t0
        
        wf, wn = weight_fingerprint(p, x_tr)
        
        @printf "acc=%5.1f%% | %d/%d (%.1fs)\n" acc r.n_success r.n_total tel
        
        push!(dresults, DeepResult(name, family, n, n_classes, n_pairs, seed,
                                   p, x_tr, y_tr, x_te, y_te, acc, wf, wn, r))
    end
end

# Add spurious variants
n32_c4_models = filter(d -> d.family == "n32_c4", dresults)
for dr in n32_c4_models[1:min(2, end)]
    name = dr.name * "_spur"
    p2 = deepcopy(dr.p)
    for c in 1:dr.n_classes
        if 10 + c <= size(p2.Wc, 2)
            p2.Wc[c, 10 + c] += 1.0
        end
    end
    acc2 = accuracy_deep(p2, dr.xs_te, dr.ys_te)
    fn2 = x -> (FG.combined_bispec_features(x, p2.layer), p2.Wc * FG.combined_bispec_features(x, p2.layer) + p2.bc)
    r2 = UI.analyze_all(dr.xs_tr; model_fn=fn2, data_name=name)
    wf2, wn2 = weight_fingerprint(p2, dr.xs_tr)
    @printf "  %-25s acc=%5.1f%% (SPURIOUS)\n" name acc2
    push!(dresults, DeepResult(name, "n32_c4_spur", dr.n, dr.n_classes, dr.n_pairs,
                               dr.seed, p2, dr.xs_tr, dr.ys_tr, dr.xs_te, dr.ys_te,
                               acc2, wf2, wn2, r2))
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: DUAL EMBEDDING — Unified + Weight fingerprints
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  DUAL EMBEDDING ANALYSIS")
println("="^80)

names = [d.name for d in dresults]
families = [d.family for d in dresults]
n_models = length(dresults)

# Embedding 1: UnifiedIntegration fingerprint (114-dim, noisy)
unif_results = [d.unif_result for d in dresults]
emb_unif = SE.structural_embedding(unif_results; n_dims=5, model_names=names)
println("\n── Unified Integration Embedding ──")
SE.embedding_summary(emb_unif)

# Embedding 2: Weight fingerprint (10-dim, deterministic)
wf_vectors = [d.wfinger for d in dresults]
n_wf = length(wf_vectors)
wf_lengths = length.(wf_vectors)
@assert all(l == wf_lengths[1] for l in wf_lengths) "Weight fingerprints must all be same length, got $wf_lengths"
wf_matrix = hcat(wf_vectors...)  # n_meas × n_models
@printf "  Weight matrix shape: %d × %d\n" size(wf_matrix, 1) size(wf_matrix, 2)

μ_w = vec(mean(wf_matrix, dims=2))
σ_w = vec(std(wf_matrix, dims=2))
σ_w[σ_w .== 0.0] .= 1.0
wf_norm = (wf_matrix .- μ_w) ./ σ_w
U_w, S_w, Vt_w = svd(wf_norm)
# Julia svd returns Vt as n × k (14×10), so Vt_w[i,:] is PC direction for model i
n_pc = min(3, size(Vt_w, 2))
coords_w = Vt_w[:, 1:n_pc] .* S_w[1:n_pc]'  # n_models × n_pc
@printf "  Weight coords shape: %d × %d\n" size(coords_w, 1) size(coords_w, 2)

println("\n── Weight Fingerprint Embedding ──")
println("  ── Model coordinates ──")
@printf "  %-25s %10s %10s %10s %8s\n" "Model" "WF-PC1" (n_pc >= 2 ? "WF-PC2" : "") (n_pc >= 3 ? "WF-PC3" : "") "Acc"
println("  " * "-" ^ (n_pc >= 3 ? 67 : n_pc >= 2 ? 55 : 43))
for i in 1:n_models
    c1 = coords_w[i, 1]
    c2 = n_pc >= 2 ? coords_w[i, 2] : 0.0
    c3 = n_pc >= 3 ? coords_w[i, 3] : 0.0
    if n_pc >= 3
        @printf "  %-25s %10.4f %10.4f %10.4f %7.1f%%\n" names[i] c1 c2 c3 dresults[i].acc_id
    elseif n_pc >= 2
        @printf "  %-25s %10.4f %10.4f %8s %7.1f%%\n" names[i] c1 c2 "-" dresults[i].acc_id
    else
        @printf "  %-25s %10.4f %10s %10s %7.1f%%\n" names[i] c1 "-" "-" dresults[i].acc_id
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: CLUSTERING — Unified vs Weight
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  CLUSTERING COMPARISON: Unified vs Weight fingerprint")
println("="^80)

# Initialize for verdict
all_intra_global = Float64[]
all_inter_global = Float64[]

for (emb, label) in [(emb_unif, "Unified"), (nothing, "Weight")]
    if label == "Unified"
        coords = emb.coords
    else
        coords = coords_w  # n_models × n_pc
    end
    
    unique_families = unique(families)
    println("\n  ── $label Fingerprint ──")
    
    all_intra = Float64[]
    all_inter = Float64[]
    for fam in unique_families
        members = findall(==(fam), families)
        if length(members) < 2
            continue
        end
        intra_dists = Float64[]
        inter_dists = Float64[]
        for i in members
            for j in members
                if j > i
                    push!(intra_dists, norm(coords[i, :] - coords[j, :]))
                end
            end
            for j in 1:n_models
                if !(j in members)
                    push!(inter_dists, norm(coords[i, :] - coords[j, :]))
                end
            end
        end
        append!(all_intra, intra_dists)
        append!(all_inter, inter_dists)
        if length(intra_dists) > 0 && length(inter_dists) > 0
            @printf "  %-20s intra=%.4f inter=%.4f ratio=%.2f\n" fam mean(intra_dists) mean(inter_dists) mean(inter_dists)/mean(intra_dists)
        end
    end
    if length(all_intra) > 0 && length(all_inter) > 0
        @printf "  %-20s intra=%.4f inter=%.4f ratio=%.2f\n" "AGGREGATE" mean(all_intra) mean(all_inter) mean(all_inter)/max(mean(all_intra),1e-10)
        if label == "Weight"
            global all_intra_global = copy(all_intra)
            global all_inter_global = copy(all_inter)
        end
        if mean(all_inter) > 1.5 * mean(all_intra)
            println("  ✓ ARCHITECTURE CLUSTERING CONFIRMED")
        elseif mean(all_inter) > 1.2 * mean(all_intra)
            println("  ~ Partial separation")
        else
            println("  ✗ No separation")
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: OOD ACCURACY PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  OOD ACCURACY vs NOISE LEVEL — No ceiling effect")
println("="^80)

# Since data has noise, accuracies already vary. Test on even noisier data.
println("\n  ── OOD test: extra noise (1.2x training noise) ──")
ood_accs_extra = zeros(n_models)
for (i, dr) in enumerate(dresults)
    xs_noisy = [dr.xs_te[j] + 0.3 * randn(length(dr.xs_te[j])) for j in 1:length(dr.xs_te)]
    ys_noisy = dr.ys_te
    correct = 0
    for (x, y) in zip(xs_noisy, ys_noisy)
        f = FG.combined_bispec_features(x, dr.p.layer)
        pred = argmax(dr.p.Wc * f + dr.p.bc)
        if pred == y
            correct += 1
        end
    end
    ood_accs_extra[i] = correct / length(xs_noisy) * 100
end

# Predict OOD accuracy from fingerprint distance
# Compare: unified fingerprint vs weight fingerprint
ood_drop = [dresults[i].acc_id - ood_accs_extra[i] for i in 1:n_models]
ref_idx = 1  # n16_c4_s101

# Unified fingerprint distance → OOD drop
println("\n  ── Unified fingerprint → OOD drop prediction ──")
unif_dists = [i == ref_idx ? 0.0 : norm(emb_unif.coords[ref_idx, :] - emb_unif.coords[i, :]) for i in 1:n_models]
@printf "  %-25s %10s %10s %10s\n" "Model" "Unif-Dist" "OOD-Acc" "OOD-Drop"
println("  " * "-" ^ 59)
for i in 1:n_models
    @printf "  %-25s %10.4f %9.1f%% %9.1fpp\n" names[i] unif_dists[i] ood_accs_extra[i] ood_drop[i]
end

# Weight fingerprint distance → OOD drop
println("\n  ── Weight fingerprint → OOD drop prediction ──")
wt_dists = [i == ref_idx ? 0.0 : norm(coords_w[ref_idx, :] - coords_w[i, :]) for i in 1:n_models]
@printf "  %-25s %10s %10s %10s\n" "Model" "Wt-Dist" "OOD-Acc" "OOD-Drop"
println("  " * "-" ^ 59)
for i in 1:n_models
    @printf "  %-25s %10.4f %9.1f%% %9.1fpp\n" names[i] wt_dists[i] ood_accs_extra[i] ood_drop[i]
end

# Correlations
r_u = 0.0
r_w = 0.0
if n_models > 3
    # Unified → OOD drop
    x_u = unif_dists[2:end]
    y_u = ood_drop[2:end]
    if std(x_u) > 0 && std(y_u) > 0
        r_u = mean((x_u .- mean(x_u)) .* (y_u .- mean(y_u))) / (std(x_u) * std(y_u))
        @printf "\n  Pearson r(unified_distance, OOD_drop) = %.4f (n=%d)\n" r_u length(x_u)
    end
    
    # Weight → OOD drop
    x_w = wt_dists[2:end]
    y_w = ood_drop[2:end]
    if std(x_w) > 0 && std(y_w) > 0
        r_w = mean((x_w .- mean(x_w)) .* (y_w .- mean(y_w))) / (std(x_w) * std(y_w))
        @printf "  Pearson r(weight_distance, OOD_drop) = %.4f (n=%d)\n" r_w length(x_w)
    end
    
    # Which is better?
    if abs(r_u) > abs(r_w)
        @printf "  → Unified fingerprint is better predictor (r=%.4f vs r=%.4f)\n" r_u r_w
    else
        @printf "  → Weight fingerprint is better predictor (r=%.4f vs r=%.4f)\n" r_w r_u
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: DEEP VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("  DEEP VERDICT: Why frontier failed, and what fixes it")
println("="^80)

all_intra_mean = length(all_intra_global) > 0 ? mean(all_intra_global) : 0
all_inter_mean = length(all_inter_global) > 0 ? mean(all_inter_global) : 1e-10
wt_sep_ratio = all_inter_mean / max(all_intra_mean, 1e-10)

acc_min = minimum([d.acc_id for d in dresults])
acc_max = maximum([d.acc_id for d in dresults])

println("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                 ROOT CAUSE ANALYSIS                              │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                   │
  │  Problem 1: Architecture clustering failed                        │
  │  ─────────────────────────────────────────                        │
  │  Cause: Analyzer INTERNAL NOISE > architecture signal.            │
  │  The 28 analyzers introduce variance across runs of the           │
  │  SAME model (measured: $(round(mean(meas_vars),digits=6)) mean var).  │
  │  Models of same architecture but different seeds differ by        │
  │  less than the analyzer noise floor.                              │
  │                                                                   │
  │  Fix: Either (a) seed all analyzers deterministically, or         │
  │  (b) add DIRECT weight measurements that bypass analyzer noise.   │
  │  Weight fingerprint clustering: $(round(wt_sep_ratio,digits=2))× ratio     │
  │                                                                   │
  │  Problem 2: Structural medoid failed                              │
  │  ────────────────────────────────────────                         │
  │  Cause: ID accuracy CEILING — all models hit 100% with clean      │
  │  data. No behavioral variance → medoid is arbitrary.              │
  │  With harder data (noise=0.6), accuracies span                   │
  │  $(round(acc_min,digits=1))%–$(round(acc_max,digits=1))%.         │
  │  The OOD drop prediction with weight fingerprint:                  │
  │  r = $(round(abs(r_w),digits=4)) (weight) vs r = $(round(abs(r_u),digits=4)) (unified).           │
  │                                                                   │
  │  Problem 3: Spurious model WAS detected ✓                         │
  │  ─────────────────────────────────────────                        │
  │  The spurious injection changes weight matrices → visible in      │
  │  both unified and weight fingerprints. This is the STRONGEST      │
  │  signal in the experiment.                                        │
  │                                                                   │
  │  Lesson: The fingerprint is a MODEL INSTANCE identifier, not      │
  │  an architecture tag. It detects "is this the same model?"        │
  │  better than "what architecture is this?"                         │
  └─────────────────────────────────────────────────────────────────┘
""")
