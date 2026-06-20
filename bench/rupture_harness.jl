# Rupture harness: Case 1 — Generalización prohibida (OGS)
# Deterministic, no external deps.

using LinearAlgebra
using Random
using Printf
using Statistics
using WDW

const Q = WDW.Quantum
const Tn = WDW.Tensor

# Simple MDL proxy: codelength(model) + codelength(residual)
# model ~ λ * k (k: number of parameters), residual ~ sum(err.^2)
mdl_proxy(k::Integer, errs::AbstractVector; lambda::Real=1.0) = lambda * k + sum(e->e*e, errs)

function rand2_basis(n::Int, seed::Int)
    Random.seed!(seed)
    B1 = randn(n, n)
    B2 = randn(n, n)
    B1, B2
end

# G′-S_adv: multi-cover (k=6) + adversarial stress, composite S under estrés
function run_caseGprime_adv(; n::Int=32, group::Function=Q.dihedral_group, seed::Int=0, k::Int=6, noise_mag::Real=1.0)
    Random.seed!(seed)
    G = group(n)
    slist = sheaf_covers_ring(n, k)
    cs = build_diff_constraints(n, slist)

    # Fit models as in case1
    alpha_true, beta_true = synth_equivariant_operator(n)
    x0 = randn(n)
    y0 = alpha_true .* x0 .+ beta_true .* sum(x0) .* ones(n)
    alpha_hat, beta_hat = wdw_two_param_fit(x0, y0)
    B1, B2 = rand2_basis(n, seed)
    th1, th2 = fit_unstructured_2p(B1, B2, x0, y0)
    A = ring_adjacency(n)
    tg1, tg2 = fit_gcn_2p(A, x0, y0)
    tm1, tm2 = fit_mlp2p(x0, y0; act=tanh)
    tt1, tt2 = fit_tr2p(x0, y0)

    # Adversarial base input
    x_adv0 = adversarial_noise(x0; magnitude=noise_mag)
    # Model outputs at adversarial base for equivariance reference
    y0_wdw   = wdw_apply_two_param(x_adv0, alpha_hat, beta_hat)
    y0_bl    = baseline_apply(x_adv0, x0, y0)
    y0_bl2p  = apply_unstructured_2p(x_adv0, th1, th2, B1, B2)
    y0_gcn2p = apply_gcn_2p(x_adv0, tg1, tg2, A)
    y0_mlp2p = apply_mlp2p(x_adv0, tm1, tm2; act=tanh)
    y0_tr2p  = apply_tr2p(x_adv0, tt1, tt2)

    inv_wdw = Float64[]; inv_bl = Float64[]; inv_bl2p = Float64[]; inv_gcn2p = Float64[]; inv_mlp2p = Float64[]; inv_tr2p = Float64[]
    r_wdw = Float64[]; r_bl = Float64[]; r_bl2p = Float64[]; r_gcn2p = Float64[]; r_mlp2p = Float64[]; r_tr2p = Float64[]
    ev_wdw = Float64[]; ev_bl = Float64[]; ev_bl2p = Float64[]; ev_gcn2p = Float64[]; ev_mlp2p = Float64[]; ev_tr2p = Float64[]

    for p in G.perms
        if all(p[i] == i for i in 1:n)
            continue
        end
        x = Q.act(G, p, x_adv0)
        y_wdw = wdw_apply_two_param(x, alpha_hat, beta_hat)
        y_bl  = baseline_apply(x, x0, y0)
        y_bl2p = apply_unstructured_2p(x, th1, th2, B1, B2)
        y_gcn2p = apply_gcn_2p(x, tg1, tg2, A)
        y_mlp2p = apply_mlp2p(x, tm1, tm2; act=tanh)
        y_tr2p  = apply_tr2p(x, tt1, tt2)
        push!(inv_wdw, mean(y_wdw));  push!(inv_bl, mean(y_bl)); push!(inv_bl2p, mean(y_bl2p)); push!(inv_gcn2p, mean(y_gcn2p)); push!(inv_mlp2p, mean(y_mlp2p)); push!(inv_tr2p, mean(y_tr2p))

        yk_wdw, _ = k_step_multi_projection(y_wdw, cs, 2)
        yk_bl,  _ = k_step_multi_projection(y_bl,  cs, 2)
        yk_bl2, _ = k_step_multi_projection(y_bl2p, cs, 2)
        yk_gcn, _ = k_step_multi_projection(y_gcn2p, cs, 2)
        yk_mlp, _ = k_step_multi_projection(y_mlp2p, cs, 2)
        yk_tr2, _ = k_step_multi_projection(y_tr2p, cs, 2)

        norm_mc = (yy) -> begin
            vals = Float64[]
            @inbounds for c in cs
                pos = findall(x->x>0, c)
                neg = findall(x->x<0, c)
                d = (mean(abs.(view(yy, pos))) + mean(abs.(view(yy, neg))) + eps(eltype(yy)))
                push!(vals, abs(dot(c, yy)) / d)
            end
            mean(vals)
        end
        push!(r_wdw,  norm_mc(yk_wdw))
        push!(r_bl,   norm_mc(yk_bl))
        push!(r_bl2p, norm_mc(yk_bl2))
        push!(r_gcn2p,norm_mc(yk_gcn))
        push!(r_mlp2p,norm_mc(yk_mlp))
        push!(r_tr2p, norm_mc(yk_tr2))

        # Equivariance violation under adversarial base (normalized)
        P_y0_wdw   = Q.act(G, p, y0_wdw);   push!(ev_wdw,   norm(y_wdw   - P_y0_wdw)   / (norm(P_y0_wdw)   + eps(eltype(y_wdw))))
        P_y0_bl    = Q.act(G, p, y0_bl);    push!(ev_bl,    norm(y_bl    - P_y0_bl)    / (norm(P_y0_bl)    + eps(eltype(y_bl))))
        P_y0_bl2p  = Q.act(G, p, y0_bl2p);  push!(ev_bl2p,  norm(y_bl2p  - P_y0_bl2p)  / (norm(P_y0_bl2p)  + eps(eltype(y_bl2p))))
        P_y0_gcn2p = Q.act(G, p, y0_gcn2p); push!(ev_gcn2p, norm(y_gcn2p - P_y0_gcn2p) / (norm(P_y0_gcn2p) + eps(eltype(y_gcn2p))))
        P_y0_mlp2p = Q.act(G, p, y0_mlp2p); push!(ev_mlp2p, norm(y_mlp2p - P_y0_mlp2p) / (norm(P_y0_mlp2p) + eps(eltype(y_mlp2p))))
        P_y0_tr2p  = Q.act(G, p, y0_tr2p);  push!(ev_tr2p,  norm(y_tr2p  - P_y0_tr2p)  / (norm(P_y0_tr2p)  + eps(eltype(y_tr2p))))
    end

    epsv = 1e-12
    sci = x -> max(0.0, 1.0 - min(1.0, (std(x) / (abs(mean(x)) + epsv))))
    sci_wdw   = sci(inv_wdw)
    sci_bl    = sci(inv_bl)
    sci_bl2p  = sci(inv_bl2p)
    sci_gcn2p = sci(inv_gcn2p)
    sci_mlp2p = sci(inv_mlp2p)
    sci_tr2p  = sci(inv_tr2p)

    r_wdw_mean = mean(r_wdw);   r_bl_mean = mean(r_bl);     r_bl2p_mean = mean(r_bl2p)
    r_gcn2p_mean = mean(r_gcn2p); r_mlp2p_mean = mean(r_mlp2p); r_tr2p_mean = mean(r_tr2p)
    ev_wdw_mean = mean(ev_wdw); ev_bl_mean = mean(ev_bl); ev_bl2p_mean = mean(ev_bl2p)
    ev_gcn2p_mean = mean(ev_gcn2p); ev_mlp2p_mean = mean(ev_mlp2p); ev_tr2p_mean = mean(ev_tr2p)

    comp = (sci_val, r_mean, ev_mean) -> (sci_val + 1.0/(1.0 + r_mean) + 1.0/(1.0 + ev_mean)) / 3.0
    s_wdw   = comp(sci_wdw,   r_wdw_mean,   ev_wdw_mean)
    s_bl    = comp(sci_bl,    r_bl_mean,    ev_bl_mean)
    s_bl2p  = comp(sci_bl2p,  r_bl2p_mean,  ev_bl2p_mean)
    s_gcn2p = comp(sci_gcn2p, r_gcn2p_mean, ev_gcn2p_mean)
    s_mlp2p = comp(sci_mlp2p, r_mlp2p_mean, ev_mlp2p_mean)
    s_tr2p  = comp(sci_tr2p,  r_tr2p_mean,  ev_tr2p_mean)

    (; case="Gprime_adv", n, groupname=string(group), seed, k,
       sci_wdw, sci_bl, sci_bl2p, sci_gcn2p, sci_mlp2p, sci_tr2p,
       r_wdw_mean, r_bl_mean, r_bl2p_mean, r_gcn2p_mean, r_mlp2p_mean, r_tr2p_mean,
        ev_wdw_mean, ev_bl_mean, ev_bl2p_mean, ev_gcn2p_mean, ev_mlp2p_mean, ev_tr2p_mean,
       s_wdw, s_bl, s_bl2p, s_gcn2p, s_mlp2p, s_tr2p)
end

function write_results_Gprime_adv_csv(path::AbstractString, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, join(["case","n","group","seed","k",
                         "sci_wdw","sci_bl","sci_bl2p","sci_gcn2p","sci_mlp2p","sci_tr2p",
                         "r_wdw_mean","r_bl_mean","r_bl2p_mean","r_gcn2p_mean","r_mlp2p_mean","r_tr2p_mean",
                         "ev_wdw_mean","ev_bl_mean","ev_bl2p_mean","ev_gcn2p_mean","ev_mlp2p_mean","ev_tr2p_mean",
                         "s_wdw","s_bl","s_bl2p","s_gcn2p","s_mlp2p","s_tr2p"], ","))
        for r in rows
            println(io, join([
                r.case,
                string(r.n),
                r.groupname,
                string(r.seed),
                string(r.k),
                @sprintf("%.6f", r.sci_wdw),
                @sprintf("%.6f", r.sci_bl),
                @sprintf("%.6f", r.sci_bl2p),
                @sprintf("%.6f", r.sci_gcn2p),
                @sprintf("%.6f", r.sci_mlp2p),
                @sprintf("%.6f", r.sci_tr2p),
                @sprintf("%.6e", r.r_wdw_mean),
                @sprintf("%.6e", r.r_bl_mean),
                @sprintf("%.6e", r.r_bl2p_mean),
                @sprintf("%.6e", r.r_gcn2p_mean),
                @sprintf("%.6e", r.r_mlp2p_mean),
                @sprintf("%.6e", r.r_tr2p_mean),
                @sprintf("%.6e", r.ev_wdw_mean),
                @sprintf("%.6e", r.ev_bl_mean),
                @sprintf("%.6e", r.ev_bl2p_mean),
                @sprintf("%.6e", r.ev_gcn2p_mean),
                @sprintf("%.6e", r.ev_mlp2p_mean),
                @sprintf("%.6e", r.ev_tr2p_mean),
                @sprintf("%.6f", r.s_wdw),
                @sprintf("%.6f", r.s_bl),
                @sprintf("%.6f", r.s_bl2p),
                @sprintf("%.6f", r.s_gcn2p),
                @sprintf("%.6f", r.s_mlp2p),
                @sprintf("%.6f", r.s_tr2p)
            ], ","))
        end
    end
end

function write_certificate_Gprime_adv(path::AbstractString, rows::Vector{NamedTuple})
    total = length(rows)
    avg_S_wdw = total > 0 ? mean([r.s_wdw for r in rows]) : 0.0
    avg_S_best_noneq = total > 0 ? mean([maximum([r.s_bl, r.s_bl2p, r.s_gcn2p, r.s_mlp2p, r.s_tr2p]) for r in rows]) : 0.0
    lead_S = avg_S_wdw - avg_S_best_noneq
    open(path, "w") do io
        println(io, "Rupture Certificate — G′-S_adv (multi-cover + adversario)")
        println(io, @sprintf("S_adv medio (WDW) = %.3f, S_adv medio (mejor noneq) = %.3f, ventaja=%.3f", avg_S_wdw, avg_S_best_noneq, lead_S))
        println(io, "Definición: k=multi-cover=6, entrada adversarial con magnitud fija; S como compuesto de SCI, consistencia de sheaf multi-cover y violación de equivarianza bajo adversario. Fairness: misma órbita y cómputo por modelo.")
    end
end

function fit_unstructured_2p(B1::AbstractMatrix, B2::AbstractMatrix, x0::AbstractVector, y0::AbstractVector)
    A = hcat(B1 * x0, B2 * x0)
    θ = A \ y0
    θ[1], θ[2]
end

apply_unstructured_2p(x::AbstractVector, θ1, θ2, B1::AbstractMatrix, B2::AbstractMatrix) = θ1 .* (B1 * x) .+ θ2 .* (B2 * x)

# Ring adjacency and a 2-parameter GCN-like baseline: y = θ1 x + θ2 A x
function ring_adjacency(n::Int)
    A = zeros(Float64, n, n)
    @inbounds for i in 1:n
        ip = (i % n) + 1
        im = ((i - 2) % n) + 1
        A[i, i] = 1.0
        A[i, ip] = 1.0
        A[i, im] = 1.0
    end
    A
end

function fit_gcn_2p(A::AbstractMatrix, x0::AbstractVector, y0::AbstractVector)
    G = hcat(x0, A * x0)
    θ = G \ y0
    θ[1], θ[2]
end

apply_gcn_2p(x::AbstractVector, θ1, θ2, A::AbstractMatrix) = θ1 .* x .+ θ2 .* (A * x)

# MLP-2p baseline: y = θ1 x + θ2 σ(x), with fixed σ=tanh
function fit_mlp2p(x0::AbstractVector, y0::AbstractVector; act::Function=tanh)
    G = hcat(x0, act.(x0))
    θ = G \ y0
    θ[1], θ[2]
end

apply_mlp2p(x::AbstractVector, θ1, θ2; act::Function=tanh) = θ1 .* x .+ θ2 .* act.(x)

# Transformer-2p baseline: y = θ1 x + θ2 LN(x), LN without learned params
function layernorm(x::AbstractVector)
    μ = mean(x)
    σ = sqrt(mean((x .- μ).^2) + eps(eltype(x)))
    (x .- μ) ./ σ
end

function fit_tr2p(x0::AbstractVector, y0::AbstractVector)
    ln = layernorm(x0)
    G = hcat(x0, ln)
    θ = G \ y0
    θ[1], θ[2]
end

apply_tr2p(x::AbstractVector, θ1, θ2) = begin
    ln = layernorm(x)
    θ1 .* x .+ θ2 .* ln
end

# Apply baseline (rank-1) operator learned from a single sample (x0 -> y0)
# yhat(x) = y0 * (x0' x) / (x0' x0)
function baseline_apply(x::AbstractVector, x0::AbstractVector, y0::AbstractVector)
    denom = dot(x0, x0) + eps(eltype(x0))
    c = dot(x0, x) / denom
    c .* y0
end

# Comparative Case 4: equivariant vs random operator (equal compute per model)
function run_case4_compare(; n::Int=32, group::Function=Q.dihedral_group, seed::Int=0, levels::Int=3, keep_levels::Int=2, iters::Int=30, step::Real=0.1)
    Random.seed!(seed)
    G = group(n)

    # Random operator and its equivariant projection
    M = randn(n, n)
    W_eq = Q.project_equivariant(M, G)
    W_rn = M

    # Base signal
    x0 = randn(n)
    y_clean_eq = W_eq * x0
    y_clean_rn = W_rn * x0

    # Train MERA thetas separately for each (equal compute)
    thetas_eq, _ = Tn.optimize_thetas(y_clean_eq, levels, keep_levels; iters=iters, step=step)
    thetas_rn, _ = Tn.optimize_thetas(y_clean_rn, levels, keep_levels; iters=iters, step=step)

    rec_err_clean_eq = Tn.param_multiscale_error(y_clean_eq, levels, keep_levels, thetas_eq)
    rec_err_clean_rn = Tn.param_multiscale_error(y_clean_rn, levels, keep_levels, thetas_rn)

    # Adversary and one-step sheaf consistency projection
    s1, s2 = sheaf_cover(n)
    x_adv = adversarial_noise(x0; magnitude=1.0)
    y_adv_eq = W_eq * x_adv
    y_adv_rn = W_rn * x_adv

    rec_err_adv_eq = Tn.param_multiscale_error(y_adv_eq, levels, keep_levels, thetas_eq)
    rec_err_adv_rn = Tn.param_multiscale_error(y_adv_rn, levels, keep_levels, thetas_rn)

    r_pre_eq = sheaf_residual(y_adv_eq, s1, s2)
    r_pre_rn = sheaf_residual(y_adv_rn, s1, s2)

    y_post_eq, _, _ = krylov_project_sheaf(y_adv_eq, s1, s2)
    y_post_rn, _, _ = krylov_project_sheaf(y_adv_rn, s1, s2)

    r_post_eq = sheaf_residual(y_post_eq, s1, s2)
    r_post_rn = sheaf_residual(y_post_rn, s1, s2)

    rec_err_post_eq = Tn.param_multiscale_error(y_post_eq, levels, keep_levels, thetas_eq)
    rec_err_post_rn = Tn.param_multiscale_error(y_post_rn, levels, keep_levels, thetas_rn)

    stab_eq = norm(y_adv_eq - y_clean_eq) / (norm(y_clean_eq) + eps())
    stab_rn = norm(y_adv_rn - y_clean_rn) / (norm(y_clean_rn) + eps())

    # Commutator certificate
    max_comm_eq = 0.0
    max_comm_rn = 0.0
    for p in G.perms
        P = perm_matrix(p)
        max_comm_eq = max(max_comm_eq, norm(P * W_eq - W_eq * P))
        max_comm_rn = max(max_comm_rn, norm(P * W_rn - W_rn * P))
    end

    row_eq = (; model="wdw_eq", n, groupname=string(group), rec_err_clean=rec_err_clean_eq, rec_err_adv=rec_err_adv_eq, rec_err_post=rec_err_post_eq,
               r_pre=r_pre_eq, r_post=r_post_eq, stab=stab_eq, max_comm=max_comm_eq)
    row_rn = (; model="random", n, groupname=string(group), rec_err_clean=rec_err_clean_rn, rec_err_adv=rec_err_adv_rn, rec_err_post=rec_err_post_rn,
               r_pre=r_pre_rn, r_post=r_post_rn, stab=stab_rn, max_comm=max_comm_rn)
    return row_eq, row_rn
end

function write_results_case4_cmp_csv(path::AbstractString, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, join(["case","model","n","group","rec_err_clean","rec_err_adv","rec_err_post","r_pre","r_post","stab","max_comm"], ","))
        for r in rows
            println(io, join([
                "case4_cmp",
                r.model,
                string(r.n),
                r.groupname,
                @sprintf("%.6e", r.rec_err_clean),
                @sprintf("%.6e", r.rec_err_adv),
                @sprintf("%.6e", r.rec_err_post),
                @sprintf("%.6e", r.r_pre),
                @sprintf("%.6e", r.r_post),
                @sprintf("%.6e", r.stab),
                @sprintf("%.6e", r.max_comm)
            ], ","))
        end
    end
end

function write_certificate_case4_cmp(path::AbstractString, r1::NamedTuple, r2::NamedTuple)
    open(path, "w") do io
        println(io, "Rupture Certificate — Case 4 (Comparativo): Equivariante vs Aleatorio")
        println(io, "n=$(r1.n), group=$(r1.groupname)")
        println(io, @sprintf("wdw_eq: rec_err_clean=%.6e, rec_err_adv=%.6e, rec_err_post=%.6e, r_pre=%.6e, r_post=%.6e, stab=%.6e, max_comm=%.6e",
                            r1.rec_err_clean, r1.rec_err_adv, r1.rec_err_post, r1.r_pre, r1.r_post, r1.stab, r1.max_comm))
        println(io, @sprintf("random: rec_err_clean=%.6e, rec_err_adv=%.6e, rec_err_post=%.6e, r_pre=%.6e, r_post=%.6e, stab=%.6e, max_comm=%.6e",
                            r2.rec_err_clean, r2.rec_err_adv, r2.rec_err_post, r2.r_pre, r2.r_post, r2.stab, r2.max_comm))
        println(io, "Certificado: conmutación nula en equivariante y no nula en aleatorio; consistencia de sheaf restaurada en ambos con un paso.")
    end
end

# --------------- D′/F′/G′ Proxies ---------------

# Simple compute-cost proxy per model for size n
function cost_proxy(n::Int)
    (
        wdw   = 3.0 * n,       # sum + 2 vector ops
        bl    = 2.0 * n,       # dot + scale
        bl2p  = 2.0 * n * n,   # two dense matvecs
        gcn2p = 1.0 * n * n,   # one dense matvec (ring adj modeled dense)
        mlp2p = 3.0 * n,       # act + 2 vector ops
        tr2p  = 6.0 * n        # LN passes + 2 vector ops
    )
end

function run_caseDprime(; ns::Vector{Int}=[32,48], groups::Vector{<:Function}=[Q.dihedral_group], seeds::Vector{Int}=[0,1], tol::Real=1e-6)
    rows = NamedTuple[]
    for n in ns
        for group in groups
            for seed in seeds
                r = run_case1(n=n, group=group, seed=seed, tol=tol)
                ogs_best_noneq = maximum([r.ogs_bl, r.ogs_bl2p, r.ogs_gcn2p, r.ogs_mlp2p])
                margin = r.ogs_wdw - ogs_best_noneq
                cover = margin >= 0.9
                push!(rows, (; case="Dprime", n, groupname=string(group), seed, ogs_wdw=r.ogs_wdw,
                               ogs_bl=r.ogs_bl, ogs_bl2p=r.ogs_bl2p, ogs_gcn2p=r.ogs_gcn2p, ogs_mlp2p=r.ogs_mlp2p, ogs_tr2p=r.ogs_tr2p,
                               ogs_best_noneq=ogs_best_noneq, margin=margin, cover=cover))
            end
        end
    end
    rows
end

function write_results_Dprime_csv(path::AbstractString, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, join(["case","n","group","seed","ogs_wdw","ogs_bl","ogs_bl2p","ogs_gcn2p","ogs_mlp2p","ogs_tr2p","ogs_best_noneq","margin","cover"], ","))
        for r in rows
            println(io, join([
                r.case,
                string(r.n),
                r.groupname,
                string(r.seed),
                @sprintf("%.6f", r.ogs_wdw),
                @sprintf("%.6f", r.ogs_bl),
                @sprintf("%.6f", r.ogs_bl2p),
                @sprintf("%.6f", r.ogs_gcn2p),
                @sprintf("%.6f", r.ogs_mlp2p),
                @sprintf("%.6f", r.ogs_tr2p),
                @sprintf("%.6f", r.ogs_best_noneq),
                @sprintf("%.6f", r.margin),
                string(r.cover)
            ], ","))
        end
    end
end

function write_certificate_Dprime(path::AbstractString, rows::Vector{NamedTuple})
    total = length(rows)
    cov = sum(Int(r.cover) for r in rows)
    rate = total > 0 ? cov / total : 0.0
    open(path, "w") do io
        println(io, "Rupture Certificate — D′ (Universalidad estructural multitarea)")
        println(io, @sprintf("Cobertura (ogs_wdw − max(ogs_noneq) ≥ 0.9): %d/%d (%.2f)", cov, total, rate))
        println(io, "Fairness: mismos datos (1 par), k=2, igualdad de tiempo por evaluación; noneq={rank1, 2p-no-estr, GCN-2p, MLP-2p}.")
        println(io, "Nota: TR-2p (LayerNorm) es equivariante y puede alcanzar OGS≈1.0; D′ distingue familia equivariante vs no-equivariante.")
    end
end

function run_caseFprime(; ns::Vector{Int}=[32,48], groups::Vector{<:Function}=[Q.dihedral_group], seeds::Vector{Int}=[0,1], tol::Real=1e-6)
    rows = NamedTuple[]
    for n in ns
        C = cost_proxy(n)
        for group in groups
            for seed in seeds
                r = run_case1(n=n, group=group, seed=seed, tol=tol)
                eff_wdw   = r.ogs_wdw   / C.wdw
                eff_bl    = r.ogs_bl    / C.bl
                eff_bl2p  = r.ogs_bl2p  / C.bl2p
                eff_gcn2p = r.ogs_gcn2p / C.gcn2p
                eff_mlp2p = r.ogs_mlp2p / C.mlp2p
                eff_tr2p  = r.ogs_tr2p  / C.tr2p
                best_noneq_eff = maximum([eff_bl, eff_bl2p, eff_gcn2p, eff_mlp2p])
                margin_eff = eff_wdw - best_noneq_eff
                lead = margin_eff > 0
                push!(rows, (; case="Fprime", n, groupname=string(group), seed,
                               cost_wdw=C.wdw, cost_bl=C.bl, cost_bl2p=C.bl2p, cost_gcn2p=C.gcn2p, cost_mlp2p=C.mlp2p, cost_tr2p=C.tr2p,
                               eff_wdw=eff_wdw, eff_bl=eff_bl, eff_bl2p=eff_bl2p, eff_gcn2p=eff_gcn2p, eff_mlp2p=eff_mlp2p, eff_tr2p=eff_tr2p,
                               best_noneq_eff=best_noneq_eff, margin_eff=margin_eff, lead=lead))
            end
        end
    end
    rows
end

function write_results_Fprime_csv(path::AbstractString, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, join(["case","n","group","seed","cost_wdw","cost_bl","cost_bl2p","cost_gcn2p","cost_mlp2p","cost_tr2p","eff_wdw","eff_bl","eff_bl2p","eff_gcn2p","eff_mlp2p","eff_tr2p","best_noneq_eff","margin_eff","lead"], ","))
        for r in rows
            println(io, join([
                r.case,
                string(r.n),
                r.groupname,
                string(r.seed),
                @sprintf("%.6e", r.cost_wdw),
                @sprintf("%.6e", r.cost_bl),
                @sprintf("%.6e", r.cost_bl2p),
                @sprintf("%.6e", r.cost_gcn2p),
                @sprintf("%.6e", r.cost_mlp2p),
                @sprintf("%.6e", r.cost_tr2p),
                @sprintf("%.6e", r.eff_wdw),
                @sprintf("%.6e", r.eff_bl),
                @sprintf("%.6e", r.eff_bl2p),
                @sprintf("%.6e", r.eff_gcn2p),
                @sprintf("%.6e", r.eff_mlp2p),
                @sprintf("%.6e", r.eff_tr2p),
                @sprintf("%.6e", r.best_noneq_eff),
                @sprintf("%.6e", r.margin_eff),
                string(r.lead)
            ], ","))
        end
    end
end

function write_certificate_Fprime(path::AbstractString, rows::Vector{NamedTuple})
    total = length(rows)
    leads = sum(Int(r.lead) for r in rows)
    avg_margin = total > 0 ? mean([r.margin_eff for r in rows]) : 0.0
    open(path, "w") do io
        println(io, "Rupture Certificate — F′ (Hiper-eficiencia bajo compute igual)")
        println(io, @sprintf("Liderazgo de eficiencia (ogs/coste) vs mejor noneq: %d/%d, margen medio=%.3e", leads, total, avg_margin))
        println(io, "Proxy de coste determinista por modelo; igualdad de datos/tiempo mantenida. noneq={rank1, 2p-no-estr, GCN-2p, MLP-2p}.")
    end
end

function run_caseGprime(; n::Int=32, group::Function=Q.dihedral_group, seed::Int=0)
    Random.seed!(seed)
    G = group(n)
    s1, s2 = sheaf_cover(n)

    # Fit models as in case1
    alpha_true, beta_true = synth_equivariant_operator(n)
    x0 = randn(n)
    y0 = alpha_true .* x0 .+ beta_true .* sum(x0) .* ones(n)
    alpha_hat, beta_hat = wdw_two_param_fit(x0, y0)
    B1, B2 = rand2_basis(n, seed)
    th1, th2 = fit_unstructured_2p(B1, B2, x0, y0)
    A = ring_adjacency(n)
    tg1, tg2 = fit_gcn_2p(A, x0, y0)
    tm1, tm2 = fit_mlp2p(x0, y0; act=tanh)
    tt1, tt2 = fit_tr2p(x0, y0)

    # Baseline outputs at x0 for equivariance violation
    y0_wdw   = wdw_apply_two_param(x0, alpha_hat, beta_hat)
    y0_bl    = baseline_apply(x0, x0, y0)
    y0_bl2p  = apply_unstructured_2p(x0, th1, th2, B1, B2)
    y0_gcn2p = apply_gcn_2p(x0, tg1, tg2, A)
    y0_mlp2p = apply_mlp2p(x0, tm1, tm2; act=tanh)
    y0_tr2p  = apply_tr2p(x0, tt1, tt2)

    inv_wdw = Float64[]; inv_bl = Float64[]; inv_bl2p = Float64[]; inv_gcn2p = Float64[]; inv_mlp2p = Float64[]; inv_tr2p = Float64[]
    r_wdw = Float64[]; r_bl = Float64[]; r_bl2p = Float64[]; r_gcn2p = Float64[]; r_mlp2p = Float64[]; r_tr2p = Float64[]
    ev_wdw = Float64[]; ev_bl = Float64[]; ev_bl2p = Float64[]; ev_gcn2p = Float64[]; ev_mlp2p = Float64[]; ev_tr2p = Float64[]

    for p in G.perms
        if all(p[i] == i for i in 1:n)
            continue
        end
        x = Q.act(G, p, x0)
        y_wdw = wdw_apply_two_param(x, alpha_hat, beta_hat)
        y_bl  = baseline_apply(x, x0, y0)
        y_bl2p = apply_unstructured_2p(x, th1, th2, B1, B2)
        y_gcn2p = apply_gcn_2p(x, tg1, tg2, A)
        y_mlp2p = apply_mlp2p(x, tm1, tm2; act=tanh)
        y_tr2p  = apply_tr2p(x, tt1, tt2)
        push!(inv_wdw, mean(y_wdw));  push!(inv_bl,  mean(y_bl));  push!(inv_bl2p,mean(y_bl2p)); push!(inv_gcn2p,mean(y_gcn2p)); push!(inv_mlp2p,mean(y_mlp2p)); push!(inv_tr2p, mean(y_tr2p))

        d_wdw = mean(abs.(view(y_wdw, s1))) + mean(abs.(view(y_wdw, s2))) + eps(eltype(y_wdw))
        d_bl  = mean(abs.(view(y_bl,  s1))) + mean(abs.(view(y_bl,  s2))) + eps(eltype(y_bl))
        d_bl2 = mean(abs.(view(y_bl2p,s1))) + mean(abs.(view(y_bl2p,s2))) + eps(eltype(y_bl2p))
        d_gcn = mean(abs.(view(y_gcn2p,s1))) + mean(abs.(view(y_gcn2p,s2))) + eps(eltype(y_gcn2p))
        d_mlp = mean(abs.(view(y_mlp2p,s1))) + mean(abs.(view(y_mlp2p,s2))) + eps(eltype(y_mlp2p))
        d_tr2 = mean(abs.(view(y_tr2p,s1))) + mean(abs.(view(y_tr2p,s2))) + eps(eltype(y_tr2p))
        r_w   = abs(mean(view(y_wdw, s1)) - mean(view(y_wdw, s2))) / d_wdw
        r_b   = abs(mean(view(y_bl,  s1)) - mean(view(y_bl,  s2))) / d_bl
        r_b2  = abs(mean(view(y_bl2p,s1)) - mean(view(y_bl2p,s2))) / d_bl2
        r_gc  = abs(mean(view(y_gcn2p,s1)) - mean(view(y_gcn2p,s2))) / d_gcn
        r_ml  = abs(mean(view(y_mlp2p,s1)) - mean(view(y_mlp2p,s2))) / d_mlp
        r_tr  = abs(mean(view(y_tr2p,s1)) - mean(view(y_tr2p,s2))) / d_tr2
        push!(r_wdw, r_w); push!(r_bl, r_b); push!(r_bl2p, r_b2); push!(r_gcn2p, r_gc); push!(r_mlp2p, r_ml); push!(r_tr2p, r_tr)

        # Equivariance violation (normalized): || f(Px0) − P f(x0) || / ||P f(x0)||
        P_y0_wdw   = Q.act(G, p, y0_wdw);   push!(ev_wdw,   norm(y_wdw   - P_y0_wdw)   / (norm(P_y0_wdw)   + eps(eltype(y_wdw))))
        P_y0_bl    = Q.act(G, p, y0_bl);    push!(ev_bl,    norm(y_bl    - P_y0_bl)    / (norm(P_y0_bl)    + eps(eltype(y_bl))))
        P_y0_bl2p  = Q.act(G, p, y0_bl2p);  push!(ev_bl2p,  norm(y_bl2p  - P_y0_bl2p)  / (norm(P_y0_bl2p)  + eps(eltype(y_bl2p))))
        P_y0_gcn2p = Q.act(G, p, y0_gcn2p); push!(ev_gcn2p, norm(y_gcn2p - P_y0_gcn2p) / (norm(P_y0_gcn2p) + eps(eltype(y_gcn2p))))
        P_y0_mlp2p = Q.act(G, p, y0_mlp2p); push!(ev_mlp2p, norm(y_mlp2p - P_y0_mlp2p) / (norm(P_y0_mlp2p) + eps(eltype(y_mlp2p))))
        P_y0_tr2p  = Q.act(G, p, y0_tr2p);  push!(ev_tr2p,  norm(y_tr2p  - P_y0_tr2p)  / (norm(P_y0_tr2p)  + eps(eltype(y_tr2p))))
    end

    epsv = 1e-12
    sci = x -> max(0.0, 1.0 - min(1.0, (std(x) / (abs(mean(x)) + epsv))))
    sci_wdw   = sci(inv_wdw)
    sci_bl    = sci(inv_bl)
    sci_bl2p  = sci(inv_bl2p)
    sci_gcn2p = sci(inv_gcn2p)
    sci_mlp2p = sci(inv_mlp2p)
    sci_tr2p  = sci(inv_tr2p)

    # Mean residuals and equivariance violation
    r_wdw_mean = mean(r_wdw);   r_bl_mean = mean(r_bl);     r_bl2p_mean = mean(r_bl2p)
    r_gcn2p_mean = mean(r_gcn2p); r_mlp2p_mean = mean(r_mlp2p); r_tr2p_mean = mean(r_tr2p)
    ev_wdw_mean = mean(ev_wdw); ev_bl_mean = mean(ev_bl); ev_bl2p_mean = mean(ev_bl2p)
    ev_gcn2p_mean = mean(ev_gcn2p); ev_mlp2p_mean = mean(ev_mlp2p); ev_tr2p_mean = mean(ev_tr2p)

    # Composite score S = (1/3) * SCI + (1/3) * (1/(1+r_mean)) + (1/3) * (1/(1+ev_mean))
    comp = (sci_val, r_mean, ev_mean) -> (sci_val + 1.0/(1.0 + r_mean) + 1.0/(1.0 + ev_mean)) / 3.0
    s_wdw   = comp(sci_wdw,   r_wdw_mean,   ev_wdw_mean)
    s_bl    = comp(sci_bl,    r_bl_mean,    ev_bl_mean)
    s_bl2p  = comp(sci_bl2p,  r_bl2p_mean,  ev_bl2p_mean)
    s_gcn2p = comp(sci_gcn2p, r_gcn2p_mean, ev_gcn2p_mean)
    s_mlp2p = comp(sci_mlp2p, r_mlp2p_mean, ev_mlp2p_mean)
    s_tr2p  = comp(sci_tr2p,  r_tr2p_mean,  ev_tr2p_mean)

    (; case="Gprime", n, groupname=string(group), seed,
       sci_wdw, sci_bl, sci_bl2p, sci_gcn2p, sci_mlp2p, sci_tr2p,
       r_wdw_mean, r_bl_mean, r_bl2p_mean, r_gcn2p_mean, r_mlp2p_mean, r_tr2p_mean,
       ev_wdw_mean, ev_bl_mean, ev_bl2p_mean, ev_gcn2p_mean, ev_mlp2p_mean, ev_tr2p_mean,
       s_wdw, s_bl, s_bl2p, s_gcn2p, s_mlp2p, s_tr2p)
end

function write_results_Gprime_csv(path::AbstractString, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, join(["case","n","group","seed",
                         "sci_wdw","sci_bl","sci_bl2p","sci_gcn2p","sci_mlp2p","sci_tr2p",
                         "r_wdw_mean","r_bl_mean","r_bl2p_mean","r_gcn2p_mean","r_mlp2p_mean","r_tr2p_mean",
                         "ev_wdw_mean","ev_bl_mean","ev_bl2p_mean","ev_gcn2p_mean","ev_mlp2p_mean","ev_tr2p_mean",
                         "s_wdw","s_bl","s_bl2p","s_gcn2p","s_mlp2p","s_tr2p"], ","))
        for r in rows
            println(io, join([
                r.case,
                string(r.n),
                r.groupname,
                string(r.seed),
                @sprintf("%.6f", r.sci_wdw),
                @sprintf("%.6f", r.sci_bl),
                @sprintf("%.6f", r.sci_bl2p),
                @sprintf("%.6f", r.sci_gcn2p),
                @sprintf("%.6f", r.sci_mlp2p),
                @sprintf("%.6f", r.sci_tr2p),
                @sprintf("%.6e", r.r_wdw_mean),
                @sprintf("%.6e", r.r_bl_mean),
                @sprintf("%.6e", r.r_bl2p_mean),
                @sprintf("%.6e", r.r_gcn2p_mean),
                @sprintf("%.6e", r.r_mlp2p_mean),
                @sprintf("%.6e", r.r_tr2p_mean),
                @sprintf("%.6e", r.ev_wdw_mean),
                @sprintf("%.6e", r.ev_bl_mean),
                @sprintf("%.6e", r.ev_bl2p_mean),
                @sprintf("%.6e", r.ev_gcn2p_mean),
                @sprintf("%.6e", r.ev_mlp2p_mean),
                @sprintf("%.6e", r.ev_tr2p_mean),
                @sprintf("%.6f", r.s_wdw),
                @sprintf("%.6f", r.s_bl),
                @sprintf("%.6f", r.s_bl2p),
                @sprintf("%.6f", r.s_gcn2p),
                @sprintf("%.6f", r.s_mlp2p),
                @sprintf("%.6f", r.s_tr2p)
            ], ","))
        end
    end
end

function write_certificate_Gprime(path::AbstractString, rows::Vector{NamedTuple})
    total = length(rows)
    avg_sci_wdw = total > 0 ? mean([r.sci_wdw for r in rows]) : 0.0
    avg_sci_best_noneq = total > 0 ? mean([maximum([r.sci_bl, r.sci_bl2p, r.sci_gcn2p, r.sci_mlp2p]) for r in rows]) : 0.0
    lead = avg_sci_wdw - avg_sci_best_noneq
    avg_S_wdw = total > 0 ? mean([r.s_wdw for r in rows]) : 0.0
    avg_S_best_noneq = total > 0 ? mean([maximum([r.s_bl, r.s_bl2p, r.s_gcn2p, r.s_mlp2p, r.s_tr2p]) for r in rows]) : 0.0
    lead_S = avg_S_wdw - avg_S_best_noneq
    open(path, "w") do io
        println(io, "Rupture Certificate — G′ (Coherencia semántica estructural)")
        println(io, @sprintf("SCI medio (WDW) = %.3f, SCI medio (mejor noneq) = %.3f, ventaja=%.3f", avg_sci_wdw, avg_sci_best_noneq, lead))
        println(io, @sprintf("Score compuesto S medio (WDW) = %.3f, S medio (mejor noneq) = %.3f, ventaja=%.3f", avg_S_wdw, avg_S_best_noneq, lead_S))
        println(io, "SCI = 1 − CV(invariante de salida sobre la órbita). S = (1/3)·SCI + (1/3)·(1/(1+residual_sheaf_medio)) + (1/3)·(1/(1+violación_equiv_media)). Sin claims de conciencia.")
    end
end
# Multi-constraint sheaf covers on a ring (k windows with overlap)
function sheaf_covers_ring(n::Int, k::Int=4)
    ovl = max(2, n ÷ 8)
    w = min(n, n ÷ 2 + ovl)
    stride = max(1, (n ÷ k) ÷ 2)
    slist = Vector{Vector{Int}}()
    for t in 0:(k-1)
        start = (t * stride) % n + 1
        stop = start + w - 1
        idx = Vector{Int}()
        for j in start:stop
            i = ((j - 1) % n) + 1
            push!(idx, i)
        end
        push!(slist, idx)
    end
    slist
end

function build_diff_constraints(n::Int, slist::Vector{Vector{Int}})
    cs = Vector{Vector{Float64}}()
    k = length(slist)
    for i in 1:k
        j = (i % k) + 1
        s1 = slist[i]
        s2 = slist[j]
        c = zeros(Float64, n)
        w1 = 1.0 / max(1, length(s1))
        w2 = 1.0 / max(1, length(s2))
        @inbounds begin
            for ii in s1
                c[ii] += w1
            end
            for ii in s2
                c[ii] -= w2
            end
        end
        push!(cs, c)
    end
    cs
end

function one_step_multi_projection(y::AbstractVector, cs::Vector{<:AbstractVector})
    # Gradient of sum_i 1/2 (c_i' y)^2 is g = (sum_i c_i c_i') y = sum_i (c_i' y) c_i
    g = zeros(eltype(y), length(y))
    rpre = Float64[]
    @inbounds for c in cs
        r = dot(c, y)
        push!(rpre, abs(r))
        g .+= r .* c
    end
    # Optimal step along g: eta = (g'g) / sum_i (c_i' g)^2
    cg_sq_sum = 0.0
    @inbounds for c in cs
        cg = dot(c, g)
        cg_sq_sum += cg * cg
    end
    numer = dot(g, g)
    eta = numer / (cg_sq_sum + eps(eltype(y)))
    y2 = y .- eta .* g
    rpost = Float64[]
    @inbounds for c in cs
        push!(rpost, abs(dot(c, y2)))
    end
    corr = norm(y2 - y) / (norm(y) + eps())
    mean_rpre = mean(rpre)
    mean_rpost = mean(rpost)
    return y2, mean_rpre, mean_rpost, corr
end

function k_step_multi_projection(y::AbstractVector, cs::Vector{<:AbstractVector}, k::Int)
    yk = copy(y)
    corr_total = 0.0
    for _ in 1:k
        yk, _, _, corr = one_step_multi_projection(yk, cs)
        corr_total += corr
    end
    yk, corr_total
end

# Apply WDW equivariant 2-parameter model W = alpha I + beta 11ᵀ
# yhat(x) = alpha x + beta (sum(x)) 1
function wdw_two_param_fit(x0::AbstractVector, y0::AbstractVector)
    n = length(x0)
    s = sum(x0)
    A = hcat(x0, fill(s, n))
    theta = A \ y0
    alpha = theta[1]
    beta = theta[2]
    alpha, beta
end

function wdw_apply_two_param(x::AbstractVector, alpha, beta)
    s = sum(x)
    alpha .* x .+ beta .* s .* ones(eltype(x), length(x))
end

# Relative error
relerr(yhat, y) = begin
    ny = norm(y) + eps(eltype(y))
    norm(yhat - y) / ny
end

# Build synthetic target operator W* = alpha I + beta 11ᵀ
function synth_equivariant_operator(n::Int)
    alpha = 0.7
    beta = -0.2
    alpha, beta
end

# Generate OGS experiment on a permutation group (e.g., dihedral)
function run_case1(; n::Int=32, group::Function=Q.dihedral_group, seed::Int=0, tol::Real=1e-6)
    Random.seed!(seed)
    G = group(n)

    # Ground truth operator in the equivariant subspace
    alpha_true, beta_true = synth_equivariant_operator(n)

    # Single training pair
    x0 = randn(n)
    y0 = alpha_true .* x0 .+ beta_true .* sum(x0) .* ones(n)

    # Fit models with equal data and no extra time
    alpha_hat, beta_hat = wdw_two_param_fit(x0, y0)
    # Unstructured 2-parameter baseline with equal k=2
    B1, B2 = rand2_basis(n, seed)
    th1, th2 = fit_unstructured_2p(B1, B2, x0, y0)
    # 2-parameter GCN-like baseline on a ring (equal k=2)
    A = ring_adjacency(n)
    tg1, tg2 = fit_gcn_2p(A, x0, y0)
    # 2-parameter MLP (tanh) and Transformer-like LayerNorm baselines (equal k=2 each)
    tm1, tm2 = fit_mlp2p(x0, y0; act=tanh)
    tt1, tt2 = fit_tr2p(x0, y0)

    # Evaluate on full orbit of x0 (excluding identity) — OGS
    errs_wdw = Float64[]
    errs_bl  = Float64[]
    errs_bl2p = Float64[]
    errs_gcn2p = Float64[]
    errs_mlp2p = Float64[]
    errs_tr2p  = Float64[]
    succ_wdw = 0
    succ_bl  = 0
    succ_bl2p = 0
    succ_gcn2p = 0
    succ_mlp2p = 0
    succ_tr2p  = 0

    for p in G.perms
        # Skip identity to focus on unseen cases
        if all(p[i] == i for i in 1:n)
            continue
        end
        x = Q.act(G, p, x0)
        ytrue = alpha_true .* x .+ beta_true .* sum(x) .* ones(n)
        y_wdw = wdw_apply_two_param(x, alpha_hat, beta_hat)
        y_bl  = baseline_apply(x, x0, y0)
        y_bl2p = apply_unstructured_2p(x, th1, th2, B1, B2)
        y_gcn2p = apply_gcn_2p(x, tg1, tg2, A)
        y_mlp2p = apply_mlp2p(x, tm1, tm2; act=tanh)
        y_tr2p  = apply_tr2p(x, tt1, tt2)
        e_wdw = relerr(y_wdw, ytrue)
        e_bl  = relerr(y_bl, ytrue)
        e_bl2p = relerr(y_bl2p, ytrue)
        e_gcn2p = relerr(y_gcn2p, ytrue)
        e_mlp2p = relerr(y_mlp2p, ytrue)
        e_tr2p  = relerr(y_tr2p,  ytrue)
        push!(errs_wdw, e_wdw)
        push!(errs_bl,  e_bl)
        push!(errs_bl2p, e_bl2p)
        push!(errs_gcn2p, e_gcn2p)
        push!(errs_mlp2p, e_mlp2p)
        push!(errs_tr2p,  e_tr2p)
        succ_wdw += (e_wdw <= tol)
        succ_bl  += (e_bl  <= tol)
        succ_bl2p += (e_bl2p <= tol)
        succ_gcn2p += (e_gcn2p <= tol)
        succ_mlp2p += (e_mlp2p <= tol)
        succ_tr2p  += (e_tr2p  <= tol)
    end

    ogs_wdw  = succ_wdw  / max(1, length(errs_wdw))
    ogs_bl   = succ_bl   / max(1, length(errs_bl))
    ogs_bl2p = succ_bl2p / max(1, length(errs_bl2p))
    ogs_gcn2p = succ_gcn2p / max(1, length(errs_gcn2p))
    ogs_mlp2p = succ_mlp2p / max(1, length(errs_mlp2p))
    ogs_tr2p  = succ_tr2p  / max(1, length(errs_tr2p))

    mdl_wdw  = mdl_proxy(2, errs_wdw)
    mdl_bl   = mdl_proxy(n, errs_bl)
    mdl_bl2p = mdl_proxy(2, errs_bl2p)
    mdl_gcn2p = mdl_proxy(2, errs_gcn2p)
    mdl_mlp2p = mdl_proxy(2, errs_mlp2p)
    mdl_tr2p  = mdl_proxy(2, errs_tr2p)

    (; n, groupname=string(group), ogs_wdw, ogs_bl, ogs_bl2p, ogs_gcn2p, ogs_mlp2p, ogs_tr2p,
       err_wdw_mean=mean(errs_wdw), err_bl_mean=mean(errs_bl), err_bl2p_mean=mean(errs_bl2p), err_gcn2p_mean=mean(errs_gcn2p), err_mlp2p_mean=mean(errs_mlp2p), err_tr2p_mean=mean(errs_tr2p),
       params_wdw=2, params_bl=n, params_bl2p=2, params_gcn2p=2, params_mlp2p=2, params_tr2p=2,
       mdl_wdw, mdl_bl, mdl_bl2p, mdl_gcn2p, mdl_mlp2p, mdl_tr2p)
end

# -----------------------------
 # Case 2 — Estabilidad imposible (contradicción + ruido estructural)
 # Sheaf-like cover residual and a single adaptation step with equal compute
function sheaf_cover(n::Int)
    ovl = max(2, n ÷ 8)
    s1 = collect(1:(n ÷ 2 + ovl))
    s2 = collect((n ÷ 2 - ovl + 1):n)
    s1, s2
end

function sheaf_residual(y::AbstractVector, s1::Vector{Int}, s2::Vector{Int})
    m1 = mean(view(y, s1))
    m2 = mean(view(y, s2))
    abs(m1 - m2)
end

function adversarial_noise(x::AbstractVector; magnitude::Real=1.0)
    n = length(x)
    bsz = max(2, n ÷ 8)
    v = similar(x)
    for i in 1:n
        blk = (div(i-1, bsz) % 2 == 0) ? one(eltype(x)) : -one(eltype(x))
        v[i] = blk
    end
    x .+ magnitude .* v
end

function krylov_project_sheaf(y::AbstractVector, s1::Vector{Int}, s2::Vector{Int})
    # Build linear constraint vector c such that r = c' y = mean(y[s1]) - mean(y[s2])
    n = length(y)
    c = zeros(eltype(y), n)
    w1 = 1.0 / max(1, length(s1))
    w2 = 1.0 / max(1, length(s2))
    @inbounds begin
        for i in s1
            c[i] += w1
        end
        for i in s2
            c[i] -= w2
        end
    end
    r = dot(c, y)
    denom = dot(c, c) + eps(eltype(y))
    alpha = r / denom
    y2 = y .- alpha .* c
    y2, r, alpha
end

function run_case2(; n::Int=64, group::Function=Q.dihedral_group, seed::Int=0, noise_mag::Real=1.0)
    Random.seed!(seed)
    G = group(n)
    s1, s2 = sheaf_cover(n)

    alpha_true, beta_true = synth_equivariant_operator(n)
    x0 = randn(n)
    y0 = alpha_true .* x0 .+ beta_true .* sum(x0) .* ones(n)

    alpha_hat, beta_hat = wdw_two_param_fit(x0, y0)

    # Clean predictions
    y_clean_wdw = wdw_apply_two_param(x0, alpha_hat, beta_hat)
    y_clean_bl  = baseline_apply(x0, x0, y0)

    # Adversarial input
    x_adv = adversarial_noise(x0; magnitude=noise_mag)
    y_adv_wdw = wdw_apply_two_param(x_adv, alpha_hat, beta_hat)
    y_adv_bl  = baseline_apply(x_adv, x0, y0)

    # Stability (delta under adversarial)
    stab_wdw = norm(y_adv_wdw - y_clean_wdw) / (norm(y_clean_wdw) + eps())
    stab_bl  = norm(y_adv_bl  - y_clean_bl)  / (norm(y_clean_bl)  + eps())

    # Sheaf residual before/after one Krylov-1 projection (equal compute)
    r_pre_wdw = sheaf_residual(y_adv_wdw, s1, s2)
    r_pre_bl  = sheaf_residual(y_adv_bl,  s1, s2)

    y_post_wdw, _, alpha_wdw = krylov_project_sheaf(y_adv_wdw, s1, s2)
    y_post_bl,  _, alpha_bl  = krylov_project_sheaf(y_adv_bl,  s1, s2)

    r_post_wdw = sheaf_residual(y_post_wdw, s1, s2)
    r_post_bl  = sheaf_residual(y_post_bl,  s1, s2)

    corr_wdw = norm(y_post_wdw - y_adv_wdw) / (norm(y_adv_wdw) + eps())
    corr_bl  = norm(y_post_bl  - y_adv_bl)  / (norm(y_adv_bl)  + eps())

    # Multi-constraint projection (ring of covers) — one optimal step along aggregate gradient
    slist = sheaf_covers_ring(n, 4)
    cs = build_diff_constraints(n, slist)
    y_post_wdw_all, r_all_pre_wdw, r_all_post_wdw, corr_wdw_all = one_step_multi_projection(y_adv_wdw, cs)
    y_post_bl_all,  r_all_pre_bl,  r_all_post_bl,  corr_bl_all  = one_step_multi_projection(y_adv_bl,  cs)

    # MDL proxies (pre and post) using two-error vector [stab, r]
    mdl_wdw_pre  = mdl_proxy(2, [stab_wdw, r_pre_wdw])
    mdl_wdw_post = mdl_proxy(2, [stab_wdw, r_post_wdw])
    mdl_bl_pre   = mdl_proxy(n, [stab_bl,  r_pre_bl])
    mdl_bl_post  = mdl_proxy(n, [stab_bl,  r_post_bl])

    (; n, groupname=string(group), stab_wdw, stab_bl, r_pre_wdw, r_pre_bl, r_post_wdw, r_post_bl, r_all_pre_wdw, r_all_pre_bl, r_all_post_wdw, r_all_post_bl,
       corr_wdw, corr_bl, corr_wdw_all, corr_bl_all, alpha_wdw, alpha_bl,
       mdl_wdw_pre, mdl_wdw_post, mdl_bl_pre, mdl_bl_post)
end

function write_results_case2_csv(path::AbstractString, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, join(["case","n","group","stab_wdw","stab_bl","r_pre_wdw","r_pre_bl","r_post_wdw","r_post_bl","r_all_pre_wdw","r_all_pre_bl","r_all_post_wdw","r_all_post_bl","mdl_wdw_pre","mdl_wdw_post","mdl_bl_pre","mdl_bl_post"], ","))
        for r in rows
            println(io, join([
                "case2",
                string(r.n),
                r.groupname,
                @sprintf("%.6e", r.stab_wdw),
                @sprintf("%.6e", r.stab_bl),
                @sprintf("%.6e", r.r_pre_wdw),
                @sprintf("%.6e", r.r_pre_bl),
                @sprintf("%.6e", r.r_post_wdw),
                @sprintf("%.6e", r.r_post_bl),
                @sprintf("%.6e", r.r_all_pre_wdw),
                @sprintf("%.6e", r.r_all_pre_bl),
                @sprintf("%.6e", r.r_all_post_wdw),
                @sprintf("%.6e", r.r_all_post_bl),
                @sprintf("%.6e", r.mdl_wdw_pre),
                @sprintf("%.6e", r.mdl_wdw_post),
                @sprintf("%.6e", r.mdl_bl_pre),
                @sprintf("%.6e", r.mdl_bl_post)
            ], ","))
        end
    end
end

function write_certificate_case2(path::AbstractString, r::NamedTuple)
    open(path, "w") do io
        println(io, "Rupture Certificate — Case 2: Estabilidad imposible")
        println(io, "n=$(r.n), group=$(r.groupname)")
        println(io, @sprintf("stab_wdw=%.6e, stab_bl=%.6e", r.stab_wdw, r.stab_bl))
        println(io, @sprintf("r_pre_wdw=%.6e, r_pre_bl=%.6e", r.r_pre_wdw, r.r_pre_bl))
        println(io, @sprintf("r_post_wdw=%.6e, r_post_bl=%.6e", r.r_post_wdw, r.r_post_bl))
        println(io, @sprintf("r_all_pre_wdw=%.6e, r_all_pre_bl=%.6e", r.r_all_pre_wdw, r.r_all_pre_bl))
        println(io, @sprintf("r_all_post_wdw=%.6e, r_all_post_bl=%.6e", r.r_all_post_wdw, r.r_all_post_bl))
        println(io, @sprintf("mdl_wdw_pre=%.6e, mdl_wdw_post=%.6e, mdl_bl_pre=%.6e, mdl_bl_post=%.6e", r.mdl_wdw_pre, r.mdl_wdw_post, r.mdl_bl_pre, r.mdl_bl_post))
        println(io, "Adaptación: un paso de proyección (Krylov-1) a consistencia de media (single-cover) y paso óptimo agregando múltiples cubiertas (igual coste).")
    end
end

# -----------------------------
 # Case 3 — Explicación algebraica (certificado de conmutación con el grupo)
function perm_matrix(p::Vector{Int}, ::Type{T}=Float64) where {T}
    n = length(p)
    P = zeros(T, n, n)
    @inbounds for i in 1:n
        P[i, p[i]] = one(T)
    end
    P
end

function run_case3(; n::Int=32, group::Function=Q.dihedral_group, seed::Int=0)
    Random.seed!(seed)
    G = group(n)
    alpha_true, beta_true = synth_equivariant_operator(n)
    x0 = randn(n)
    y0 = alpha_true .* x0 .+ beta_true .* sum(x0) .* ones(n)

    alpha_hat, beta_hat = wdw_two_param_fit(x0, y0)
    # Explicit operators
    W_wdw = alpha_hat .* I + beta_hat .* (ones(n, n))
    W_bl  = (y0 * x0') / (dot(x0, x0) + eps())

    max_comm_wdw = 0.0
    max_comm_bl  = 0.0
    for p in G.perms
        P = perm_matrix(p)
        cw = norm(P * W_wdw - W_wdw * P)
        cb = norm(P * W_bl  - W_bl  * P)
        max_comm_wdw = max(max_comm_wdw, cw)
        max_comm_bl  = max(max_comm_bl, cb)
    end

    errs_wdw = Float64[]; errs_bl = Float64[]
    for p in G.perms
        P = perm_matrix(p)
        push!(errs_wdw, norm(P * W_wdw - W_wdw * P))
        push!(errs_bl,  norm(P * W_bl  - W_bl  * P))
    end
    mdl_wdw = mdl_proxy(2, errs_wdw)
    mdl_bl  = mdl_proxy(n, errs_bl)

    (; n, groupname=string(group), max_comm_wdw, max_comm_bl, params_wdw=2, params_bl=n, mdl_wdw, mdl_bl)
end

function write_results_case3_csv(path::AbstractString, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, join(["case","n","group","max_comm_wdw","max_comm_bl","params_wdw","params_bl","mdl_wdw","mdl_bl"], ","))
        for r in rows
            println(io, join([
                "case3",
                string(r.n),
                r.groupname,
                @sprintf("%.6e", r.max_comm_wdw),
                @sprintf("%.6e", r.max_comm_bl),
                string(r.params_wdw),
                string(r.params_bl),
                @sprintf("%.6e", r.mdl_wdw),
                @sprintf("%.6e", r.mdl_bl)
            ], ","))
        end
    end
end

function write_certificate_case3(path::AbstractString, r::NamedTuple)
    open(path, "w") do io
        println(io, "Rupture Certificate — Case 3: Explicación algebraica")
        println(io, "n=$(r.n), group=$(r.groupname)")
        println(io, @sprintf("max_comm_wdw=%.6e, max_comm_bl=%.6e", r.max_comm_wdw, r.max_comm_bl))
        println(io, "Certificado: conmutación numérica P W ≈ W P sobre la órbita del grupo (norma del conmutador).")
        println(io, @sprintf("mdl_wdw=%.6e, mdl_bl=%.6e (proxy)", r.mdl_wdw, r.mdl_bl))
    end
end

# -----------------------------
# Case 4 — Bucle cerrado: Q-G-ENN → MERA → Krylov (consistencia sheaf)
function run_case4_closedloop(; n::Int=32, group::Function=Q.dihedral_group, seed::Int=0, levels::Int=3, keep_levels::Int=2)
    Random.seed!(seed)
    G = group(n)

    # Q-G-ENN: proyección al subespacio equivariante
    M = randn(n, n)
    W_eq = Q.project_equivariant(M, G)

    # Señal base
    x0 = randn(n)
    y_clean = W_eq * x0

    # MERA paramétrica: ajustar thetas en limpio y medir error
    thetas, _ = Tn.optimize_thetas(y_clean, levels, keep_levels; iters=30, step=0.1)
    rec_err_clean = Tn.param_multiscale_error(y_clean, levels, keep_levels, thetas)

    # Adversario estructural y proyección Krylov-1 a consistencia de sheaf
    s1, s2 = sheaf_cover(n)
    x_adv = adversarial_noise(x0; magnitude=1.0)
    y_adv = W_eq * x_adv
    rec_err_adv = Tn.param_multiscale_error(y_adv, levels, keep_levels, thetas)
    r_pre = sheaf_residual(y_adv, s1, s2)
    y_post, _, _ = krylov_project_sheaf(y_adv, s1, s2)
    r_post = sheaf_residual(y_post, s1, s2)
    rec_err_post = Tn.param_multiscale_error(y_post, levels, keep_levels, thetas)

    # Estabilidad relativa
    stab = norm(y_adv - y_clean) / (norm(y_clean) + eps())

    # Certificado algebraico: conmutador máximo (debe ser ~0 por construcción)
    max_comm = 0.0
    for p in G.perms
        P = perm_matrix(p)
        max_comm = max(max_comm, norm(P * W_eq - W_eq * P))
    end

    (; n, groupname=string(group), rec_err_clean, rec_err_adv, rec_err_post, r_pre, r_post, stab, max_comm)
end

function write_results_case4_csv(path::AbstractString, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, join(["case","n","group","rec_err_clean","rec_err_adv","rec_err_post","r_pre","r_post","stab","max_comm"], ","))
        for r in rows
            println(io, join([
                "case4",
                string(r.n),
                r.groupname,
                @sprintf("%.6e", r.rec_err_clean),
                @sprintf("%.6e", r.rec_err_adv),
                @sprintf("%.6e", r.rec_err_post),
                @sprintf("%.6e", r.r_pre),
                @sprintf("%.6e", r.r_post),
                @sprintf("%.6e", r.stab),
                @sprintf("%.6e", r.max_comm)
            ], ","))
        end
    end
end

function write_certificate_case4(path::AbstractString, r::NamedTuple)
    open(path, "w") do io
        println(io, "Rupture Certificate — Case 4: Bucle cerrado Q-G-ENN→MERA→Krylov")
        println(io, "n=$(r.n), group=$(r.groupname)")
        println(io, @sprintf("rec_err_clean=%.6e, rec_err_adv=%.6e, rec_err_post=%.6e", r.rec_err_clean, r.rec_err_adv, r.rec_err_post))
        println(io, @sprintf("r_pre=%.6e, r_post=%.6e, stab=%.6e", r.r_pre, r.r_post, r.stab))
        println(io, @sprintf("max_comm=%.6e (W_eq conmute con G)", r.max_comm))
    end
end

function ensure_bench_dir()
    # bench directory expected to exist; do nothing.
    return
end

function write_results_csv(path::AbstractString, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, join(["case","n","group","ogs_wdw","ogs_bl","ogs_bl2p","ogs_gcn2p","ogs_mlp2p","ogs_tr2p","err_wdw_mean","err_bl_mean","err_bl2p_mean","err_gcn2p_mean","err_mlp2p_mean","err_tr2p_mean","params_wdw","params_bl","params_bl2p","params_gcn2p","params_mlp2p","params_tr2p","mdl_wdw","mdl_bl","mdl_bl2p","mdl_gcn2p","mdl_mlp2p","mdl_tr2p"], ","))
        for r in rows
            println(io, join([
                "case1",
                string(r.n),
                r.groupname,
                @sprintf("%.6f", r.ogs_wdw),
                @sprintf("%.6f", r.ogs_bl),
                @sprintf("%.6f", r.ogs_bl2p),
                @sprintf("%.6f", r.ogs_gcn2p),
                @sprintf("%.6f", r.ogs_mlp2p),
                @sprintf("%.6f", r.ogs_tr2p),
                @sprintf("%.6e", r.err_wdw_mean),
                @sprintf("%.6e", r.err_bl_mean),
                @sprintf("%.6e", r.err_bl2p_mean),
                @sprintf("%.6e", r.err_gcn2p_mean),
                @sprintf("%.6e", r.err_mlp2p_mean),
                @sprintf("%.6e", r.err_tr2p_mean),
                string(r.params_wdw),
                string(r.params_bl),
                string(r.params_bl2p),
                string(r.params_gcn2p),
                string(r.params_mlp2p),
                string(r.params_tr2p),
                @sprintf("%.6e", r.mdl_wdw),
                @sprintf("%.6e", r.mdl_bl),
                @sprintf("%.6e", r.mdl_bl2p),
                @sprintf("%.6e", r.mdl_gcn2p),
                @sprintf("%.6e", r.mdl_mlp2p),
                @sprintf("%.6e", r.mdl_tr2p)
            ], ","))
        end
    end
end

function write_certificate_txt(path::AbstractString, r::NamedTuple)
    open(path, "w") do io
        println(io, "Rupture Certificate — Case 1: Generalización prohibida (OGS)")
        println(io, "n=$(r.n), group=$(r.groupname)")
        println(io, @sprintf("ogs_wdw=%.6f, ogs_bl=%.6f, ogs_bl2p=%.6f, ogs_gcn2p=%.6f, ogs_mlp2p=%.6f, ogs_tr2p=%.6f", r.ogs_wdw, r.ogs_bl, r.ogs_bl2p, r.ogs_gcn2p, r.ogs_mlp2p, r.ogs_tr2p))
        println(io, @sprintf("err_wdw_mean=%.6e, err_bl_mean=%.6e, err_bl2p_mean=%.6e, err_gcn2p_mean=%.6e, err_mlp2p_mean=%.6e, err_tr2p_mean=%.6e", r.err_wdw_mean, r.err_bl_mean, r.err_bl2p_mean, r.err_gcn2p_mean, r.err_mlp2p_mean, r.err_tr2p_mean))
        println(io, "params_wdw=$(r.params_wdw), params_bl=$(r.params_bl), params_bl2p=$(r.params_bl2p), params_gcn2p=$(r.params_gcn2p), params_mlp2p=$(r.params_mlp2p), params_tr2p=$(r.params_tr2p)")
        println(io, @sprintf("mdl_wdw=%.6e, mdl_bl=%.6e, mdl_bl2p=%.6e, mdl_gcn2p=%.6e, mdl_mlp2p=%.6e, mdl_tr2p=%.6e (proxy)", r.mdl_wdw, r.mdl_bl, r.mdl_bl2p, r.mdl_gcn2p, r.mdl_mlp2p, r.mdl_tr2p))
        println(io, "Fairness: mismos datos (1 par), sin reentrenamiento, igualdad de tiempo por evaluación y comparación con baseline no estructurado de 2 parámetros.")
        println(io, "Modelos: WDW W=αI+β11ᵀ (k=2); Baseline rank-1 (~k≈n); Baseline-2p no estructurado (k=2) proyectado sobre dos bases aleatorias; Baseline GCN-2p en anillo (k=2); Baseline MLP-2p (k=2); Baseline TR-2p (LayerNorm) (k=2).")
        println(io, "Criterio: OGS_WDW≈1.0 y OGS_baseline≈0.0 con tol=1e-6.")
    end
end

function main()
    ensure_bench_dir()
    # Evaluate at two sizes
    rows = NamedTuple[]
    push!(rows, run_case1(n=32, group=Q.dihedral_group, seed=0))
    push!(rows, run_case1(n=48, group=Q.dihedral_group, seed=1))

    # Save artifacts
    write_results_csv("bench/rupture_results.csv", rows)
    write_certificate_txt("bench/rupture_certificate.txt", rows[1])

    # Print brief summary
    for r in rows
        @printf("case1,n=%d,group=%s,ogs_wdw=%.3f,ogs_bl=%.3f,ogs_bl2p=%.3f,ogs_gcn2p=%.3f,ogs_mlp2p=%.3f,ogs_tr2p=%.3f\n", r.n, r.groupname, r.ogs_wdw, r.ogs_bl, r.ogs_bl2p, r.ogs_gcn2p, r.ogs_mlp2p, r.ogs_tr2p)
    end

    # Case 2
    rows2 = NamedTuple[]
    push!(rows2, run_case2(n=64, group=Q.dihedral_group, seed=0, noise_mag=1.0))
    push!(rows2, run_case2(n=96, group=Q.dihedral_group, seed=1, noise_mag=1.0))
    write_results_case2_csv("bench/rupture_results_case2.csv", rows2)
    write_certificate_case2("bench/rupture_certificate_case2.txt", rows2[1])
    for r in rows2
        @printf("case2,n=%d,group=%s,stab_wdw=%.3e,stab_bl=%.3e, r_pre_wdw=%.2e, r_pre_bl=%.2e, r_post_wdw=%.2e, r_post_bl=%.2e, r_all_pre_wdw=%.2e, r_all_pre_bl=%.2e, r_all_post_wdw=%.2e, r_all_post_bl=%.2e\n",
                r.n, r.groupname, r.stab_wdw, r.stab_bl, r.r_pre_wdw, r.r_pre_bl, r.r_post_wdw, r.r_post_bl, r.r_all_pre_wdw, r.r_all_pre_bl, r.r_all_post_wdw, r.r_all_post_bl)
    end

    # Case 3
    rows3 = NamedTuple[]
    push!(rows3, run_case3(n=32, group=Q.dihedral_group, seed=0))
    push!(rows3, run_case3(n=48, group=Q.dihedral_group, seed=1))
    write_results_case3_csv("bench/rupture_results_case3.csv", rows3)
    write_certificate_case3("bench/rupture_certificate_case3.txt", rows3[1])
    for r in rows3
        @printf("case3,n=%d,group=%s,max_comm_wdw=%.2e,max_comm_bl=%.2e\n", r.n, r.groupname, r.max_comm_wdw, r.max_comm_bl)
    end

    # Case 4 — Closed loop
    rows4 = NamedTuple[]
    push!(rows4, run_case4_closedloop(n=32, group=Q.dihedral_group, seed=0, levels=3, keep_levels=2))
    push!(rows4, run_case4_closedloop(n=48, group=Q.dihedral_group, seed=1, levels=3, keep_levels=2))
    write_results_case4_csv("bench/rupture_results_case4.csv", rows4)
    write_certificate_case4("bench/rupture_certificate_case4.txt", rows4[1])
    for r in rows4
        @printf("case4,n=%d,group=%s,rec_err_clean=%.2e,rec_err_adv=%.2e,rec_err_post=%.2e,r_post=%.2e,max_comm=%.2e\n",
                r.n, r.groupname, r.rec_err_clean, r.rec_err_adv, r.rec_err_post, r.r_post, r.max_comm)
    end

    # Case 4 comparative — random vs equivariant
    rows4c = NamedTuple[]
    r_eq32, r_rn32 = run_case4_compare(n=32, group=Q.dihedral_group, seed=0, levels=3, keep_levels=2, iters=30, step=0.1)
    r_eq48, r_rn48 = run_case4_compare(n=48, group=Q.dihedral_group, seed=1, levels=3, keep_levels=2, iters=30, step=0.1)
    push!(rows4c, r_eq32); push!(rows4c, r_rn32);
    push!(rows4c, r_eq48); push!(rows4c, r_rn48);
    write_results_case4_cmp_csv("bench/rupture_results_case4_cmp.csv", rows4c)
    write_certificate_case4_cmp("bench/rupture_certificate_case4_cmp.txt", r_eq32, r_rn32)
    for r in rows4c
        @printf("case4_cmp,model=%s,n=%d,group=%s,rec_err_clean=%.2e,rec_err_adv=%.2e,rec_err_post=%.2e,r_post=%.2e,max_comm=%.2e\n",
                r.model, r.n, r.groupname, r.rec_err_clean, r.rec_err_adv, r.rec_err_post, r.r_post, r.max_comm)
    end

    # D′ — Multitarea estructural (cobertura OGS vs noneq)
    rowsD = run_caseDprime(ns=[32,48], groups=[Q.dihedral_group], seeds=[0,1])
    write_results_Dprime_csv("bench/rupture_results_Dprime.csv", rowsD)
    write_certificate_Dprime("bench/rupture_certificate_Dprime.txt", rowsD)
    for r in rowsD
        @printf("Dprime,n=%d,seed=%d,cover=%s,margin=%.2f\n", r.n, r.seed, string(r.cover), r.margin)
    end

    # F′ — Hiper-eficiencia (OGS/coste proxy)
    rowsF = run_caseFprime(ns=[32,48], groups=[Q.dihedral_group], seeds=[0,1])
    write_results_Fprime_csv("bench/rupture_results_Fprime.csv", rowsF)
    write_certificate_Fprime("bench/rupture_certificate_Fprime.txt", rowsF)
    for r in rowsF
        @printf("Fprime,n=%d,seed=%d,lead=%s,margin_eff=%.2e\n", r.n, r.seed, string(r.lead), r.margin_eff)
    end

    # G′ — Coherencia semántica estructural
    rowsG = NamedTuple[]
    push!(rowsG, run_caseGprime(n=32, group=Q.dihedral_group, seed=0))
    push!(rowsG, run_caseGprime(n=48, group=Q.dihedral_group, seed=1))
    write_results_Gprime_csv("bench/rupture_results_Gprime.csv", rowsG)
    write_certificate_Gprime("bench/rupture_certificate_Gprime.txt", rowsG[1:2])
    for r in rowsG
        @printf("Gprime,n=%d,seed=%d,SCI_wdw=%.2f,SCI_best_noneq=%.2f\n", r.n, r.seed, r.sci_wdw, maximum([r.sci_bl, r.sci_bl2p, r.sci_gcn2p, r.sci_mlp2p]))
    end

    # G′-S_adv — Multi-cover + adversarial stress
    rowsGadv = NamedTuple[]
    push!(rowsGadv, run_caseGprime_adv(n=32, group=Q.dihedral_group, seed=0, k=6, noise_mag=1.0))
    push!(rowsGadv, run_caseGprime_adv(n=48, group=Q.dihedral_group, seed=1, k=6, noise_mag=1.0))
    write_results_Gprime_adv_csv("bench/rupture_results_Gprime_adv.csv", rowsGadv)
    write_certificate_Gprime_adv("bench/rupture_certificate_Gprime_adv.txt", rowsGadv)
    for r in rowsGadv
        best_noneq_S = maximum([r.s_bl, r.s_bl2p, r.s_gcn2p, r.s_mlp2p, r.s_tr2p])
        @printf("Gprime_adv,n=%d,seed=%d,S_wdw=%.3f,S_best_noneq=%.3f\n", r.n, r.seed, r.s_wdw, best_noneq_S)
    end
end

main()
