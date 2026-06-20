#!/usr/bin/env julia
# WDW FFTPIPELINE — Certified Benchmark
# Outputs a certificate file for all verified claims.

using WDW, Printf, Statistics, Random, Dates, LinearAlgebra

const FP = WDW.FFTPipeline
const CERT_PATH = "bench/fft_pipeline/fft_certificate.txt"

function certify(; sizes=[16, 32, 64, 128], seeds=1:5)
    open(CERT_PATH, "w") do io
        println(io, "="^66)
        println(io, "  WDW FFT PIPELINE — CERTIFIED BENCHMARK")
        println(io, "  Date: $(now())")
        println(io, "="^66)

        # ── 1. Theory ──
        println(io, "\n── SECTION 1: THEORETICAL PROPERTIES ──")

        n = 32
        layer = WDW.FFTGroup.CyclicFourierLayer(n; seed=42)
        fill!(layer.A, 1.0 + 0.0im)
        rng = MersenneTwister(1)
        x = randn(rng, n); x /= norm(x)
        B_orig = WDW.FFTGroup.bispec_features(x, layer)
        max_shift_err = 0.0
        for k in [1, 5, 13, 31]
            x_shift = [x[mod1(i-k, n)] for i in 1:n]
            B_shift = WDW.FFTGroup.bispec_features(x_shift, layer)
            max_shift_err = max(max_shift_err, norm(B_orig - B_shift))
        end
        s1 = max_shift_err < 1e-10 ? "PASS" : "FAIL"
        @printf io "  Shift invariance error: %.2e (threshold <1e-10) [%s]\n" max_shift_err s1

        x_ref = [x[mod1(-i+2, n)] for i in 1:n]
        B_ref = WDW.FFTGroup.bispec_features(x_ref, layer)
        dn_diff = norm(B_orig - B_ref)
        s2 = dn_diff > 0.01 ? "PASS" : "FAIL"
        @printf io "  Dn sensitivity (diff=%.2f, threshold >0.01) [%s]\n" dn_diff s2

        x_rec = WDW.FFTGroup.exact_recovery(x, layer)
        rec_mse = mean(abs2, x - x_rec)
        s3 = rec_mse < 1e-15 ? "PASS" : "FAIL"
        @printf io "  Recovery MSE: %.2e (threshold <1e-15) [%s]\n" rec_mse s3

        # ── 2. Classification ──
        println(io, "\n── SECTION 2: ONE-SHOT (n=32, 4 classes, 1 sample/class) ──")
        c1_ok = true
        for seed in seeds
            xs_tr, ys_tr, xs_te, ys_te = FP.make_dataset(32, 2, 1, seed)
            p = FP.SignalPipeline(32; n_classes=4, n_pairs=2, seed=seed)
            FP.train_pipeline!(p, xs_tr, ys_tr; epochs=500)
            cn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_te, ys_te; dn=false)
            xs_dn = [FP.reflect(x) for x in xs_te]
            dn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_dn, ys_te; dn=false)
            mlp_a, _ = FP.mlp_baseline(xs_tr, ys_tr, xs_te, ys_te)
            @printf io "  seed=%d: Cn=%.1f  Dn=%.1f  gap=%.1fpp  MLP=%.1f\n" seed cn dn (cn-dn) mlp_a
            c1_ok &= (cn > 99 && cn-dn > 50 && mlp_a < cn)
        end
        @printf io "  Result: [%s]\n" (c1_ok ? "PASS" : "FAIL")

        # ── 3. Scalability ──
        println(io, "\n── SECTION 3: SCALABILITY ──")
        @printf io "  %-6s %-6s %-7s %-10s %-6s %-6s\n" "n" "Cn(%)" "gap(pp)" "MSE" "params" "MLP(%)"
        println(io, "  " * "-"^49)
        all_pass = true
        for n in sizes
            npairs = max(1, n ÷ 16)
            ncls = 2 * npairs
            shots = max(1, n ÷ 32)
            xs_tr, ys_tr, xs_te, ys_te = FP.make_dataset(n, npairs, shots, 42)
            p = FP.SignalPipeline(n; n_classes=ncls, n_pairs=npairs, seed=42)
            FP.train_pipeline!(p, xs_tr, ys_tr; epochs=500)
            cn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_te, ys_te; dn=false)
            xs_dn = [FP.reflect(x) for x in xs_te]
            dn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_dn, ys_te; dn=false)
            mse = mean(abs2, xs_te[1] - WDW.FFTGroup.exact_recovery(xs_te[1], p.layer))
            np = 2*n + n + 3*n*ncls + ncls
            mlp_a, _ = FP.mlp_baseline(xs_tr, ys_tr, xs_te, ys_te)
            pass_ = cn > 99 && cn-dn > 50 && mse < 1e-15 && mlp_a < cn
            all_pass &= pass_
            @printf io "  n=%-3d %5.1f  %5.1f   %.2e  %d  %5.1f  %s\n" n cn (cn-dn) mse np mlp_a (pass_ ? "OK" : "FAIL")
        end

        # ── Summary ──
        println(io, "\n" * "="^66)
        println(io, "  CERTIFICATION SUMMARY")
        println(io, "="^66)
        println(io, "  Shift invariance:    $(s1)")
        println(io, "  Dn sensitivity:      $(s2)")
        println(io, "  Recovery:            $(s3)")
        println(io, "  1-shot classif:      $(c1_ok ? "PASS" : "FAIL")")
        println(io, "  Scalability:         $(all_pass ? "PASS" : "FAIL")")
        overall = c1_ok && all_pass ? "ALL CLAIMS CERTIFIED" : "SOME FAILURES"
        println(io, "  Overall:             $(overall)")
        println(io, "="^66)
    end

    println("Certificate written to: $CERT_PATH")
    println(read(CERT_PATH, String))
end

certify()
