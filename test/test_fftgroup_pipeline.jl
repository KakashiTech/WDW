using Test
using WDW
using LinearAlgebra: norm
using Statistics: mean, std
using Random: MersenneTwister
using Zygote

# =============================================================================
# THEORY VERIFICATION TESTS
# =============================================================================

@testset "Bispectrum theory" begin
    n = 32
    # Use A=I (symmetric) to isolate bispectrum properties from A effects
    layer = WDW.FFTGroup.CyclicFourierLayer(n; seed=42)
    fill!(layer.A, 1.0 + 0.0im)
    rng = MersenneTwister(1)
    x = randn(rng, n); x /= norm(x)

    B_orig = WDW.FFTGroup.bispec_features(x, layer)

    @testset "1. Shift invariance (exact)" begin
        for k in [1, 7, 13, 31]
            x_shift = [x[mod1(i - k, n)] for i in 1:n]
            B_shift = WDW.FFTGroup.bispec_features(x_shift, layer)
            @test norm(B_orig - B_shift) < 1e-12
        end
    end

    @testset "2. Dₙ sensitivity (features change)" begin
        x_ref = [x[mod1(-i + 2, n)] for i in 1:n]
        B_ref = WDW.FFTGroup.bispec_features(x_ref, layer)
        @test norm(B_orig - B_ref) > 0.1
    end

    @testset "3. Recovery (exact to float64)" begin
        x_rec = WDW.FFTGroup.exact_recovery(x, layer)
        @test mean(abs2, x - x_rec) < 1e-15
    end

    @testset "4. A=I symmetry → cn_ne_dn_asymmetry = 0" begin
        asym = WDW.FFTGroup.cn_ne_dn_asymmetry(layer)
        @test asym < 1e-15
    end
end

# =============================================================================
# CLASSIFICATION TESTS
# =============================================================================

@testset "Classification" begin
    n = 32
    FP = WDW.FFTPipeline
    rng = MersenneTwister(42)

    function make_sig(seed)
        n2=n÷2; x̂=Complex{Float64}[]
        push!(x̂, randn(rng)*sqrt(n))
        for ω in 2:n2
            push!(x̂, abs(randn(rng))*sqrt(n/2)*exp(im*rand(rng)*2π))
        end
        n%2==0 && push!(x̂, randn(rng)*sqrt(n/2))
        for ω in n2+2:n; push!(x̂, conj(x̂[n-ω+2])); end
        x = real(WDW.FFTGroup.myifft(x̂))
        x .+= 0.05 * randn(rng, n)
        return x / norm(x)
    end

    function make_data(n_pairs, shots)
        xs_tr=Vector{Float64}[]; ys_tr=Int[]
        xs_te=Vector{Float64}[]; ys_te=Int[]
        for pair in 1:n_pairs
            base = make_sig(pair*100)
            rev = FP.reflect(base)
            for (ci,sig) in enumerate([base,rev])
                cls = 2*(pair-1)+ci
                for _ in 1:shots
                    push!(xs_tr, sig); push!(ys_tr, cls)
                end
                for _ in 1:50
                    push!(xs_te, FP.shift(sig, rand(rng, 0:n-1)))
                    push!(ys_te, cls)
                end
            end
        end
        xs_tr, ys_tr, xs_te, ys_te
    end

    @testset "5. 4-class 1-shot: 100% Cₙ, 100pp gap" begin
        xs_tr, ys_tr, xs_te, ys_te = make_data(2, 1)
        p = FP.SignalPipeline(32; n_classes=4, n_pairs=2, seed=42)
        FP.train_pipeline!(p, xs_tr, ys_tr; epochs=500)
        cn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_te, ys_te; dn=false)
        @test cn > 99.0
        xs_dn = [FP.reflect(x) for x in xs_te]
        dn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_dn, ys_te; dn=false)
        @test cn - dn > 50.0
        mse = mean(abs2, xs_te[1] - WDW.FFTGroup.exact_recovery(xs_te[1], p.layer))
        @test mse < 1e-15
    end

    @testset "6. Binary 2-sample: 100% Cₙ" begin
        sig1 = make_sig(100)
        sig2 = FP.reflect(sig1)
        xs_tr = [sig1, sig2]; ys_tr = [1, 2]
        xs_te = Vector{Float64}[]; ys_te = Int[]
        for _ in 1:100
            push!(xs_te, FP.shift(sig1, rand(rng, 0:31))); push!(ys_te, 1)
            push!(xs_te, FP.shift(sig2, rand(rng, 0:31))); push!(ys_te, 2)
        end
        p = FP.SignalPipeline(32; n_classes=2, n_pairs=1, seed=42)
        FP.train_pipeline!(p, xs_tr, ys_tr; epochs=500)
        cn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_te, ys_te; dn=false)
        @test cn > 99.0
    end

    @testset "7. Power spectrum baseline: ~50% accuracy" begin
        xs_tr, ys_tr, xs_te, ys_te = make_data(2, 4)
        ps_acc, _ = FP.power_spectrum_baseline(xs_tr, ys_tr, xs_te, ys_te; epochs=300)
        # Power spectrum is Dₙ-invariant → cannot distinguish pairs → ~50%
        @test ps_acc < 55.0
    end

    @testset "8. MLP baseline: cannot reach 100%" begin
        xs_tr, ys_tr, xs_te, ys_te = make_data(2, 4)
        mlp_acc, _ = FP.mlp_baseline(xs_tr, ys_tr, xs_te, ys_te; h=64, epochs=500)
        @test mlp_acc < 50.0
    end
end

# =============================================================================
# PIPELINE TESTS
# =============================================================================

@testset "Pipeline" begin
    FP = WDW.FFTPipeline

    @testset "9. make_signal" begin
        s = FP.make_signal(32; seed=42)
        @test length(s) == 32
        @test abs(norm(s) - 1.0) < 0.1
    end

    @testset "10. shift/reflect" begin
        x = collect(1.0:5.0)
        @test FP.shift(x, 1) ≈ [5.0, 1.0, 2.0, 3.0, 4.0]
        @test FP.reflect(FP.reflect(x)) ≈ x
    end

    @testset "11. make_dataset" begin
        xs_tr, ys_tr, xs_te, ys_te = FP.make_dataset(32, 2, 1, 42)
        @test length(xs_tr) == 4
        @test length(xs_te) == 200
        @test unique(ys_tr) == [1, 2, 3, 4]
    end

    @testset "12. evaluate_pipeline" begin
        xs_tr, ys_tr, xs_te, ys_te = FP.make_dataset(32, 2, 2, 42)
        p = FP.SignalPipeline(32; n_classes=4, n_pairs=2, seed=42)
        FP.train_pipeline!(p, xs_tr, ys_tr; epochs=300)
        r = FP.evaluate_pipeline(p, xs_te, ys_te)
        @test r.cn_acc > 99.0
        @test r.gap > 50.0
        @test r.mse < 1e-15
    end

    @testset "13. run_pipeline" begin
        r = FP.run_pipeline(n=16, n_classes=2, n_pairs=1, shots=2, epochs=200)
        @test r.cn > 99.0
        @test r.gap > 50.0
    end
end

# =============================================================================
# SCALABILITY TESTS
# =============================================================================

@testset "Scalability" begin
    FP = WDW.FFTPipeline
    for n in [16, 32]
        n_pairs = max(1, n ÷ 16)
        n_classes = 2 * n_pairs
        xs_tr, ys_tr, xs_te, ys_te = FP.make_dataset(n, n_pairs, 2, 42)
        p = FP.SignalPipeline(n; n_classes=n_classes, n_pairs=n_pairs, seed=42)
        FP.train_pipeline!(p, xs_tr, ys_tr; epochs=300)
        cn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_te, ys_te; dn=false)
        xs_dn = [FP.reflect(x) for x in xs_te]
        dn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_dn, ys_te; dn=false)
        @test cn > 99.0
        @test cn - dn > 50.0
    end
end
