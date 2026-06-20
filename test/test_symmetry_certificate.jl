using Test
using WDW
using LinearAlgebra, Random, Statistics

const SC = WDW.SymmetryCertificate
const FP = WDW.FFTPipeline

@testset "SymmetryCertificate" begin

    @testset "1. Dataset symmetry audit (ground truth)" begin
        xs_tr, ys_tr, _, _ = FP.make_dataset(32, 2, 4, 42)
        data_audit = SC.audit_dataset(xs_tr)
        profile, probes, n = data_audit
        @test n == 32
        @test length(profile) == 8
        @test length(probes) == 8

        # Shifts are exact symmetries
        for i in 1:5
            @test profile[i] < 1e-10
        end
        # Reflection breaks symmetry
        @test profile[6] > 0.1
    end

    @testset "2. Model symmetry audit" begin
        xs_tr, ys_tr, xs_te, ys_te = FP.make_dataset(32, 2, 4, 42)
        n = 32

        # Build a simple model function
        p = FP.SignalPipeline(n; n_classes=4, n_pairs=2, seed=42)
        FP.train_pipeline!(p, xs_tr, ys_tr; epochs=200)

        function model_fn(x)
            feats = WDW.FFTGroup.combined_bispec_features(x, p.layer)
            logits = p.Wc * feats + p.bc
            return (feats, logits)
        end

        # Audit
        audit = SC.audit_model(xs_tr, model_fn, ["features", "logits"], [3*32, 4])

        @test audit.n == 32
        @test audit.n_layers == 2
        @test length(audit.layer_names) == 2
        @test audit.layer_names[1] == "features"
        @test audit.layer_names[2] == "logits"
        @test length(audit.data_profile) == 8
        @test length(audit.layer_profiles) == 2
        @test length(audit.layer_divergences) == 2

        # Data profile: shifts are exact symmetries
        for i in 1:5
            @test audit.data_profile[i] < 1e-10
        end
    end

    @testset "3. Certificate generation" begin
        xs_tr, ys_tr, _, _ = FP.make_dataset(32, 2, 4, 42)
        n = 32
        p = FP.SignalPipeline(n; n_classes=4, n_pairs=2, seed=42)
        FP.train_pipeline!(p, xs_tr, ys_tr; epochs=200)

        function model_fn(x)
            feats = WDW.FFTGroup.combined_bispec_features(x, p.layer)
            logits = p.Wc * feats + p.bc
            return (feats, logits)
        end

        audit = SC.audit_model(xs_tr, model_fn, ["features", "logits"], [3*32, 4])
        cert = SC.generate_certificate(audit)

        @test cert isa SC.ModelCertificate{Float64}
        @test startswith(cert.certificate_id, "SC-")
        @test 0 <= cert.symmetry_fidelity <= 1.0
        @test 0 <= cert.deployability_score <= 1.0
        @test 0 <= cert.generalization_readiness <= 1.0
        @test length(cert.predicted_failure_modes) > 0
        @test length(cert.recommended_actions) > 0
    end

    @testset "4. Deployability score" begin
        xs_tr, ys_tr, _, _ = FP.make_dataset(32, 2, 4, 42)
        n = 32
        p = FP.SignalPipeline(n; n_classes=4, n_pairs=2, seed=42)
        FP.train_pipeline!(p, xs_tr, ys_tr; epochs=200)

        function model_fn(x)
            feats = WDW.FFTGroup.combined_bispec_features(x, p.layer)
            logits = p.Wc * feats + p.bc
            return (feats, logits)
        end

        audit = SC.audit_model(xs_tr, model_fn, ["features", "logits"], [3*32, 4])
        cert = SC.generate_certificate(audit)

        score = SC.deployability_score(cert)
        @test score isa Float64
        @test 0 <= score <= 1.0

        failures = SC.failure_modes(cert)
        @test failures isa Vector{String}
        @test length(failures) > 0
    end

    @testset "5. Quick audit one-call API" begin
        xs_tr, ys_tr, _, _ = FP.make_dataset(32, 2, 4, 42)
        n = 32
        p = FP.SignalPipeline(n; n_classes=4, n_pairs=2, seed=42)
        FP.train_pipeline!(p, xs_tr, ys_tr; epochs=200)

        function model_fn(x)
            feats = WDW.FFTGroup.combined_bispec_features(x, p.layer)
            logits = p.Wc * feats + p.bc
            return (feats, logits)
        end

        cert = SC.quick_audit(xs_tr, model_fn, ["features", "logits"], [3*32, 4])
        @test cert isa SC.ModelCertificate{Float64}
        @test 0 <= cert.deployability_score <= 1.0
    end

    @testset "6. Different layer counts" begin
        xs_tr, ys_tr, _, _ = FP.make_dataset(32, 2, 4, 42)
        n = 32
        p = FP.SignalPipeline(n; n_classes=4, n_pairs=2, seed=42)
        FP.train_pipeline!(p, xs_tr, ys_tr; epochs=200)

        # Single layer model
        function single_layer(x)
            feats = WDW.FFTGroup.combined_bispec_features(x, p.layer)
            return (feats,)
        end

        audit = SC.audit_model(xs_tr, single_layer, ["features"], [3*32])
        @test audit.n_layers == 1
        cert = SC.generate_certificate(audit)
        @test cert isa SC.ModelCertificate{Float64}
    end

    @testset "7. Certificate structure completeness" begin
        xs_tr, ys_tr, _, _ = FP.make_dataset(32, 2, 4, 42)
        n = 32
        p = FP.SignalPipeline(n; n_classes=4, n_pairs=2, seed=42)
        FP.train_pipeline!(p, xs_tr, ys_tr; epochs=200)

        function model_fn(x)
            feats = WDW.FFTGroup.combined_bispec_features(x, p.layer)
            logits = p.Wc * feats + p.bc
            return (feats, logits)
        end

        audit = SC.audit_model(xs_tr, model_fn, ["features", "logits"], [3*32, 4])
        cert = SC.generate_certificate(audit)

        # All fields must be present
        @test isdefined(cert, :timestamp)
        @test isdefined(cert, :symmetry_fidelity)
        @test isdefined(cert, :layer_homogeneity)
        @test isdefined(cert, :compression_efficiency)
        @test isdefined(cert, :generalization_readiness)
        @test isdefined(cert, :deployability_score)
        @test isdefined(cert, :predicted_failure_modes)
        @test isdefined(cert, :recommended_actions)
        @test isdefined(cert, :certificate_id)
    end

end
