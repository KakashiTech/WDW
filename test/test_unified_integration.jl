using Test
using WDW
using Random

@testset "UnifiedIntegration" begin
    @testset "1. Smoke test: analyze_all with random data" begin
        rng = MersenneTwister(42)
        n_dim = 32
        n_samples = 10
        data = [randn(rng, n_dim) for _ in 1:n_samples]
        model_fn = x -> (copy(x), [0.5])

        result = WDW.UnifiedIntegration.analyze_all(data; model_fn=model_fn, seed=123)

        @test hasproperty(result, :measurement_matrix)
        @test hasproperty(result, :measurement_names)
        @test hasproperty(result, :n_success)
        @test hasproperty(result, :n_total)
        @test result.n_total == 28
        @test result.n_success >= 0
        @test result.n_success <= result.n_total
    end

    @testset "2. measurement_names has at least 10 entries" begin
        rng = MersenneTwister(42)
        data = [randn(rng, 32) for _ in 1:10]
        result = WDW.UnifiedIntegration.analyze_all(data; seed=123)

        @test length(result.measurement_names) >= 10
    end

    @testset "3. Deterministic: same seed -> same measurement_matrix" begin
        rng = MersenneTwister(42)
        data = [randn(rng, 32) for _ in 1:10]
        model_fn = x -> (copy(x), [0.5])

        r1 = WDW.UnifiedIntegration.analyze_all(data; model_fn=model_fn, seed=999)
        r2 = WDW.UnifiedIntegration.analyze_all(data; model_fn=model_fn, seed=999)

        @test r1.measurement_matrix ≈ r2.measurement_matrix
        @test r1.n_success == r2.n_success
    end

    @testset "4. Different seeds -> different results" begin
        rng = MersenneTwister(42)
        data = [randn(rng, 32) for _ in 1:10]
        model_fn = x -> (copy(x), [0.5])

        r1 = WDW.UnifiedIntegration.analyze_all(data; model_fn=model_fn, seed=111)
        r2 = WDW.UnifiedIntegration.analyze_all(data; model_fn=model_fn, seed=222)

        @test r1.measurement_matrix != r2.measurement_matrix || r1.n_success != r2.n_success
    end
end

@testset "StructuralEmbedding" begin
    @testset "5. structural_embedding with 3 fake results" begin
        names = ["m1", "m2", "m3"]
        meas_names = ["a", "b", "c", "d", "e"]

        results = [
            WDW.UnifiedIntegration.UnifiedResult(
                "", Dict(), WDW.UnifiedIntegration.AnalyzerResult[],
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 3, 3,
                reshape([1.0, 2.0, 3.0, 4.0, 5.0], :, 1),
                copy(meas_names)
            ),
            WDW.UnifiedIntegration.UnifiedResult(
                "", Dict(), WDW.UnifiedIntegration.AnalyzerResult[],
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 3, 3,
                reshape([5.0, 4.0, 3.0, 2.0, 1.0], :, 1),
                copy(meas_names)
            ),
            WDW.UnifiedIntegration.UnifiedResult(
                "", Dict(), WDW.UnifiedIntegration.AnalyzerResult[],
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 3, 3,
                reshape([1.5, 2.5, 3.5, 4.5, 5.5], :, 1),
                copy(meas_names)
            )
        ]

        emb = WDW.StructuralEmbedding.structural_embedding(results; model_names=names)
        @test size(emb.coords, 1) == 3
        @test emb.model_names == names
        @test emb.n_dims >= 1
    end
end
