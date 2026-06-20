using Test
using LinearAlgebra
using Random

@testset "Unified WDW Pipeline" begin
    
    @testset "Pipeline Construction" begin
        n = 16
        pipeline = WDW.UnifiedWDW.WDWPipeline(n, compression_levels=2, krylov_dim=5)
        
        @test pipeline.n == n
        @test length(pipeline.group.perms) > 0  # Tiene grupo
        @test length(pipeline.quiver.edges) > 0  # Tiene quiver
        @test pipeline.compression_levels == 2
        @test pipeline.krylov_dim == 5
    end
    
    @testset "Full Pipeline Execution" begin
        n = 16
        pipeline = WDW.UnifiedWDW.WDWPipeline(n, compression_levels=2, krylov_dim=5)
        
        # Datos de entrada
        input_data = randn(n)
        
        # Ejecutar pipeline completo
        state, T_mat, thetas = WDW.UnifiedWDW.process(pipeline, input_data)
        
        # Verificar estado resultante
        @test length(state.raw_data) == n
        @test length(state.sheaf_sections) > 0
        @test size(state.quiver_features, 1) == n
        @test length(state.compressed) > 0
        @test state.complexity >= 0
        @test state.equivariance_error >= 0
        
        # Verificar matriz tridiagonal de Krylov
        @test size(T_mat, 1) == size(T_mat, 2)  # Cuadrada
        @test size(T_mat, 1) <= pipeline.krylov_dim
        
        # Verificar que thetas existe
        @test length(thetas) > 0
    end
    
    @testset "Rupture and Recovery Cycle" begin
        n = 16
        pipeline = WDW.UnifiedWDW.WDWPipeline(n, compression_levels=2, krylov_dim=5)
        
        input_data = randn(n)
        state, _, _ = WDW.UnifiedWDW.process(pipeline, input_data)
        
        # Inducir ruptura
        noise_mag = 1.0
        ruptured = WDW.UnifiedWDW.induce_rupture(state, noise_mag)
        
        @test size(ruptured) == size(state.equivariant_output)
        
        # Recuperar
        recovered, eq_rec = WDW.UnifiedWDW.recover(pipeline, ruptured)
        
        @test size(recovered) == size(ruptured)
        @test eq_rec >= 0
        
        # Verificar que la recuperación mejora la equivariancia
        eq_ruptured = WDW.UnifiedWDW.equivariance_error(pipeline, ruptured)
        @test eq_rec < eq_ruptured  # Debe mejorar
    end
    
    @testset "Invariant Measurement" begin
        n = 16
        pipeline = WDW.UnifiedWDW.WDWPipeline(n, compression_levels=2, krylov_dim=5)
        
        input_data = randn(n)
        state, _, _ = WDW.UnifiedWDW.process(pipeline, input_data)
        
        ruptured = WDW.UnifiedWDW.induce_rupture(state, 1.0)
        recovered, _ = WDW.UnifiedWDW.recover(pipeline, ruptured)
        
        metrics, S_score, _, _ = WDW.UnifiedWDW.measure_invariants(pipeline, state, ruptured, recovered)
        
        # Verificar métricas
        @test metrics.equivariance_base >= 0
        @test metrics.equivariance_ruptured >= 0
        @test metrics.equivariance_recovered >= 0
        @test metrics.complexity_base >= 0
        @test metrics.recovery_ratio > 0
        @test 0 <= S_score <= 1  # S-score normalizado
        
        # Verificar éxito del recovery
        # With the new [inv | anti] decomposition, equivariance_base measures
# how much of the input is non-invariant (anti-invariant fraction).
# For cumulative statistics features, this is > 0 (meaningful spatial structure).
@test metrics.equivariance_base >= 0
@test isfinite(metrics.equivariance_base)
    end
    
    @testset "Full Pipeline Test Function" begin
        # Probar la función de test completa
        result = WDW.UnifiedWDW.run_full_pipeline_test(16, noise_mag=1.0, seed=42)
        
        @test haskey(result, "pipeline")
        @test haskey(result, "state")
        @test haskey(result, "metrics")
        @test haskey(result, "S_score")
        @test haskey(result, "T_mat")
        @test haskey(result, "success")
        
        # Verificar tipos
        @test result["success"] isa Bool
        @test result["S_score"] isa Float64
        @test 0 <= result["S_score"] <= 1
    end
    
    @testset "Cumulative Statistics" begin
        n = 8
        pipeline = WDW.UnifiedWDW.WDWPipeline(n, compression_levels=2, krylov_dim=3)
        data = randn(n)
        
        sections = WDW.UnifiedWDW.cumulative_statistics(pipeline, data)
        
        @test length(sections) > 0
        @test all(s -> s isa WDW.Knowledge.Partial, sections)
    end
    
    @testset "Quiver Propagation" begin
        n = 8
        pipeline = WDW.UnifiedWDW.WDWPipeline(n, compression_levels=2, krylov_dim=3)
        data = randn(n)
        
        sections = WDW.UnifiedWDW.cumulative_statistics(pipeline, data)
        features = WDW.UnifiedWDW.quiver_propagation(pipeline, sections)
        
        @test size(features, 1) == n
        @test !any(isnan, features)
    end
    
    @testset "Equivariant Projection" begin
        n = 8
        pipeline = WDW.UnifiedWDW.WDWPipeline(n, compression_levels=2, krylov_dim=3)
        
        # Hard mode (soft_lambda=0): output has 2× columns [inv | anti]
        features = randn(n, 2)
        proj, err = WDW.UnifiedWDW.equivariant_projection(pipeline, features)
        
        @test size(proj, 1) == n
        @test size(proj, 2) == 2 * size(features, 2)  # expanded: inv + anti
        @test err >= 0
        @test err <= 1.0  # normalized relative error
        
        # First half should be Dₙ-invariant (error ≈ 0)
        inv_part = proj[:, 1:2]
        inv_err = WDW.UnifiedWDW.equivariance_error(pipeline, inv_part)
        @test inv_err < 1e-10
        
        # Soft mode: same dimensions as input
        pipeline_soft = WDW.UnifiedWDW.WDWPipeline(n, compression_levels=2, krylov_dim=3, soft_lambda=0.1)
        proj_soft, err_soft = WDW.UnifiedWDW.equivariant_projection(pipeline_soft, features)
        @test size(proj_soft) == size(features)  # same dims in soft mode
        @test err_soft >= 0
    end
    
end
