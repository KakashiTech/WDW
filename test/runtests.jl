using Test
using WDW

if haskey(ENV, "WDW_FULL_TESTS")
    @info "Running full test suite..."
    for f in sort(readdir(joinpath(@__DIR__, "..", "extras", "test")))
        startswith(f, "test_") && endswith(f, ".jl") && include(joinpath(@__DIR__, "..", "extras", "test", f))
    end
else
    @testset "WDW (DEMO)" begin
        @testset "demo_equivariance_recovery" begin include("test_demo_equivariance_recovery.jl") end
        @testset "unified_pipeline" begin include("test_unified_wdw.jl") end
        @testset "rupture_ABC" begin include("test_rupture_ABC.jl") end
        @testset "fftgroup_pipeline" begin include("test_fftgroup_pipeline.jl") end
        @testset "fftgroup_2d" begin include("test_fftgroup_2d.jl") end
        @testset "symmetry_discovery" begin include("test_symmetry_discovery.jl") end
        @testset "symmetry_certificate" begin include("test_symmetry_certificate.jl") end
        @testset "unified_integration" begin include("test_unified_integration.jl") end
        @testset "auto_symmetry_discovery" begin include("test_auto_symmetry_discovery.jl") end
        @testset "autosymmetry_flux" begin include("test_autosymmetry_flux.jl") end
        @testset "next_level_complete" begin include("test_next_level_complete.jl") end
    end
end

# Demo/benchmark scripts (no @test blocks) available in test/:
#   test_scalability.jl, test_real_baselines.jl, test_real_world.jl,
#   test_paper_metrics.jl, test_wdw_v2_demo.jl, test_wdw_v2_scientific.jl,
#   test_rigorous_metrics.jl, test_theoretical_metrics.jl, test_multi_dataset.jl,
#   test_physics_phonons.jl, test_breakthrough.jl

# 2D extension and UnifiedIntegration tests (created by fix agent)
# included above in DEMO testset
