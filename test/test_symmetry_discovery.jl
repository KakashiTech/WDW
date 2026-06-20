using Test
using WDW
using LinearAlgebra: norm
using Statistics: mean

const SD = WDW.SymmetryDiscovery
const FP = WDW.FFTPipeline

@testset "SymmetryDiscovery" begin

    @testset "1. default_probes" begin
        probes = SD.default_probes(32)
        @test length(probes) == 8
        for (i, M) in enumerate(probes)
            @test size(M) == (32, 32)
            # All probes must be valid transformations
            # Check that M * ones(32) has finite norm
            @test isfinite(norm(M * ones(32)))
        end
        # First 5 probes are shift matrices (check cyclic property)
        M_shift = probes[1]
        for i in 1:32
            @test M_shift[i, mod1(i-1, 32)] == 1.0
        end
        # Probe 6 should be reflection matrix
        M_ref = probes[6]
        @test M_ref * M_ref ≈ Matrix{Float64}(I, 32, 32)
    end

    @testset "2. symmetry_profile: shift = exact symmetry" begin
        xs = [randn(32) for _ in 1:10]
        for x in xs; x ./= norm(x); end
        profile = SD.symmetry_profile(xs)
        # First 5 probes are shifts → divergence ~ 0
        for i in 1:5
            @test profile[i] < 1e-10
        end
    end

    @testset "3. symmetry_profile: reflection != symmetry" begin
        xs = [randn(32) for _ in 1:10]
        for x in xs; x ./= norm(x); end
        profile = SD.symmetry_profile(xs)
        # Probe 6 = reflection → divergence > 0
        @test profile[6] > 0.001
        # Reflection diverges more than shift
        @test profile[6] > profile[1] + 0.001
    end

    @testset "4. symmetry_profile: random transformation >> shift" begin
        xs = [randn(32) for _ in 1:5]
        for x in xs; x ./= norm(x); end
        profile = SD.symmetry_profile(xs)
        # Random (probe 7,8) should diverge more than shifts (probes 1-5)
        for i in 7:8
            for j in 1:5
                @test profile[i] > profile[j] + 0.01
            end
        end
    end

    @testset "5. symmetry_profile: profile unique per dataset" begin
        xs1 = [randn(32) for _ in 1:5]
        xs2 = [randn(32) .+ 2.0 for _ in 1:5]
        for x in xs1; x ./= norm(x); end
        for x in xs2; x ./= norm(x); end
        p1 = SD.symmetry_profile(xs1)
        p2 = SD.symmetry_profile(xs2)
        # Different datasets should have different profiles
        @test norm(p1 - p2) > 0.001
    end

    @testset "6. symmetry_profile: real C_n pair data" begin
        xs_tr, _, _, _ = FP.make_dataset(32, 2, 4, 42)
        profile = SD.symmetry_profile(xs_tr)
        # On time-reversal pair data:
        # Shifts are exact symmetries (bispectrum is shift-invariant)
        for i in 1:5
            @test profile[i] < 1e-10
        end
        # Reflection breaks symmetry (time-reversal pairs get confused)
        @test profile[6] > 0.1
    end

    @testset "7. profile_divergence" begin
        p_ref = [0.0, 0.0, 1.0, 1.0]
        p_same = [0.0, 0.0, 1.0, 1.0]
        p_diff = [1.0, 1.0, 0.0, 0.0]
        @test SD.profile_divergence(p_ref, p_same) < 1e-15
        @test SD.profile_divergence(p_ref, p_diff) > 0.5
    end

end
