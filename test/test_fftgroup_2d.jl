using Test
using WDW
using LinearAlgebra: norm
using Random: MersenneTwister

const FG = WDW.FFTGroup

function shift2d(x::Matrix, dx::Int, dy::Int)
    nx, ny = size(x)
    return [x[mod1(i-dx, nx), mod1(j-dy, ny)] for i in 1:nx, j in 1:ny]
end

@testset "myfft2d roundtrip" begin
    for n in [4, 8, 16]
        x = randn(n, n)
        @test FG.myifft2d(FG.myfft2d(x)) ≈ x atol=1e-14
    end
end

@testset "myfft2d separability" begin
    for n in [4, 8, 16]
        x = randn(n, n)
        direct = FG.myfft2d(x)
        temp = Matrix{Complex{Float64}}(undef, n, n)
        for i in 1:n
            temp[i, :] = FG.myfft(x[i, :])
        end
        sep = Matrix{Complex{Float64}}(undef, n, n)
        for j in 1:n
            sep[:, j] = FG.myfft(temp[:, j])
        end
        @test direct ≈ sep atol=1e-14
    end
end

@testset "myfft2d non-power-of-2" begin
    x = randn(6, 6)
    @test FG.myifft2d(FG.myfft2d(x)) ≈ x atol=5e-13
end

@testset "myfft2d linearity" begin
    for n in [4, 8]
        a, b = 2.0, -1.5
        x = randn(n, n)
        y = randn(n, n)
        lhs = FG.myfft2d(a * x + b * y)
        rhs = a * FG.myfft2d(x) + b * FG.myfft2d(y)
        @test norm(lhs - rhs) < 1e-12
    end
end

@testset "CyclicFourierLayer2D construction" begin
    for (nx, ny) in [(4, 4), (8, 8), (16, 16), (32, 32)]
        layer = FG.CyclicFourierLayer2D(nx, ny; seed=42)
        @test layer.nx == nx
        @test layer.ny == ny
        @test size(layer.A) == (nx, ny)
        @test length(layer.b) == nx * ny
    end
end

@testset "2D bispectrum features" begin
    for n in [4, 8]
        layer = FG.CyclicFourierLayer2D(n, n; seed=42)
        x = randn(n, n)
        feats = FG.combined_bispec_features_2d(x, layer)
        @test length(feats) == 3 * n * n
        @test eltype(feats) == Float64
    end
end

@testset "2D shift invariance" begin
    for n in [4, 8]
        layer = FG.CyclicFourierLayer2D(n, n; seed=42)
        x = randn(n, n)
        feats_orig = FG.combined_bispec_features_2d(x, layer)
        for (dx, dy) in [(1, 0), (0, 1), (1, 1), (3, 2), (n-1, n-1)]
            x_shift = shift2d(x, dx, dy)
            feats_shift = FG.combined_bispec_features_2d(x_shift, layer)
            @test norm(feats_orig - feats_shift) < 1e-11
        end
    end
end

@testset "2D accuracy" begin
    n = 8
    rng = MersenneTwister(42)
    layer = FG.CyclicFourierLayer2D(n, n; seed=42)
    n_feat = 3 * n * n
    Wc = randn(rng, 2, n_feat) * 0.1
    bc = zeros(2)
    xs = Matrix{Float64}[]
    ys = Int[]
    for i in 1:10
        x1 = [sin(2π * j / n) for i_ in 1:n, j in 1:n]
        push!(xs, x1); push!(ys, 1)
        x2 = [sin(2π * i_ / n) for i_ in 1:n, j in 1:n]
        push!(xs, x2); push!(ys, 2)
    end
    acc = FG.accuracy_bispec_2d(layer, Wc, bc, xs, ys)
    @test 0 ≤ acc ≤ 100
end

@testset "2D training" begin
    n = 8
    rng = MersenneTwister(42)
    layer = FG.CyclicFourierLayer2D(n, n; seed=42)
    n_feat = 3 * n * n
    Wc = randn(rng, 2, n_feat) * 0.1
    bc = zeros(2)
    xs = Matrix{Float64}[]
    ys = Int[]
    for i in 1:5
        x1 = [sin(2π * j / n) for i_ in 1:n, j in 1:n]
        push!(xs, x1); push!(ys, 1)
        x2 = [cos(2π * i_ / n) for i_ in 1:n, j in 1:n]
        push!(xs, x2); push!(ys, 2)
    end
    acc_before = FG.accuracy_bispec_2d(layer, Wc, bc, xs, ys)
    FG.train_bispec_2d!(layer, Wc, bc, xs, ys, 0.1)
    acc_after = FG.accuracy_bispec_2d(layer, Wc, bc, xs, ys)
    @test acc_after >= acc_before
end
