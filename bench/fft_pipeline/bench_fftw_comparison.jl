#!/usr/bin/env julia
# Compare myfft (pure Julia) vs FFTW speed at various sizes
using WDW, LinearAlgebra, Printf

const FG = WDW.FFTGroup
const SIZES = [16, 32, 64, 128, 256, 512, 1024]

# Check if FFTW is available (not a direct dependency; may be loaded via Julia stdlib or separately)
fftw_available = try
    using FFTW
    true
catch
    false
end

println("="^60)
println("  FFT Performance: myfft (Julia) vs FFTW")
println("="^60)
if fftw_available
    @printf "  %-8s %-20s %-20s %-15s\n" "n" "myfft (μs)" "FFTW (μs)" "ratio"
    println("  " * "-"^63)
    for n in SIZES
        x = randn(n)
        t_my = @elapsed FG.myfft(x)
        t_fftw = @elapsed fft(x)
        @printf "  n=%-4d %18.3f %18.3f %13.2f×\n" n t_my*1e6 t_fftw*1e6 t_my/t_fftw
    end
else
    println("  FFTW not available in this environment; measuring myfft only")
    println()
    @printf "  %-8s %-20s\n" "n" "myfft (μs)"
    println("  " * "-"^29)
    for n in SIZES
        x = randn(n)
        t_my = @elapsed FG.myfft(x)
        @printf "  n=%-4d %18.3f\n" n t_my*1e6
    end
    println()
    println("  Note: Install FFTW via `using Pkg; Pkg.add(\"FFTW\")` for comparison.")
end
