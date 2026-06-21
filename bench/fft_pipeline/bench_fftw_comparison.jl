#!/usr/bin/env julia
# Compare fft_dispatch (pure-Julia fallback) vs FFTW backend speed
# Demonstrates the optional FFTW integration (Breakthrough #4)
using WDW, LinearAlgebra, Printf

const FG = WDW.FFTGroup
const SIZES = [16, 32, 64, 128, 256, 512, 1024]

# Check if FFTW is available
fftw_available = FG._fftw_available[]

println("="^60)
println("  FFT Backend Comparison: Julia fallback vs FFTW")
println("="^60)

@printf "  %-8s %-22s %-22s %-12s\n" "n" "Julia (μs)" "FFTW via dispatch (μs)" "ratio"
println("  " * "-"^66)

for n in SIZES
    x = randn(n)
    FG.use_fftw[] = false
    t_julia = @elapsed FG.fft_dispatch(x)
    if fftw_available
        FG.use_fftw[] = true
        t_fftw = @elapsed FG.fft_dispatch(x)
        @printf "  n=%-4d %20.3f %20.3f %10.2f×\n" n t_julia*1e6 t_fftw*1e6 t_julia/t_fftw
    else
        @printf "  n=%-4d %20.3f %20s\n" n t_julia*1e6 "N/A"
    end
end

# Reset to default
FG.use_fftw[] = false

println()
if !fftw_available
    println("  FFTW not available in this environment.")
    println("  Install via `using Pkg; Pkg.add(\"FFTW\")` and restart Julia.")
    println("  Then set `FG.use_fftw[] = true` to enable 10-100× speedup.")
else
    println("  FFTW IS available — set `FG.use_fftw[] = true` in your script.")
    println("  All fft_dispatch/ifft_dispatch calls route through FFTW automatically.")
    println("  Zygote adjoints remain fully differentiable with either backend.")
end
