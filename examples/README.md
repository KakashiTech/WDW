# Examples

## 1. Equivariance Detection & Recovery (Canonical Demo)

Minimal reproducible demo for equivariance error measurement and projection recovery.

```bash
julia --project=. examples/equivariance_recovery.jl 12 0.10 1
```

Parameters: `n` (problem size), `noise` (rupture level), `seed`.

Measures: equivariance error before rupture, after rupture, and after projection recovery.

## 2. Fourier Bispectrum Pipeline (FFTGroup + FFTPipeline)

End-to-end demonstration of all 4 verified claims using WDW's Fourier bispectrum features:

```bash
# Full pipeline demo
julia --project=. bench/fft_pipeline/run_pipeline_completo.jl

# Certified benchmark (timestamped certificate)
julia --project=. bench/fft_pipeline/run_certified_benchmark.jl

# Individual experiments
julia --project=. bench/fft_pipeline/run_oneshot.jl
julia --project=. bench/fft_pipeline/run_robustness.jl
```

## 3. Programmatic (FFT Pipeline)

```julia
using WDW
const FP = WDW.FFTPipeline

# 1-shot, 4 classes, 32 dimensions
xs_tr, ys_tr, xs_te, ys_te = FP.make_dataset(32, 2, 1, 42)
p = FP.SignalPipeline(32; n_classes=4)
FP.train_pipeline!(p, xs_tr, ys_tr; epochs=500)

# Evaluate
cn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_te, ys_te; dn=false)
println("Cₙ accuracy: $(cn)%")
```

## 4. Theory Verification

```bash
julia --project=. bench/fft_pipeline/verify_bispec_theory.jl
```

## 5. Run All Tests

```bash
julia --project=. -e 'using WDW, Test; Pkg.test()'
```
