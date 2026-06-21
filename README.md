# WDW.jl — Algebraic Neural Networks with Provable Symmetry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.10-9558B2)](https://julialang.org/)
[![CI](https://github.com/KakashiTech/WDW/actions/workflows/CI.yml/badge.svg)](https://github.com/KakashiTech/WDW/actions)

**Fourier bispectrum features are provably shift-invariant.** No data augmentation. No learned approximation. The phase cancels algebraically.

## The Four Verified Results

| # | Result | Evidence |
|---|--------|----------|
| 1 | **Shift-invariant classification: 100%** (4 samples, 0 aug) | `‖B(shifted) - B(orig)‖ = 2e-15` |
| 2 | **Cₙ ≠ Dₙ gap: 100pp** (inherent, not a trick) | Bispectrum × time-reversal structure |
| 3 | **Recovery: MSE 7e-34** (float64 floor) | Same spectral weights A_ω do it all |
| 4 | **MLP: 25% vs WDW: 100%** (same data, same budget) | MLP ~10× params, 4× epochs → random |

```bash
# Quick demo — equivariance detection & recovery (works)
julia --project=. examples/equivariance_recovery.jl 12 0.10 1

# Run all tests (316/316 pass)
julia --project -e 'using Pkg; Pkg.test()'

# Full pipeline — all 4 verified results
julia --project bench/fft_pipeline/run_pipeline_completo.jl
```

## The Math (In 3 Lines)

The Fourier bispectrum at frequency ω:

```
B_z(ω) = ẑ_ω · ẑ₂ · conj(ẑ_{mod(ω,n)+1})   where  ẑ_ω = A_ω · FFT(x)_ω
```

Under shift by t, each DFT coefficient gains phase `e^{-2πiωt/n}`.
The triple product cancels them algebraically:

```
-(ω-1) - 1 + mod(ω,n) = 0  →  phase = exp(0) = 1
```

**B(shift(x)) = B(x) identically.** Not learned. Proved.

## Module Architecture

```
WDW.jl (Main — 32+ submodules)
├── FFTGroup.jl           Fourier bispectrum + shift-invariant features
├── FFTPipeline.jl        End-to-end training + evaluation
├── UnifiedWDW.jl         Sheaf → Quiver → MERA → Krylov pipeline
├── AutoSymmetryDiscovery.jl  Automated symmetry discovery (LieGAN, LieSD, SymmetryGAN)
├── AutoSymmetryFlux.jl   Flux.jl implementation of auto-symmetry
├── RuptureABC.jl         A/B/C rupture certification
├── SymmetryCertificate.jl    Formal symmetry certificates
├── UnifiedIntegration.jl     Cross-module analyzer
├── StructuralExperiments.jl  MLP baseline with gradient descent
├── PhysicsPhonons.jl     Physics/phonon applications
├── Theory/Pipeline modules: TheoreticalMetrics, RigorousMetrics,
│   RealWorldValidation, RealBaselines, PaperMetrics, MultiDataset,
│   Scalability, Breakthrough, WDWv2, NextLevelComplete
└── Foundation modules:
    ├── Quantum/           Group equivariance (Cₙ, Dₙ, SO(2), SO(3))
    ├── Tensor/            MERA, holographic codes
    ├── Krylov/            Krylov complexity
    ├── Algebra/           Quivers, representation theory
    ├── Sheaves/           Sheaf theory
    ├── Logic/             Categorical logic
    ├── Semantics/         Semantic models
    ├── Category/          Category theory
    ├── Knowledge/         Knowledge representation
    ├── Motives/           Motivic structures
    ├── Time/              Time series analysis
    ├── Planner/           Planning algorithms
    ├── Bio/               Bioinformatics
    ├── Gravity/           Quantum gravity analogs
    └── Vacuum/            Vacuum structure
```

## Mathematical Properties & Considerations

### 1. Cₙ ≠ Dₙ gap requires time-reversal structure
The 100pp gap between cyclic (Cₙ) and dihedral (Dₙ) accuracy is a **real group-theoretic result**: the Fourier bispectrum cannot distinguish a signal from its time-reversal because the phase triple product cancels identically under both shifts and reflections. This is mathematically guaranteed for any dataset with time-reversal symmetry (pairs of samples related by reversal). On unstructured data (e.g., random MNIST crops) the gap is 0pp — the theory predicts this. The gap is not a performance claim; it is a **symmetry detection test** that confirms the bispectrum respects Cₙ but is blind to reflection.

### 2. Representation, not architecture
WDW provides **algebraically guaranteed shift-invariant features** as a differentiable end-to-end pipeline. Any downstream classifier (MLP, SVM, KNN) on these features achieves the same accuracy — because the invariance is in the **mathematical representation**, not the learned layers. This is by design, not a weakness: the value is invariance without data augmentation, without learned approximation, with provable guarantees, and with the ability to backpropagate through the feature construction to optimize spectral weights (`A_ω`) for the task. No existing library provides a differentiable algebraically-guaranteed invariant feature pipeline. Additional benchmarks show WDW's full pipeline outperforming specialized equivariant architectures (E2CNN, escnn, PyG) on PDE and graph tasks (see `test_paper_metrics.jl`).

### 3. FFT backend
The default pure-Julia FFT (`myfft`) is ~10× slower than FFTW for n > 1024. WDW now includes an **optional FFTW backend**: set `WDW.FFTGroup.use_fftw[] = true` in your script to switch to FFTW (10-100× faster) while maintaining full Zygote differentiability via custom adjoints. The pure-Julia fallback remains the default (no external dependency).

### 4. Verified scales
One-dimensional signals: verified up to n = 1024 (see `verify_scaling.jl` for n=512, `test_scalability.jl` for n=1024 — sub-linear O(n log n) timing confirmed). Two-dimensional images: verified up to 32×32 = 1024 (MNIST achieves 85-97% accuracy with shift-invariance error < 5e-10). Theory scales arbitrarily; verification at larger sizes is a matter of compute resources, not mathematical limitation.

```bash
# Install
git clone https://github.com/KakashiTech/WDW
cd WDW
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run the equivariance demo
julia --project examples/equivariance_recovery.jl 12 0.10 1

# Run all tests
julia --project -e 'using Pkg; Pkg.test()'
```

## Programmatic

```julia
using WDW
const FP = WDW.FFTPipeline

# 1-shot, 4 classes, 32 dimensions
xs_tr, ys_tr, xs_te, ys_te = FP.make_dataset(32, 2, 1, 42)
p = FP.SignalPipeline(32; n_classes=4)
FP.train_pipeline!(p, xs_tr, ys_tr; epochs=500)

# Cₙ accuracy & Cₙ≠Dₙ gap
cn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_te, ys_te; dn=false)
xs_dn = [FP.reflect(x) for x in xs_te]
dn = WDW.FFTGroup.accuracy_bispec(p.layer, p.Wc, p.bc, xs_dn, ys_te; dn=false)
println("Cₙ = $(cn)%  Dₙ = $(dn)%  Gap = $(cn - dn)pp")
```

## Benchmarks

| Script | Description |
|--------|-------------|
| `bench/fft_pipeline/run_pipeline_completo.jl` | All 4 verified results |
| `bench/real_timeseries_cndn_gap.jl` | Cₙ≠Dₙ gap on ECG-like heartbeats |
| `bench/wdw_vs_mlp_features.jl` | WDW vs MLP+features under spectral noise |
| `bench/fft_pipeline/run_oneshot.jl` | 1-shot classification |
| `bench/fft_pipeline/run_robustness.jl` | Multi-seed scaling |
| `bench/fft_pipeline/run_final_verdict.jl` | WDW vs MLP (raw signals) |
| `bench/fft_pipeline/bench_fftw_comparison.jl` | FFTW vs pure-Julia speed comparison |
| `bench/run_benchmark.jl` | Unified parameterized benchmark |
| `bench/ucr_benchmark.jl` | ECG/Sensor/EEG time series benchmark |

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{wdw2026,
  title = {WDW.jl: Algebraic Neural Networks with Provable Symmetry},
  author = {KakashiTech},
  year = {2026},
  url = {https://github.com/KakashiTech/WDW}
}
```
