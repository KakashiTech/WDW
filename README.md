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

## Quick Start

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
| `bench/fft_pipeline/run_certified_benchmark.jl` | [⚠️ WIP] Timestamped certificate |
| `bench/fft_pipeline/run_oneshot.jl` | [⚠️ WIP] 1-shot classification |
| `bench/fft_pipeline/run_robustness.jl` | [⚠️ WIP] Multi-seed scaling |
| `bench/fft_pipeline/run_final_verdict.jl` | [⚠️ WIP] WDW vs MLP comparison |
| `bench/run_benchmark.jl` | Unified parameterized benchmark |

## Limitations (Honest)

1. **Time-reversal pair dataset** — the 100pp Cₙ≠Dₙ gap requires this structure. On general data (MNIST), power spectrum already classifies and the gap is 0pp.
2. **Feature engineering, not architecture** — an MLP on pre-computed bispectrum features matches WDW. The advantage is in the FFT + bispectrum construction, not the classifier.
3. **Pure-Julia FFT** — ~10× slower than FFTW for n > 1024. Chosen for Zygote gradient compatibility.
4. **Tested up to n=128** — theory scales arbitrarily, pipeline verified at these sizes.

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
