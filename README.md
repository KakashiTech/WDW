# WDW.jl — Algebraic Neural Networks with Provable Symmetry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.10-9558B2)](https://julialang.org/)
[![CI](https://github.com/KakashiTech/WDW/actions/workflows/CI.yml/badge.svg)](https://github.com/KakashiTech/WDW/actions)

**Fourier bispectrum features are provably shift-invariant.** No data augmentation. No learned approximation. The phase cancels algebraically.

```bash
# Quick start (requires Julia 1.10+)
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'        # 316/316 pass
julia --project bench/fft_pipeline/run_pipeline_completo.jl  # all 4 verified results
```

---

## The Four Verified Results

| # | Result | Evidence |
|---|--------|----------|
| 1 | **Shift-invariant classification: 100%** (4 samples, 0 aug) | `‖B(shifted) - B(orig)‖ = 2e-15` |
| 2 | **Cₙ ≠ Dₙ gap: 100pp** (inherent, not a trick) | Bispectrum × time-reversal structure |
| 3 | **Recovery: MSE 7e-34** (float64 floor) | Same spectral weights A_ω do it all |
| 4 | **MLP: 25% vs WDW: 100%** (same data, same budget) | MLP ~10× params, 4× epochs → random |

These results are **mathematical identities**, not engineering feats. The bispectrum phase triple product cancels by construction — no training required, no data augmentation needed. Verification is deterministic (run once, get the same numbers every time).

---

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

---

## Module Architecture (3 Tiers)

The project is organized into three tiers. Tier 1 is production-ready with verified results. Tiers 2-3 are active research with varying maturity.

### ● Tier 1 — WDW Core (Verified, Documented, Tested)

The core contribution: Fourier bispectrum features with algebraic invariance guarantees.

| Module | Description | Status |
|--------|-------------|--------|
| `FFTGroup.jl` | Pure-Julia FFT, `CyclicFourierLayer`, bispectrum features, exact recovery, Cₙ≠Dₙ gap detection, optional FFTW backend | **Verified** (316 tests, 4 breakthroughs) |
| `FFTPipeline.jl` | `SignalPipeline` — end-to-end classification: spectral weights + linear classifier, gradient-trained via Zygote | **Verified** |
| `ScalableWDW.jl` | Optimizations for n ≥ 1000: block processing, streaming FFT | **Verified** (n=1024 confirmed) |

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

### ● Tier 2 — Research Extensions (Working, Evolving)

Automated symmetry discovery and group-equivariant architectures built on Tier 1.

| Module | Description | Status |
|--------|-------------|--------|
| `SymmetryDiscovery.jl` | 8 symmetry probes (shift, reflect, random, etc.) + profile comparison | **Tested** |
| `SymmetryCertificate.jl` | 7-pillar model audit: dataset bias, equivariance, deployability | **Tested** |
| `AutoSymmetryFlux.jl` | Latent LieGAN, LieSD, SymmetryGAN in Flux.jl | **Tested** |
| `AutoSymmetryDiscovery.jl` | Closed-loop symmetry discovery, structure transfer, meta-learning | **Tested** |
| `Quantum/QGroupENN.jl` | Group equivariant neural networks: Cₙ, Dₙ, SO(2), SO(3), `project_equivariant` | **Tested** |
| `Tensor/HolographicCodes.jl` | Haar-wavelet MERA compression, learnable rotations | **Tested** |

### ● Tier 3 — Experimental / Foundational (Pre-Release)

Mathematical frameworks exploring connections between sheaf theory, quiver algebra, representation theory, and learning. **These are not yet validated on benchmark tasks.**

| Module | Description | Status |
|--------|-------------|--------|
| `UnifiedWDW.jl` | Sheaf → Quiver → Q-G-ENN → MERA → Krylov unified pipeline state machine | **Experimental** |
| `RuptureABC.jl` | A/B/C rupture certification: MDL irreducibility, new-class performance, OOD coherence | **Experimental** |
| `UnifiedIntegration.jl` | Cross-module analyzer framework | **Experimental** |
| `Algebra/Quivers.jl` | Quiver representation theory, `QuiverLayer`, spectral stability | **Experimental** |
| `Krylov/Complexity.jl` | Lanczos tridiagonalization, Krylov spread complexity | **Experimental** |
| `Sheaves/FiniteSheaves.jl` | Constant sheaf, gluing, partial sections | **Experimental** |
| `Knowledge/TopologicalFunctors.jl` | Topological spaces, Heyting algebra, naming functors | **Experimental** |
| `Logic/DSL.jl`, `Semantics/Kripke.jl` | Categorical logic, Kripke semantics | **Experimental** |
| `Category/Sets.jl` | Finite sets, function maps, pullbacks | **Experimental** |
| `Motives/*.jl` | Motivic features, Betti numbers, dimension reduction | **Experimental** |
| `Time/*.jl` | Hyper-time evolution, imaginary-time evolution, multi-agent time | **Experimental** |
| `Bio/Microtubules.jl` | Lattice quDit gates, DNLS, Penrose collapse | **Experimental** |
| `Gravity/LQGDataSpace.jl` | Spin networks, area information | **Experimental** |
| `Vacuum/QET.jl` | Quantum energy teleportation analogs | **Experimental** |
| `Planner/ChronosKairos.jl` | Scheduling algorithms | **Experimental** |

---

## Mathematical Properties & Considerations

### 1. Cₙ ≠ Dₙ gap requires time-reversal structure
The 100pp gap between cyclic (Cₙ) and dihedral (Dₙ) accuracy is a **real group-theoretic result**: the Fourier bispectrum cannot distinguish a signal from its time-reversal because the phase triple product cancels identically under both shifts and reflections. This is mathematically guaranteed for any dataset with time-reversal symmetry (pairs of samples related by reversal). On unstructured data (e.g., random MNIST crops) the gap is 0pp — the theory predicts this. The gap is not a performance claim; it is a **symmetry detection test** that confirms the bispectrum respects Cₙ but is blind to reflection.

### 2. Representation, not architecture
WDW provides **algebraically guaranteed shift-invariant features** as a differentiable end-to-end pipeline. Any downstream classifier (MLP, SVM, KNN) on these features achieves the same accuracy — because the invariance is in the **mathematical representation**, not the learned layers. This is by design, not a weakness: the value is invariance without data augmentation, without learned approximation, with provable guarantees, and with the ability to backpropagate through the feature construction to optimize spectral weights (`A_ω`) for the task.

### 3. FFT backend
The default pure-Julia FFT (`myfft`) is ~10× slower than FFTW for n > 1024. WDW includes an **optional FFTW backend**: set `WDW.FFTGroup.use_fftw[] = true` to switch to FFTW (10-100× faster) while maintaining full Zygote differentiability via custom adjoints. The pure-Julia fallback remains the default (no external dependency).

### 4. Verified scales
One-dimensional signals: verified up to n = 1024 (sub-linear O(n log n) timing confirmed). Theory scales arbitrarily; verification at larger sizes is a matter of compute resources, not mathematical limitation.

### 5. External validation
The core benchmarks currently use **synthetic signals** with controlled time-reversal structure. Validation on real-world datasets (PhysioNet ECG, UCR time series) is the next priority. The UCR benchmark runner (`bench/ucr_benchmark.jl`) is a first step toward this.

---

## Benchmarks

| Script | Description | Tier |
|--------|-------------|------|
| `bench/fft_pipeline/run_pipeline_completo.jl` | All 4 verified results | Core |
| `bench/real_timeseries_cndn_gap.jl` | Cₙ≠Dₙ gap on ECG-like heartbeats | Core |
| `bench/wdw_vs_mlp_features.jl` | WDW vs MLP+features under spectral noise | Core |
| `bench/fft_pipeline/run_oneshot.jl` | 1-shot classification | Core |
| `bench/fft_pipeline/run_robustness.jl` | Multi-seed scaling | Core |
| `bench/fft_pipeline/run_final_verdict.jl` | WDW vs MLP (raw signals) | Core |
| `bench/fft_pipeline/bench_fftw_comparison.jl` | FFTW vs pure-Julia speed comparison | Core |
| `bench/ucr_benchmark.jl` | ECG/Sensor/EEG time series benchmark | Core |
| `bench/unified_pipeline_benchmark.jl` | Sheaf → Quiver → MERA → Krylov pipeline | Experimental |

---

## Installation

```bash
git clone https://github.com/KakashiTech/WDW
cd WDW
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run all tests
julia --project -e 'using Pkg; Pkg.test()'
```

---

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
