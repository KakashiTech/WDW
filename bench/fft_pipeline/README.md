# bench/fft_pipeline — WDW Fourier Bispectrum Experiments

## Certified Results

| Script | Description |
|--------|-------------|
| `run_pipeline_completo.jl` | End-to-end demo — all 4 results in 1 command |
| `run_certified_benchmark.jl` | Timestamped certificate (PASS/FAIL per result) |

**Certificate output**: `fft_certificate.txt`

## Individual Experiments

| Script | Description |
|--------|-------------|
| `run_oneshot.jl` | 1-shot shift-invariant classification (4 samples) |
| `run_robustness.jl` | Multi-seed robustness + n-scaling (n=16..128) |
| `run_final_verdict.jl` | WDW vs MLP comparison |
| `run_sample_efficiency.jl` | Accuracy vs training samples (1..128 shots) |
| `run_comparison.jl` | WDW vs CNN vs MLP comparison |
| `run_asym_tradeoff.jl` | A asymmetry vs Cₙ≠Dₙ gap tradeoff |
| `measure_mdl_real.jl` | Empirical MDL measurement |
| `verify_bispec_theory.jl` | Shift invariance + Dₙ sensitivity verification |
| `verify_scaling.jl` | Scaling verification (n=16..128) |

## Legacy (V2/V3/V4)

Older Cₙ/Dₙ experiments (`run_cndn_v2.jl`, `run_cndn_v3.jl`, `run_cndn_v4.jl`) —
retained for reference but superseded by the cleaned pipeline.

## Outputs

| File | Description |
|------|-------------|
| `fft_certificate.txt` | Certified benchmark output (timestamped) |
| `executive_summary.md` | Pipeline architecture + results |
| `verified_claims.md` | 4 verified results with full evidence |
| `table_baselines.tex` | LaTeX baseline comparison table |
| `table_scalability.tex` | LaTeX scalability table |

## Reproduce

```bash
# All results
julia --project run_pipeline_completo.jl

# Certificate
julia --project run_certified_benchmark.jl

# Any individual experiment
julia --project run_oneshot.jl
```
