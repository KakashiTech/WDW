# WDW++ Fase 3: Memoria Holográfica y Complejidad de Krylov

Este documento resume lo IMPLEMENTADO, cómo se prueba y cómo reproducir artefactos para la Fase 3′.

## Checklist (estado actual)
- [x] Lanczos/Krylov: `lanczos_tridiagonal`, `krylov_spread_complexity` (spread complexity estándar)
- [x] MERA (Haar y parametrizada): `mera_reconstruct_truncated`, `multiscale_error`, `optimize_thetas`
- [x] Tests de Fase 3: error multiescala decreciente, equivalencia param=0, mejora por optimización; ver `extras/test/test_tensor_mera_phase3.jl` y `extras/test/test_krylov.jl`
- [x] Artefactos reproducibles: `bench/phase3_krylov_mera_artifacts.jl` → CSVs y certificado

## Implementación
- **Krylov (Lanczos)**
  - `src/Krylov/Complexity.jl`: `lanczos_tridiagonal(H,v0,m)`, `krylov_spread_complexity(T)`.
- **MERA/Haar**
  - `src/Tensor/HolographicCodes.jl`: `mera_reconstruct_truncated`, `multiscale_error`.
  - Versión parametrizada y optimización: `param_*` y `optimize_thetas`.

## Pruebas (todas pasan)
- Ejecutado: `Pkg.test()` → 89/89 pruebas OK (incluye Fases 1–3).
- Tests relevantes:
  - `extras/test/test_krylov.jl`: dimensiones del tridiagonal de Lanczos y complejidad ≥ 0.
  - `extras/test/test_tensor_mera_phase3.jl`: monotonicidad de error por `keep_levels`, equivalencia param=0 con no-param, y `optimize_thetas` no empeora y típicamente mejora.

## Artefactos reproducibles (script)
- Script: `bench/phase3_krylov_mera_artifacts.jl`
- Genera:
  - `bench/phase3_krylov_metrics.csv` (m vs complejidad off-diagonal)
  - `bench/phase3_mera_metrics.csv` (error por `keep_levels`, base vs opt)
  - `bench/phase3_krylov_mera_certificate.txt` (certificado resumido)
- Para reproducir:
  - `julia --project=. bench/phase3_krylov_mera_artifacts.jl`

## Criterios de verificación cumplidos
- Crecimiento de complejidad de Krylov monitoreado vía tridiagonalización.
- MERA (Haar/param) con error multiescala decreciente y mejoras por optimización verificadas.

## Próximos pasos
- Predictor de coeficientes de Lanczos (Transformer) para evitar saturación.
- Métricas adicionales de complejidad y comparación bajo igualdad de cómputo.
- Integración con quiver/MoE y sheaves en un bucle cerrado (CI con artefactos).
