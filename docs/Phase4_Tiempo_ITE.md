# WDW++ Fase 4: Dinámica Temporal Hiperdimensional

Este documento resume lo IMPLEMENTADO, cómo se prueba y cómo reproducir artefactos para la Fase 4′.

## Checklist (estado actual)
- [x] ITE (evolución en tiempo imaginario) con monotonía de energía
- [x] Métrica temporal hiperdimensional (36 tiempos, 3 espacios)
- [x] Evolución multi-eje (splitting ITE por eje) con monotonía de la suma de energías por sweep
- [x] Navegación Chronos–Kairos (intercalado secuencial/futuros)
- [x] Artefactos reproducibles (script)

## Implementación
- **ITE monoeje**
  - `src/Time/ITE.jl`: `step`, `evolve`, `monotone_energy`.
- **Multi-tiempo y métrica**
  - `src/Time/HyperTime.jl` (nuevo):
    - `simulate_metric(dt,dx)` computa `ds^2 = -∑dt_i^2 + ∑dx_j^2`.
    - `evolve_wavefunction(Hs,psi0,dts,steps)` aplica ITE por eje con normalización por subpaso.
    - `monotone_energies_axes` verifica monotonía de la suma de energías por sweep.
  - `src/Time/MultiTime.jl`: operador parabólico multi-eje (`simulate`, `is_stable`).
- **Chronos–Kairos**
  - `src/Planner/ChronosKairos.jl`: `interleave_roundrobin`, `schedule_ck`.

## Pruebas (todas pasan)
- Última suite: 106/106 pruebas OK (incluye Fases 1–6).
- Tests relevantes:
  - `extras/test/test_time_ite.jl`: ITE con energía monótona.
  - `extras/test/test_time_multitime.jl`: estabilidad de trayectoria multi-eje.
  - `extras/test/test_time_hypertime.jl`: métrica 36×3 y monotonía de suma de energías.
  - `extras/test/test_planner_phase4.jl`: intercalado Chronos–Kairos.

## Artefactos reproducibles (script)
- Script: `bench/phase4_time_artifacts.jl`
- Genera:
  - `bench/phase4_time_metrics.csv` (monotonías, ds², longitud del schedule)
  - `bench/phase4_time_certificate.txt` (certificado)
- Para reproducir:
  - `julia --project=. bench/phase4_time_artifacts.jl`

## Criterios de verificación cumplidos
- ITE monoeje y multi-eje con métricas de monotonía verificadas.
- Métrica hiperdimensional computada y registrada en artefactos.
- Scheduler Chronos–Kairos funcional con propiedades simples verificadas.

## Próximos pasos
- Scheduler multi-escala con políticas de priorización.
- Integración de ITE con módulos MERA/Krylov bajo igualdad de cómputo.
- Publicación de artefactos en CI.
