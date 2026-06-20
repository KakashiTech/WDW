# WDW++ Fase 2: Geometría de Datos y Estructuras Motívicas

Este documento resume lo IMPLEMENTADO, cómo se prueba y cómo reproducir artefactos para la Fase 2′.

## Checklist (estado actual)
- [x] Álgebra de Quivers (near-ring: suma + composición no conmutativa)
- [x] Capa de Quiver `QuiverLayer` + agregación por caminos `apply_quiver_walks(depth)`
- [x] Activaciones multi-valor (proxy MoE) `mv_activation`
- [x] Correspondencias de Motivos (ciclos en X×Y) → matrices, composición y aplicación
- [x] Vector de rasgos motívicos (conteo de soluciones mod p en varios primos)
- [x] Q-G-ENN (capas/operadores equi-variantes a grupos discretos y lineales)
- [x] Integración motívica avanzada para reducción de dimensiones (SVD)

## Implementación
- **Quivers (near-ring)**
  - `src/Algebra/Quivers.jl`:
    - `near_ring_add`, `near_ring_mul` (composición no conmutativa)
    - `QuiverLayer`, `apply_quiver`, `apply_quiver_walks(depth)`
    - `mv_activation`, `relu`
- **Motivos (correspondencias y rasgos)**
  - `src/Motives/ComputableMotives.jl`:
    - `correspondence_matrix`, `apply_correspondence`, `compose_correspondences`
    - `motivic_features(A,b,primes)` (log de conteos F_p)
- **Q-G-ENN (equivarianza)**
  - `src/Quantum/QGroupENN.jl`: `dihedral_group`, `project_equivariant`, `is_equivariant`, `act`, `LinearGroup`, `SO2_linear_group`, `SO3_linear_group`.
  - Validado en matriz global (EQ1/EQ2) con proyección que anula error de equivarianza y certifica equivarianza exacta.
- **Reducción motívica (dimensionalidad)**
  - `src/Motives/MotivicReduce.jl`: `motivic_feature_matrix`, `motivic_dimreduce` (SVD; devuelve `Z,P`).
  - Tests específicos y garantía de ortonormalidad de columnas y no aumento de norma tras proyección.

## Pruebas (todas pasan)
- Ejecutado: `Pkg.test()` → 79/79 pruebas OK.
- Tests relevantes añadidos:
  - `extras/test/test_algebra_quivers_phase2.jl`: Capa de Quiver, agregación por caminos, activación multi-valor.
  - `extras/test/test_validation_matrix.jl`: EQ1/EQ2 (equivarianza y proyección) verificados cuantitativamente.
  - `extras/test/test_motivic_reduce.jl`: matriz de rasgos y reducción SVD con propiedades numéricas.

## Artefactos reproducibles (script)
- Script: `bench/phase2_quiver_motifs_artifacts.jl`
- Genera:
  - `bench/phase2_quiver_metrics.csv` (estabilidad espectral, norma de walks, columnas de activación multi-valor)
  - `bench/phase2_motifs_metrics.csv` (forma de la correspondencia, norma de salida, rasgos motívicos por primo)
  - `bench/phase2_certificate.txt` (resumen/certificado)
- Para reproducir:
  - `julia --project=. bench/phase2_quiver_motifs_artifacts.jl`

## Criterios de verificación cumplidos
- Near-ring operativo sobre mapas lineales (suma/composición) con capa de propagación y caminos.
- Proxy MoE multi-valor determinista y medible.
- Correspondencias como multi-mapas lineales y composición certificada.
- Rasgos motívicos reproducibles a partir de conteos mod p.

## Próximos pasos para cerrar Fase 2
- Implementar prototipo de **Q-G-ENN** (operadores equivariantes a grupos cuánticos matriciales compactos) bajo igualdad de cómputo.
- Diseñar pipeline de **reducción motívica** (selección de rasgos estable con invariantes) y artefactos comparativos con PCA.
- Publicar artefactos de Fase 2 en CI (añadir al workflow).
