# WDW Sistema Unificado

## Pipeline Integrado: Sheaves → Quivers → Q-G-ENN → MERA → Krylov

Este documento describe el sistema WDW unificado que conecta todas las fases en un pipeline operativo con flujo de datos real entre módulos.

---

## Arquitectura del Sistema Unificado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WDW UNIFIED PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Fase 1: Sheaves                                                             │
│  ┌─────────────┐    Secciones locales sobre abiertos topológicos            │
│  │ Input Data  │──→│ Partial{U, value} │                                    │
│  │  (n dims)   │    └─────────────────┘                                    │
│  └─────────────┘                                                           │
│         ↓                                                                    │
│  Fase 2: Quivers + Q-G-ENN                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐               │
│  │   Quiver    │──→│  Propagación  │──→│ Proyección       │               │
│  │   Graph     │    │  por caminos   │    │ Equivariante     │               │
│  │  (anillo)   │    │  (near-ring)   │    │  (grupo dihedral)│               │
│  └─────────────┘    └──────────────┘    └──────────────────┘               │
│         ↓                              ↓                                   │
│  Fase 3: MERA + Krylov                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐               │
│  │   MERA      │──→│  Haar Wavelet │──→│ Lanczos          │               │
│  │  Compress   │    │  Multiescala  │    │ Tridiagonal      │               │
│  │ (parametric)│    │  truncación    │    │ Complexity       │               │
│  └─────────────┘    └──────────────┘    └──────────────────┘               │
│         ↓                                                                    │
│  Output: Estado Unificado                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ UnifiedState: raw + sheaf_sections + quiver_features +              │   │
│  │                equivariant_output + compressed + complexity +         │   │
│  │                equivariance_error                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Ciclo Ruptura-Recuperación (Bucle Cerrado)

```
┌─────────────────────────────────────────────────────────────┐
│                    CLOSED LOOP                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Estado Base (Equivariante)                                │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────┐    ruptura (ruido diagonal)               │
│   │  Rupture    │──→ Estado Roto (no equivariante)          │
│   │  (break)    │    equiv_error ↑                          │
│   └─────────────┘                                           │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────┐    project_equivariant()                  │
│   │  Recovery   │──→ Estado Recuperado                      │
│   │  (heal)     │    equiv_error ↓ (≈0)                     │
│   └─────────────┘                                           │
│        │                                                     │
│        └────────────────┐                                    │
│                         │ (repetir ciclo)                  │
│   ┌─────────────────────┘                                    │
│   │                                                          │
│   ▼                                                          │
│   Métricas: recovery_ratio, S-score, complexity             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Uso

### Ejecutar Pipeline Unificado

```julia
using WDW
const U = WDW.UnifiedWDW

# 1. Crear pipeline
n = 32
pipeline = U.WDWPipeline(n, compression_levels=3, krylov_dim=10)

# 2. Datos de entrada
input_data = randn(n)

# 3. Ejecutar pipeline completo
state, T_mat, thetas = U.process(pipeline, input_data)

# 4. Ver resultados
println("Equivariance error: ", state.equivariance_error)
println("Complexity: ", state.complexity)
println("Compressed size: ", length(state.compressed))
```

### Ciclo Ruptura-Recuperación

```julia
# Inducir ruptura
ruptured = U.induce_rupture(state, noise_mag=1.0)

# Recuperar
recovered, eq_err = U.recover(pipeline, ruptured)

# Medir métricas
metrics, S_score = U.measure_invariants(pipeline, state, ruptured, recovered)
println("Recovery ratio: ", metrics.recovery_ratio)
println("S-score: ", S_score)
println("Success: ", metrics.success)
```

### Test Completo

```julia
# Ejecutar test completo con reporte
result = U.run_full_pipeline_test(32, noise_mag=1.0, seed=42)
```

---

## Benchmark Unificado

```bash
# Ejecutar benchmark completo
julia --project=. bench/unified_pipeline_benchmark.jl
```

Este benchmark genera:
- `bench/unified_pipeline_32.csv` - Métricas para n=32
- `bench/unified_pipeline_64.csv` - Métricas para n=64
- `bench/unified_certificate_32.txt` - Certificado de ruptura
- `bench/unified_certificate_64.txt` - Certificado de ruptura

### Métricas Reportadas

| Métrica | Descripción |
|---------|-------------|
| `equivariance_base` | Error de equivariancia estado base |
| `equivariance_ruptured` | Error tras ruptura (debe ser alto) |
| `equivariance_recovered` | Error tras recuperación (debe ser ≈0) |
| `recovery_ratio` | Factor de mejora (ruptured/recovered) |
| `complexity` | Métrica off-diagonal de Krylov |
| `S_score` | Composite score (SCI + inversos de residuos) |
| `success` | Boolean: ¿recovery exitoso? |

---

## Tests

```bash
# Ejecutar tests del sistema unificado
julia --project=. -q test/test_unified_wdw.jl

# O via Pkg.test()
julia --project=. -e 'using Pkg; Pkg.test()'
```

Tests incluidos:
- `Pipeline Construction` - Verifica creación de pipeline
- `Full Pipeline Execution` - Flujo completo de datos
- `Rupture and Recovery Cycle` - Ciclo ruptura-recuperación
- `Invariant Measurement` - Métricas compuestas
- `Full Pipeline Test Function` - Test completo integrado
- Componentes individuales (Sheaf, Quiver, Equivariant)

---

## Integración de Fases

### Fase 1 → Fase 2: Sheaves → Quivers

Los datos de entrada se convierten en secciones parciales sobre abiertos topológicos (`Partial{U, value}`). Estas secciones se usan como features iniciales de nodos en el quiver.

```julia
sections = U.cumulative_statistics(pipeline, data)
features = U.quiver_propagation(pipeline, sections)
```

### Fase 2: Quivers + Q-G-ENN

El quiver proporciona estructura de grafo (anillo). La proyección equivariante fuerza simetría bajo el grupo dihedral.

```julia
# Quiver en anillo con propagación por caminos
layer = QuiverLayer(quiver, in_dim, out_dim)
Y = apply_quiver_walks(layer, X, depth=2)

# Proyección equivariante
W_eq, eq_err = U.equivariant_projection(pipeline, Y)
```

### Fase 2 → Fase 3: MERA

La salida equivariante se comprime usando transformada Haar multiescala.

```julia
compressed, thetas, error = U.mera_compression(pipeline, equiv_features)
```

### Fase 3: MERA + Krylov

La representación comprimida se analiza vía Lanczos para obtener métrica de complejidad.

```julia
T_mat, complexity, alpha, beta = U.krylov_analysis(pipeline, compressed)
```

---

## S-Score Compuesto

El S-score mide coherencia estructural combinando:

```
S = (SCI + residual_score + equivariance_score) / 3

where:
- SCI = 1 - min(1, std(invariants) / mean(|invariants|))
- residual_score = 1 / (1 + equivariance_error)
- equivariance_score = 1 / (1 + equivariance_error)
```

Rango: [0, 1] (mayor es mejor coherencia estructural)

---

## Archivos del Sistema Unificado

| Archivo | Descripción |
|---------|-------------|
| `src/UnifiedWDW.jl` | Módulo principal de integración |
| `test/test_unified_wdw.jl` | Tests del sistema unificado |
| `bench/unified_pipeline_benchmark.jl` | Benchmarks completos |
| `bench/unified_pipeline_*.csv` | Resultados de benchmark |
| `bench/unified_certificate_*.txt` | Certificados de ruptura |

---

## Requisitos

- Julia 1.10+
- WDW.jl (módulos base)
- LinearAlgebra (stdlib)
- Random (stdlib)
- Printf (stdlib)
- DelimitedFiles (stdlib)

---

## Estructura del Estado Unificado

```julia
struct UnifiedState{T}
    raw_data::Vector{T}           # Datos originales
    sheaf_sections::Vector{Partial{Int,T}}  # Fase 1
    quiver_features::Matrix{T}    # Fase 2 (Quiver)
    equivariant_output::Matrix{T} # Fase 2 (Q-G-ENN)
    compressed::Vector{T}         # Fase 3 (MERA)
    complexity::T                 # Fase 3 (Krylov)
    equivariance_error::T         # Métrica invariante
end
```

---

## Próximos Pasos

1. **Integración Temporal**: Conectar ITE (Fase 4) para evolución temporal del estado unificado
2. **Bio-Cuántica**: Incorporar proxies de microtúbulos (Fase 5) como capa adicional
3. **Vacío/Gravedad**: Usar redes de espines LQG (Fase 6) para representación geométrica
4. **Optimización**: Entrenamiento end-to-end de todo el pipeline

---

## Verificación del Sistema Unificado

Para verificar que el sistema está correctamente unificado:

```bash
julia --project=. -e '
using WDW
const U = WDW.UnifiedWDW

# Test rápido
result = U.run_full_pipeline_test(16, noise_mag=1.0, seed=0)
println("\n✓ Sistema unificado operativo")
println("✓ Success: ", result["success"])
println("✓ S-score: ", result["S_score"])
'
```

---

**Estado**: Implementado y testeado. El sistema unificado demuestra que los 17 módulos de WDW pueden operar como pipeline integrado con flujo de datos real, ruptura-recuperación certificada, y métricas compuestas medibles.
