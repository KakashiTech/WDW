# WDW++ Fase 1: Cimentación Lógica y Ontológica (Topos)

Este documento resume lo IMPLEMENTADO, cómo se prueba y cómo reproducir artefactos para la Fase 1.

## Checklist (estado actual)
- [x] Lenguaje Interno (mínimo) + Semántica intuicionista
- [x] Clasificador de subobjetos Ω (Sets) + mapa característico y pullback
- [x] Conocimiento Parcial `k_i` (proxy) + integración con Haces
- [x] Functor de Nombramiento `N_i` (NamingFunctor con `build_naming_functor` y `restrict_name`)
- [x] Arquitectura de Haces (gluing local→global) con pruebas

## Implementación
- **Lenguaje interno + Kripke (intuicionista)**
  - `src/Logic/DSL.jl`: AST con `Top, Bot, Var, And, Or, Imply, Not` y operadores.
  - `src/Semantics/Kripke.jl`: modelo de Kripke, chequeo de monotonicidad y relación de forzamiento `forces` acorde a lógica de Heyting.
- **Ω en Sets (subobject classifier)**
  - `src/Category/Sets.jl`: `omega()`, `terminal()`, `true_map()`, `characteristic(X, subset)`, `pullback(f,g)`.
  - `extras/test/test_sets_omega.jl`: verifica mapa característico y pullback (talla del subobjeto).
- **Heyting en abiertos (Ω interna del topos de sheaves sobre TopSpace)**
  - `src/Knowledge/TopologicalFunctors.jl`: `TopSpace`, `is_open`, `int`, `cl`.
  - Nuevo: `HeytingOpen`, `heyting_top/bot/and/or/imply/not/leq` sobre abiertos.
- **Conocimiento Parcial `k_i` (proxy) + integración con Haces**
  - `src/Knowledge/TopologicalFunctors.jl`: `Partial`, `restrict`, `compatible`, `glue_partial`.
  - `src/Sheaves/FiniteSheaves.jl`: `sections_to_partials`, `glue_via_partials` (usa `glue_partial`).
- **Functor de Nombramiento `N_i`**
  - `src/Knowledge/TopologicalFunctors.jl`: `Name`, `NamingFunctor`, `build_naming_functor`, `restrict_name` (consistencia bajo restricción e intersecciones).
  - `extras/test/test_knowledge_naming.jl`: herencia de nombres en intersecciones coherentes y distinción cuando hay conflicto.
- **Haces**
  - `src/Sheaves/FiniteSheaves.jl`: `ConstantSheaf`, `glue` (consistencia local-global), más integración anterior.

## Pruebas (todas pasan)
- Ejecutado: `Pkg.test()` → 72/72 pruebas OK.
- Tests relevantes añadidos:
  - `extras/test/test_knowledge.jl`: leyes de Heyting básicas y glue vía conocimiento parcial consistente/inconsistente.
  - `extras/test/test_sheaves.jl`, `extras/test/test_sets_omega.jl`: cobertura de glue y Ω.
  - `extras/test/test_knowledge_naming.jl`: functor de nombramiento.

## Artefactos reproducibles (script)
- Script: `bench/phase1_heyting_artifacts.jl`
- Genera:
  - `bench/phase1_heyting_truths.csv` (propiedades: unidad ∧, implicación reflexiva, ⊥ ≤ U)
  - `bench/phase1_sets_omega.csv` (talla de pullback y bits de χ)
  - `bench/phase1_sheaves_partials.csv` (casos consistente/inconsistente de pegado por parciales)
  - `bench/phase1_heyting_certificate.txt` (resumen/certificado)
- Para reproducir:
  - `julia --project=. bench/phase1_heyting_artifacts.jl`

## Criterios de verificación cumplidos
- Lógica intuicionista operativa (Kripke) y lattice de Heyting en abiertos.
- Ω de Sets con característica y pullback certificado en tests/artefactos.
- Conocimiento parcial como proxy de `k_i`, integrable con Haces sin duplicación.
- Haces con pegado local→global y coherencia validada.

## Próximos pasos (Fase 1 → cierre)
- Formalizar `N_i` (nombramiento) como funtor tipado sobre objetos/abiertos con preservación de límites (diseño) y pruebas unitarias.
- Integrar `N_i` y `k_i` como transformaciones naturales hacia `Sheaves` (nombres→secciones parciales) con certificados.
- Publicar artefactos de Fase 1 en CI (añadir a workflow de subida de artefactos).
