module WDW

# =============================================================================
# FOUNDATIONS (Tier 3 — Experimental mathematics)
# Loaded first as many modules depend on these.
# =============================================================================

include("Logic/DSL.jl")               # Tier 3 — Formula DSL
include("Semantics/Kripke.jl")        # Tier 3 — Kripke semantics
include("Category/Sets.jl")           # Tier 3 — Finite sets
include("Knowledge/TopologicalFunctors.jl")  # Tier 3 — Topological spaces
include("Sheaves/FiniteSheaves.jl")   # Tier 3 — Sheaf theory
include("Algebra/Quivers.jl")         # Tier 3 — Quiver representation theory
include("Motives/ComputableMotives.jl")       # Tier 3 — Motivic features
include("Motives/MotivicReduce.jl")           # Tier 3 — Motivic dim reduce
include("Quantum/QGroupENN.jl")       # Tier 2 — Group-equivariant framework
include("Tensor/HolographicCodes.jl") # Tier 2 — MERA compression
include("Krylov/Complexity.jl")       # Tier 3 — Krylov complexity
include("Time/ITE.jl")                # Tier 3 — Imaginary-time evolution
include("Time/MultiTime.jl")          # Tier 3 — Multi-agent time
include("Time/HyperTime.jl")          # Tier 3 — Hyper-time
include("Planner/ChronosKairos.jl")   # Tier 3 — Scheduling
include("Bio/Microtubules.jl")        # Tier 3 — Microtubule lattice
include("Gravity/LQGDataSpace.jl")    # Tier 3 — Spin networks
include("Vacuum/QET.jl")              # Tier 3 — QET analogs

# =============================================================================
# UNIFIED PIPELINE (Tier 3 — loaded here because Tier 1 depends on it)
# =============================================================================

include("UnifiedWDW.jl")
include("RuptureABC.jl")

# =============================================================================
# TIER 1 — CORE: Verified, documented, production-ready
# =============================================================================

include("ScalableWDW.jl")
include("FFTGroup.jl")
include("FFTPipeline.jl")

# =============================================================================
# TIER 2 — RESEARCH EXTENSIONS + Tier 3 experiments
# =============================================================================

include("RealBaselines.jl")
include("RealWorldApplications.jl")
include("PaperMetrics.jl")
include("WDWAutoencoder.jl")
include("TheoreticalMetrics.jl")
include("RigorousMetrics.jl")
include("MultiDataset.jl")
include("LatticePhonons.jl")
include("BreakthroughExperiment.jl")
include("AutoSymmetryDiscovery.jl")
include("StructuralExperiments.jl")
include("SymmetryDiscovery.jl")
include("SymmetryCertificate.jl")
include("UnifiedIntegration.jl")
include("StructuralEmbedding.jl")
include("AutoSymmetryFlux.jl")

# =============================================================================
# DEPRECATED (kept for backward compatibility)
# =============================================================================

include("mlp_baseline.jl")

# =============================================================================
# EXPORTS
# =============================================================================

export Logic, Semantics, Category, Knowledge, Sheaves, Algebra, Motives, MotivicReduce,
       Quantum, Tensor, Krylov, TimeITE, TimeMulti, TimeHyper, Planner, Bio, Gravity,
       Vacuum, UnifiedWDW, RuptureABC, ScalableWDW, RealBaselines, RealWorldApplications,
       PaperMetrics, WDWAutoencoder, TheoreticalMetrics, RigorousMetrics, MultiDataset,
       LatticePhonons, BreakthroughExperiment, AutoSymmetryDiscovery,
       StructuralExperiments, FFTGroup, FFTPipeline, SymmetryDiscovery,
       SymmetryCertificate, UnifiedIntegration, StructuralEmbedding

end
