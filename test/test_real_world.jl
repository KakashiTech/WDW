using WDW
using Random
using Printf
using Test

const RWA = WDW.RealWorldApplications
const RB = WDW.RealBaselines
const S = WDW.ScalableWDW

println("="^90)
println("TEST DE APLICACIONES REALES WDW")
println("="^90)

seed = 42
Random.seed!(seed)

# Test 1: Poisson 2D
println("\n>>> Test 1: Ecuación de Poisson 2D (n=256)")
try
    RWA.compare_on_real_problem("poisson", 256, seed=seed)
    println("✓ Poisson 2D completado")
    @test true  # Poisson completado sin error
catch e
    println("✗ Error en Poisson: ", e)
    @test false
end

# Test 2: Graph Propagation
println("\n>>> Test 2: Propagación en Grafo (n=256)")
try
    RWA.compare_on_real_problem("graph", 256, seed=seed)
    println("✓ Graph propagation completado")
    @test true  # Graph completado sin error
catch e
    println("✗ Error en Graph: ", e)
    @test false
end

# Test 3: Oscillators
println("\n>>> Test 3: Osciladores Acoplados (n=256)")
try
    RWA.compare_on_real_problem("oscillators", 256, seed=seed)
    println("✓ Oscillators completado")
    @test true  # Oscillators completado sin error
catch e
    println("✗ Error en Oscillators: ", e)
    @test false
end

println("\n" * "="^90)
println("✓ TODAS LAS APLICACIONES REALES COMPLETADAS")
println("="^90)
