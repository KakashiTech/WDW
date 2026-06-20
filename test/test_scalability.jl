using WDW
using Test
const S = WDW.ScalableWDW

println("=== TEST DE ESCALABILIDAD WDW ===")

# Probar n=128, 256, 512, 1024
for n in [128, 256, 512, 1024]
    println("\nProbando n=$n")
    try
        t = @elapsed pipeline = S.ScalablePipeline(n)
        println("  Pipeline creado en ", round(t, digits=3), "s")
        @test t >= 0
        
        data = randn(n)
        t = @elapsed state, T_mat, thetas = S.process_scalable(pipeline, data)
        println("  Procesado en ", round(t, digits=3), "s")
        @test t >= 0
        @test isfinite(state.equivariance_error)
        @test state.complexity >= 0
        println("  Equivariance error: ", round(state.equivariance_error, digits=8))
        println("  Complexity: ", round(state.complexity, digits=4))
        println("  ✓ ÉXITO")
    catch e
        println("  ✗ ERROR: ", e)
        @test false
    end
end

println("\n=== COMPLETADO ===")
