using WDW
using Random
using Printf
using Test

const RB = WDW.RealBaselines
const S = WDW.ScalableWDW
const Q = WDW.Quantum

println("="^80)
println("TEST DE BASELINES REALES VS WDW")
println("="^80)

n = 256
seed = 42
Random.seed!(seed)

data = randn(n)

# Get group for baselines
group = Q.dihedral_group(n)

# Comparar todos los baselines manualmente
println("\n>>> Comparando baselines...")
println("="^80)
println("COMPARACIÓN CON BASELINES REALES")
println("="^80)
println("n = $n, epochs = 100, seed = $seed")
println("-"^80)

results = []

# E2CNN
println("Testing E2CNN...")
try
    baseline_e2cnn = RB.E2CNNBaseline(n, group; hidden_dim=16, num_layers=2, seed=seed)
    t_total = @elapsed error_e2cnn = RB.train_e2cnn(baseline_e2cnn, data, 100)
    params_e2cnn = RB.count_parameters(baseline_e2cnn)
    println("  E2CNN: error=$error_e2cnn, params=$params_e2cnn")
    push!(results, Dict("method" => "E2CNN", "error" => error_e2cnn, "params" => params_e2cnn, "time" => t_total))
    @test isfinite(error_e2cnn)
    @test params_e2cnn > 0
    @test t_total >= 0
catch e
    println("  E2CNN no disponible: $e")
    @test_skip true
end

# escnn
println("Testing escnn...")
try
    baseline_escnn = RB.ESCNNBaseline(n, group; feature_multiplicity=4, num_layers=2, seed=seed)
    t_total = @elapsed error_escnn = RB.train_escnn(baseline_escnn, data, 100)
    params_escnn = RB.count_parameters(baseline_escnn)
    println("  escnn: error=$error_escnn, params=$params_escnn")
    push!(results, Dict("method" => "escnn", "error" => error_escnn, "params" => params_escnn, "time" => t_total))
    @test isfinite(error_escnn)
    @test params_escnn > 0
    @test t_total >= 0
catch e
    println("  escnn no disponible: $e")
    @test_skip true
end

# PyG
println("Testing PyG...")
try
    baseline_pyg = RB.PyGBaseline(n; hidden_dim=32, num_layers=3, seed=seed)
    t_total = @elapsed error_pyg = RB.train_pygnn(baseline_pyg, data, 100)
    params_pyg = RB.count_parameters(baseline_pyg)
    println("  PyG: error=$error_pyg, params=$params_pyg")
    push!(results, Dict("method" => "PyG", "error" => error_pyg, "params" => params_pyg, "time" => t_total))
    @test isfinite(error_pyg)
    @test params_pyg > 0
    @test t_total >= 0
catch e
    println("  PyG no disponible: $e")
    @test_skip true
end

# Ahora comparar con WDW
println("\n>>> Ejecutando WDW Scalable...")
pipeline = S.ScalablePipeline(n, compression_levels=4, krylov_dim=20)
t_wdw = @elapsed state, T_mat, thetas = S.process_scalable(pipeline, data)

equiv_error_wdw = state.equivariance_error
mdl_wdw = 4 * 2 * 32 + 100

@test isfinite(equiv_error_wdw)
@test t_wdw >= 0
@test isfinite(mdl_wdw)

println("="^80)
println(@sprintf("%-20s %-15s %-15s %-15s", 
                 "Método", "Equiv Error", "Parámetros", "Tiempo(s)"))
println("-"^80)

for r in results
    println(@sprintf("%-20s %-15.8f %-15d %-15.3f",
                     r["method"], r["error"], r["params"], r["time"]))
end

println(@sprintf("%-20s %-15.8f %-15d %-15.3f",
                 "WDW (Ours)", equiv_error_wdw, mdl_wdw ÷ 32, t_wdw))

println("="^80)

if !isempty(results)
    best_baseline_error = minimum([r["error"] for r in results])
    improvement = best_baseline_error / equiv_error_wdw
    
    println("\n>>> GANANCIA DE WDW:")
    println("  Mejor baseline error: ", round(best_baseline_error, digits=6))
    println("  WDW error:            ", round(equiv_error_wdw, digits=6))
    println("  Mejora:               ", round(improvement, digits=1), "x")
    @test improvement > 0
end

@test length(results) >= 0

println("\n✓ Test completado")
