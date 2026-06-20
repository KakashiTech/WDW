#!/usr/bin/env julia

# Unified benchmark runner
# Usage:
#   julia --project bench/run_benchmark.jl [n] [model] [dataset] [epochs]
#
# Parameters:
#   n       - signal length (default: 784 for 28x28 images)
#   model   - model type: wdw, mlp, cnn (default: wdw)
#   dataset - dataset: rotated_mnist, synthetic (default: rotated_mnist)
#   epochs  - number of training epochs (default: 50)

using WDW, Statistics, Printf, Random, Dates

const ASF = WDW.AutoSymmetryFlux

function parse_args()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 784
    model = length(ARGS) >= 2 ? ARGS[2] : "wdw"
    dataset = length(ARGS) >= 3 ? ARGS[3] : "rotated_mnist"
    epochs = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 50
    return n, model, dataset, epochs
end

function run_benchmark(n::Int, model::String, dataset::String, epochs::Int)
    println("="^72)
    println("Unified WDW Benchmark")
    println("="^72)
    println("  Config:")
    println("    n       = $n")
    println("    model   = $model")
    println("    dataset = $dataset")
    println("    epochs  = $epochs")
    println("    date    = $(now())")
    println("="^72)

    n_train, n_test = 1000, 200

    println("\n[1/3] Loading dataset...")
    if dataset == "rotated_mnist"
        ds = ASF.load_rotated_mnist(n_train=n_train, n_test=n_test, max_angle=2π)
        X_train = [vec(ds.X_train[:, :, i]) for i in 1:size(ds.X_train, 3)]
        X_test  = [vec(ds.X_test[:, :, i])  for i in 1:size(ds.X_test, 3)]
        X_train_cnn = [reshape(ds.X_train[:, :, i], 28, 28, 1, 1) for i in 1:size(ds.X_train, 3)]
        X_test_cnn  = [reshape(ds.X_test[:, :, i], 28, 28, 1, 1)  for i in 1:size(ds.X_test, 3)]
    else
        error("Unknown dataset: $dataset")
    end
    Y_train, Y_test = ds.Y_train, ds.Y_test
    println("  ✓ $(length(X_train)) train, $(length(X_test)) test samples")

    println("\n[2/3] Training...")
    Random.seed!(42)

    if model == "wdw"
        m = ASF.WDWAutoSymmetryModel(n, 16, 10; liegan_hidden=[256, 128], classifier_hidden=[64, 32])
        ASF.train_wdw_model!(m, X_train, Y_train, epochs; lr=0.001, batch_size=64)
        acc = ASF.evaluate_wdw_model(m, X_test, Y_test)
    elseif model == "mlp"
        m = ASF.BaselineMLP(n, 10; hidden_dims=[256, 128, 64])
        ASF.train_baseline!(m, X_train, Y_train, epochs; lr=0.001, batch_size=64)
        acc = ASF.evaluate_baseline(m, X_test, Y_test)
    elseif model == "cnn"
        m = ASF.BaselineCNN(10)
        ASF.train_baseline!(m, X_train_cnn, Y_train, epochs; lr=0.001, batch_size=64)
        acc = ASF.evaluate_baseline(m, X_test_cnn, Y_test)
    else
        error("Unknown model: $model")
    end

    println("\n[3/3] Results")
    println("  Model:      $model")
    println("  Dataset:    $dataset")
    println("  Epochs:     $epochs")
    println("  Accuracy:   $(round(acc * 100, digits=2))%")
    println("="^72)
    return acc
end

@time run_benchmark(parse_args()...)
println("\n✓ Benchmark complete")
