#!/usr/bin/env julia
"""
    download_mnist.jl

Download MNIST dataset for WDW benchmarks.
Run once before using AutoSymmetryFlux with RotatedMNIST.

Usage:
    julia data/download_mnist.jl
"""
using Downloads

const BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
const FILES = [
    ("train-images-idx3-ubyte", "train-images-idx3-ubyte.gz", false),
    ("train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz", false),
    ("t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz", false),
    ("t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz", false),
]

for (name, url_name, _) in FILES
    path = joinpath(@__DIR__, name)
    if isfile(path)
        println("✓ $name already exists")
        continue
    end
    url = "$BASE/$url_name"
    gz_path = "$path.gz"
    println("Downloading $url...")
    Downloads.download(url, gz_path)
    run(`gunzip -f $gz_path`)
    println("✓ $name saved")
end
println("\nAll MNIST files ready.")
