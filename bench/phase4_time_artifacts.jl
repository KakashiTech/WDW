#!/usr/bin/env julia
using WDW
using LinearAlgebra

const TI = WDW.TimeITE
const TM = WDW.TimeMulti
const TH = WDW.TimeHyper
const P  = WDW.Planner

# ITE single-axis: energy monotonicity
H = Symmetric([2.0 0.0; 0.0 1.0])
psi0 = [1.0, 1.0]
psi, energies = TI.evolve(Matrix(H), psi0, 0.1, 20)
mono_ite = TI.monotone_energy(energies)

# Multi-time metric (36 time axes, 3 space)
dt = randn(36)
dx = randn(3)
ds2 = TH.simulate_metric(dt, dx)

# Hyper-time evolution with two axes
H1 = Symmetric([2.0 0.1; 0.1 1.0])
H2 = Symmetric([1.5 0.0; 0.0 0.5])
psi2, energies_axes = TH.evolve_wavefunction([Matrix(H1), Matrix(H2)], psi0, [0.1, 0.1], 20)
mono_ht = TH.monotone_energies_axes(energies_axes)

# Chronos-Kairos scheduling (T1/T3 interleave)
seq = collect(1:8)
sched = P.schedule_ck(seq, 2, 3)

isdir("bench") || mkpath("bench")
open("bench/phase4_time_metrics.csv", "w") do io
    println(io, "metric,value")
    println(io, "ite_mono,", mono_ite)
    println(io, "hypertime_mono,", mono_ht)
    println(io, "ds2,", round(ds2, digits=6))
    println(io, "sched_len,", length(sched))
end

open("bench/phase4_time_certificate.txt", "w") do io
    println(io, "WDW++ Phase 4′ Certificate: Hyper-Time & ITE")
    println(io, "ite_energy_monotone=$(mono_ite)")
    println(io, "hypertime_energy_sum_monotone=$(mono_ht)")
    println(io, "metric_ds2=$(round(ds2, digits=6))")
    println(io, "schedule_len=$(length(sched))")
end
