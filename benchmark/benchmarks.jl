using BenchmarkTools
using VelocityDistributionFunctions
using JLD2

const SUITE = BenchmarkGroup()

include(joinpath(@__DIR__, "../test/fixtures/elfin_helpers.jl"))

# Spectra benchmarks on the real ELFIN EPD fixture, tiled along time to a full
# day's worth of half-spin records (~1731).
let
    (; S, pa, LC) = load_elfin_fixture(; n_t = 1731)
    params = elfin_des_params(size(pa, 1))

    g = SUITE["spectra"] = BenchmarkGroup()
    g["directional_spectra"] = @benchmarkable directional_spectra($S, $(eachcol(pa)), $LC; tdim = 3, $params...)
    g["group_mean"] = @benchmarkable group_mean($S, $ELFIN_PA_CHANNELS; dims = 2, weights = $ELFIN_EPD_dE)
end

# plasma_moments benchmark with synthetic MMS-like data
let nt = 100, nphi = 32, ntheta = 16, nenergy = 32
    data = rand(Float32, nt, nphi, ntheta, nenergy)
    theta = collect(range(90 / ntheta, 180 - 90 / ntheta, length = ntheta))
    phi = collect(range(0.0, 360.0, length = nphi + 1)[1:nphi])
    energy = 10 .^ collect(range(log10(10.0), log10(30000.0), length = nenergy))
    scpot = zeros(nt)
    grid = EnergyAngleGrid(energy, phi, theta; layout = (:phi, :theta, :energy))
    B = eachrow(randn(nt, 3))

    g = SUITE["plasma_moments"] = BenchmarkGroup()
    g["no_magf"] = @benchmarkable plasma_moments($data, $grid; sc_pot = $scpot, tdim = 1, units = :df_cm, species = :H)
    g["with_magf"] = @benchmarkable plasma_moments($data, $grid; sc_pot = $scpot, B = $B, tdim = 1, units = :df_cm, species = :H)
end
