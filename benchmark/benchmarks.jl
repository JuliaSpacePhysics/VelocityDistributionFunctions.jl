using BenchmarkTools
using VelocityDistributionFunctions
using VelocityDistributionFunctions: directional_energy_spectra, PAspectra, sort_flux_by_pitch_angle!
using JLD2

const SUITE = BenchmarkGroup()

include(joinpath(@__DIR__, "../test/fixtures/elfin_helpers.jl"))

# Spectra benchmarks on the real ELFIN EPD fixture, tiled along time to a full
# day's worth of half-spin records (~1731).
let
    (; S, pa, LC) = load_elfin_fixture(; n_t = 1731)
    params = elfin_des_params(size(pa, 1))

    g = SUITE["spectra"] = BenchmarkGroup()
    g["directional_energy_spectra"] = @benchmarkable directional_energy_spectra($S, $pa, $LC; $params...)

    S_sorted = (s = copy(S); sort_flux_by_pitch_angle!(s, pa); s)
    g["PAspectra"] = @benchmarkable PAspectra($S_sorted, $ELFIN_EPD_dE, $(ELFIN_PA_CHANNELS)...)
end

# tmoments benchmark with synthetic MMS-like data
let nt = 100, nphi = 32, ntheta = 16, nenergy = 32
    data = rand(Float32, nt, nphi, ntheta, nenergy)
    theta = collect(range(0.0, 180.0, length = ntheta))
    phi = collect(range(0.0, 360.0, length = nphi + 1)[1:nphi])
    energy = 10 .^ collect(range(log10(10.0), log10(30000.0), length = nenergy))
    scpot = zeros(nt)
    magf = randn(nt, 3)

    g = SUITE["tmoments"] = BenchmarkGroup()
    g["no_magf"] = @benchmarkable tmoments($data, $theta, $phi, $energy, $scpot; edim = 4, tdim = 1, units = :df_cm, species = :H)
    g["with_magf"] = @benchmarkable tmoments($data, $theta, $phi, $energy, $scpot, $magf; edim = 4, tdim = 1, units = :df_cm, species = :H)
end
