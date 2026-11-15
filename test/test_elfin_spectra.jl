using Test
using VelocityDistributionFunctions
using VelocityDistributionFunctions: directional_energy_spectra, pitch_angle_spectra, sort_flux_by_pitch_angle!

include(joinpath(@__DIR__, "fixtures/elfin_helpers.jl"))

# Fixture: first time step of `ela_l2_epdef_20201001_v01.cdf` (ELFIN A, L2 EPDEF).

const PARA_T1 = Float32[20541.432, 11046.116, 2887.873, 3359.2217, 434.48523, 173.7941, 0.0, 152.99594, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 85.875145]
const ANTI_T1 = Float32[30038.154, 7533.442, 930.06287, 739.7119, 352.8135, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
const PERP_T1 = Float32[958391.0, 533326.7, 294976.4, 237197.5, 84218.24, 36682.793, 9740.965, 2550.948, 972.00995, 719.9266, 531.83734, 266.66074, 101.549866, 40.508488, 104.62714, 47.042114]
const OMNI_T1 = Float32[550893.44, 304458.8, 166312.48, 134484.42, 48099.812, 20565.47, 5485.3193, 1441.3925, 539.3132, 399.4464, 295.08636, 147.95491, 56.344257, 22.47586, 58.051662, 40.703686]
const PA_CH0_T1 = Float32[NaN, 3962.1436, 13528.301, 64589.23, 857748.6, 705126.7, 39205.92, 12680.62, 3961.9666, NaN]
const PA_CH3_T1 = Float32[NaN, 0.0, 37.952724, 8.959968, 290.94736, 160.54758, 0.0, 0.0, 0.0, NaN]

isapprox_with_nan(a, b; kw...) = length(a) == length(b) &&
    all(zip(a, b)) do (x, y)
    (isnan(x) && isnan(y)) || isapprox(x, y; kw...)
end

@testset "Spectra on ELFIN EPD fixture" begin
    (; S, pa, LC) = load_elfin_fixture()
    @test size(S) == (10, 16, 1)

    params = elfin_des_params(size(pa, 1))

    @testset "directional_energy_spectra" begin
        res = directional_energy_spectra(S, pa, LC; params...)
        @test size(res.para) == (16, 1)
        @test res.para[:, 1] ≈ PARA_T1
        @test res.anti[:, 1] ≈ ANTI_T1
        @test res.perp[:, 1] ≈ PERP_T1
        @test res.omni[:, 1] ≈ OMNI_T1
    end

    @testset "pitch_angle_spectra" begin
        S2 = copy(S)
        sort_flux_by_pitch_angle!(S2, pa)
        pas = pitch_angle_spectra(S2, ELFIN_EPD_dE, ELFIN_PA_CHANNELS...)
        @test length(pas) == 4
        @test size(pas[1]) == (10, 1)
        @test isapprox_with_nan(pas[1][:, 1], PA_CH0_T1)
        @test isapprox_with_nan(pas[4][:, 1], PA_CH3_T1)
    end
end
