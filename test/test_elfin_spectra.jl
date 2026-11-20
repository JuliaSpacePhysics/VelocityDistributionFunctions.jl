using Test
using VelocityDistributionFunctions

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

    @testset "directional_spectra" begin
        res = directional_spectra(S, eachcol(pa), LC; tdim = 3, params...)
        @test size(res.para) == (16, 1)
        @test res.para[:, 1] ≈ PARA_T1
        @test res.anti[:, 1] ≈ ANTI_T1
        @test res.perp[:, 1] ≈ PERP_T1
        @test res.omni[:, 1] ≈ OMNI_T1
    end

    @testset "group_mean" begin
        pas = group_mean(S, ELFIN_PA_CHANNELS; dims = 2, weights = ELFIN_EPD_dE)
        @test size(pas) == (10, 4, 1)
        @test isapprox_with_nan(pas[:, 1, 1], PA_CH0_T1)
        @test isapprox_with_nan(pas[:, 4, 1], PA_CH3_T1)
        # the grouped axis may sit anywhere; one scan has no time axis
        @test isequal(group_mean(permutedims(S[:, :, 1]), ELFIN_PA_CHANNELS; dims = 1, weights = ELFIN_EPD_dE), permutedims(pas[:, :, 1]))
        @test group_mean([1, 2, 3], [1:2, 3:3]; dims = 1) == [1.5, 3]
        @test group_mean([1, 2, 3], [1:2, 3:3]; dims = 1, weights = [1, 3, 2]) == [1.75, 3]
        @test_throws ArgumentError group_mean(S, [15:17]; dims = 2)
        @test_throws DimensionMismatch group_mean(S, ELFIN_PA_CHANNELS; dims = 1, weights = ELFIN_EPD_dE)
    end
end

@testset "directional_spectra method = :sum" begin
    # loss cone 10°: sectors are para, perp, perp, anti; the para sector's flux is NaN
    res = directional_spectra([NaN; 2.0; 4.0; 8.0;;], [5.0, 60.0, 120.0, 175.0], 10.0; method = :sum)
    @test isequal(map(only, res), (; omni = 14.0, para = NaN, anti = 8.0, perp = 6.0))
end

@testset "directional_spectra of an angle_distribution" begin
    # counts at pitch 15° (para), 75° (perp) and 165° (anti) for two energies
    grid = EnergyAngleGrid([1, 2], 0, [15, 75, 165]; layout = (:theta, :energy), dphi = 360, dtheta = 30)
    pad = angle_distribution([1 2; 3 4; 5 6], grid, :pitch; B = [0, 0, 1], bins = 6, method = :sum)
    scan_counts(S, centers, lc) = directional_spectra(S, centers, lc; method = :sum)
    expected = (; omni = [9, 12], para = [1, 2], anti = [5, 6], perp = [3, 4])
    @test scan_counts(pad.data, pad.centers, 30) == expected

    @testset "shared centers, tdim = $tdim" for tdim in 1:3
        series = stack((pad.data, 2 .* pad.data); dims = tdim)
        @test directional_spectra(series, pad.centers, 30; method = :sum, tdim) == map(v -> [v 2v], expected)
    end

    @testset "per-scan centers and loss cones" begin
        flipped = reverse(pad.centers)
        res = directional_spectra(stack((pad.data, pad.data)), [pad.centers, flipped], [30, 90]; method = :sum, tdim = 3)
        @test isequal(res, map(hcat, scan_counts(pad.data, pad.centers, 30), scan_counts(pad.data, flipped, 90)))
    end

    @testset "without energy axis" begin
        grid1 = EnergyAngleGrid(nothing, 0, [15, 75, 165]; layout = (:theta,), dphi = 360, dtheta = 30)
        pad1 = angle_distribution([1, 3, 5], grid1, :pitch; B = [0, 0, 1], bins = 6, method = :sum)
        @test scan_counts(pad1.data, pad1.centers, 30) == map(first, expected)
        @test directional_spectra([pad1.data 2pad1.data], pad1.centers, 30; method = :sum, tdim = 2) == map(v -> [v[1], 2v[1]], expected)
    end

    series = stack((pad.data, pad.data))
    @test directional_spectra(series, pad.centers, 30; method = :sum) == directional_spectra(series, pad.centers, 30; method = :sum, tdim = 3)
    @test_throws DimensionMismatch directional_spectra(series, [pad.centers pad.centers], 30; method = :sum, tdim = 3) # matrix: use eachcol
end

@testset "non-finite pitch centers have no weight" begin
    res = directional_spectra([1; 100; 100;;], [90, Inf, -Inf], 30; half_sector_width = 5)
    @test only(res.omni) == only(res.perp) == 1
    @test isnan(only(res.para)) && isnan(only(res.anti))
end
