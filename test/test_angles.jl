using Test
using VelocityDistributionFunctions

# MMS FPI-like grid: 32 φ × 16 θ × 32 energies
mms_φ = 5.625:11.25:360
mms_θ = 5.625:11.25:180
mms_E = exp10.(range(1, 4.5; length = 32))
mms_grid = EnergyAngleBins(:phi => mms_φ, :theta => mms_θ, :energy => mms_E)
B_tilted = [0.3, -0.5, 0.8]

# 1° cells, so that bins of 15° hold whole θ rows when B ∥ ẑ
fine_grid = EnergyAngleBins(:phi => 0.5:359.5, :theta => 0.5:179.5, :energy => [1.0, 2.0])

@testset "isotropic f is flat in every coordinate" begin
    f = reshape(mms_E .^ -2, 1, 1, :)
    data = repeat(f, 32, 16, 1)
    for (coord, kw) in ((:pitch, (; B = B_tilted)), (:gyrophase, (; B = B_tilted, gyro_ref = [1, 0, 0])), (:phi, (;)), (:theta, (;)))
        @test angle_distribution(data, mms_grid, coord; kw...).data ≈ repeat(vec(f)', 12)
    end
    @test omni_spectrum(data, mms_grid) ≈ vec(f)
    @test omni_spectrum(data, EnergyAngleBins(:phi => mms_φ, :theta => mms_θ, :energy)) == omni_spectrum(data, mms_grid)
end

@testset "flattened angle axis matches (φ, θ, E) layout" begin
    φθ = vec(collect(Iterators.product(mms_φ, mms_θ)))
    flat_grid = EnergyAngleBins(:energy => mms_E, (:phi, :theta) => (first.(φθ), last.(φθ)); dphi = 11.25, dtheta = 11.25)
    cosα = [cosd(φ) * sind(θ) for φ in mms_φ, θ in mms_θ, _ in mms_E]
    @test angle_distribution(cosα, mms_grid, :pitch; B = [1, 0, 0]).data ≈
        angle_distribution(permutedims(reshape(cosα, :, 32)), flat_grid, :pitch; B = [1, 0, 0]).data
end

@testset "f ∝ cos²α: solid-angle-weighted bin means" begin
    data = repeat(reshape(cosd.(0.5:179.5) .^ 2, 1, :, 1), 360, 1, 2)
    res = angle_distribution(data, fine_grid, :pitch; B = [0, 0, 2])
    c1, c2 = cosd.(0:15:165), cosd.(15:15:180)
    @test res.data[:, 1] ≈ @.((c1^3 - c2^3) / 3 / (c1 - c2)) atol = 5.0e-4
    @test res.edges == 0:15:180 && res.centers == 7.5:15:172.5
    @test res.solid_angle[:, 1] ≈ @.(2π * (c1 - c2))
end

@testset "gyrophase is 0° along gyro_ref ⊥ B and 90° along B × gyro_ref" begin
    data = repeat(reshape(cosd.(0.5:359.5) .+ 2, :, 1, 1), 1, 180, 2)
    phi = angle_distribution(data, fine_grid, :phi; bins = 36).data
    @test angle_distribution(data, fine_grid, :gyrophase; B = [0, 0, 1], gyro_ref = [1, 0, 1], bins = 36).data ≈ phi
    @test angle_distribution(data, fine_grid, :gyrophase; B = [0, 0, 1], gyro_ref = [0, -1, 0], bins = 36).data ≈
        circshift(phi, (9, 0))
end

@testset "no energy axis; pitch exactly 0° and 180°" begin
    grid = EnergyAngleBins(:phi => 0:90:270, :theta => [30, 90, 150]; dphi = 90, dtheta = 60)
    data = zeros(Int, 4, 3)
    data[1, 2] = 2   # φ = 0°, θ = 90°: along B
    data[3, 2] = 4   # φ = 180°: against B
    @test (@inferred angle_distribution(data, grid, :pitch; B = [1, 0, 0], bins = 3, method = :sum)).data == [2, 0, 4]
    @test isequal(angle_distribution(data, grid, :pitch; B = [1, 0, 0], bins = 6, method = :sum).data, [2, NaN, 0, 0, 0, 4])  # no cell at pitch 30°–60°
    @test omni_spectrum(data, grid; method = :sum) == 6
    # B along (θ = 78°, φ = 0°) gives a direction cosine of 1 + eps
    grid78 = EnergyAngleBins(:phi => [0, 180], :theta => 78; dphi = 10, dtheta = 10)
    @test angle_distribution([1; 3;;], grid78, :pitch; B = [sind(78), 0, cosd(78)], bins = 2).data == [1, 3]
    @test_throws ArgumentError plasma_moments(ones(4, 3, 2), EnergyAngleBins(:phi => 0:90:270, :theta => [30, 90, 150], :energy); units = :eflux, species = :H)
end

@testset "counts, non-finite cells and undefined B" begin
    @test angle_distribution(ones(Int, 32, 16, 32), mms_grid, :theta; bins = 16, method = :sum).data == fill(32, 16, 32)
    data = ones(32, 16, 32)
    data[:, 1, 1] .= NaN
    data[1, 2, 1] = Inf
    res = angle_distribution(data, mms_grid, :theta; bins = 16)
    @test isnan(res.data[1, 1]) && res.data[2:end, 1] == ones(15) && res.data[:, 2:end] == ones(16, 31)
    @test res.solid_angle[1, 1] == 0 && res.solid_angle[2, 1] ≈ 31 / 32 * res.solid_angle[2, 2]
    @test all(isnan, angle_distribution(data, mms_grid, :pitch; B = [NaN, 0, 0]).data)
    @test all(isnan, angle_distribution(data, mms_grid, :gyrophase; B = [0, 0, 1], gyro_ref = [0, 0, 2]).data)
end

@testset "time series" begin
    data = rand(32, 16, 32, 3)
    Bs = [[1, 0, 0], [0, 1, 1], B_tilted]
    grids = [EnergyAngleBins(:phi => mms_φ, :theta => mms_θ, :energy => mms_E .* k) for k in 1:3]
    res = angle_distribution(data, grids, :pitch; B = Bs, tdim = 4)
    @test size(res.data) == size(res.solid_angle) == (12, 32, 3)
    @test res.data[:, :, 2] == angle_distribution(data[:, :, :, 2], grids[2], :pitch; B = Bs[2]).data
    time_first = permutedims(data, (4, 1, 2, 3))
    @test omni_spectrum(time_first, mms_grid; tdim = 1) ≈ stack(omni_spectrum.(eachslice(data; dims = 4), Ref(mms_grid)))
    # B[3, time] as in tpitch_angle_distribution
    @test_throws DimensionMismatch angle_distribution(data, mms_grid, :pitch; B = stack(Bs), tdim = 4)
end
