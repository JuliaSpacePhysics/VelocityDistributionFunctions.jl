using Test
using VelocityDistributionFunctions

@testset "3-D input: one time step" begin
    # flux: (phi, theta, time)
    flux = zeros(Float64, 2, 2, 1)
    flux[1, 1, 1] = 2.0  # phi=0°, theta=0° -> pitch=0°
    flux[2, 2, 1] = 4.0  # phi=180°, theta=180° -> pitch=180°


    # test both time-invariant and time-varying B and phi
    for B in ([0.0, 0.0, -1.0], reshape([0.0, 0.0, -1.0], 3, 1))
        for phi in ([0.0; 180.0;;], [0.0, 180.0])
            theta = [0.1; 179.9;;]

            pad = tpitch_angle_distribution(flux, B, phi, theta; bins = 2)

            @test pad.data == [1.0; 2.0;;]
            @test pad.pitch_angles == [45.0, 135.0]
        end
    end
end

@testset "4-D input: one time and one energy" begin
    # flux: (phi, theta, energy, time)
    flux = zeros(Float64, 2, 2, 1, 1)
    flux[1, 1, 1, 1] = 3.0  # phi=0°, theta=0° -> pitch=0°
    flux[2, 2, 1, 1] = 6.0  # phi=180°, theta=180° -> pitch=180°

    B = reshape([0.0, 0.0, -1.0], 3, 1)  # (3, time)
    phi = [0.0, 180.0]
    theta = [0.1, 179.9]

    pad = tpitch_angle_distribution(flux, B, phi, theta; bins = 2)

    @test size(pad.data) == (2, 1, 1)  # (nbins, energy, time)
    @test pad.data[:, 1, 1] == [1.5, 3.0]
    @test pad.pitch_angles == [45.0, 135.0]
end
