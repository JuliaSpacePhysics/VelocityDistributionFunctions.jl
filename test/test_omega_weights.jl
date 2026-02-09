using Test
using VelocityDistributionFunctions: _domega, omega_weights, omega_weightsd

@testset "_domega" begin
    @test _domega(0, pi, 2pi) ≈ 4π atol = 1.0e-12
end

@testset "omega_weights" begin
    @testset "full-sphere solid angle" begin
        # A single bin covering the full sphere: theta ∈ [-90,90], phi ∈ [0,360]
        theta = fill(0.0, 1, 1)
        phi = fill(180.0, 1, 1)
        dtheta = fill(180.0, 1, 1)
        dphi = fill(360.0, 1, 1)

        ω = omega_weightsd.(theta, phi, dtheta, dphi)
        @test ω[1, 1][1] ≈ 4π atol = 1.0e-12
    end

    @testset "hemisphere solid angle" begin
        # Two bins splitting into north (theta=45,dtheta=90) and south (theta=-45,dtheta=90)
        theta = [45.0 -45.0]
        phi = [180.0 180.0]
        dtheta = [90.0 90.0]
        dphi = [360.0 360.0]

        ω = omega_weightsd.(theta, phi, dtheta, dphi)
        @test ω[1, 1][1] ≈ 2π atol = 1.0e-12
        @test ω[1, 2][1] ≈ 2π atol = 1.0e-12
    end

    @testset "symmetry: uniform phi bins" begin
        # 4 equal phi bins at theta=0, each 90° wide
        n_phi = 4
        theta = zeros(1, n_phi)
        phi = [45.0 135.0 225.0 315.0]
        dtheta = fill(180.0, 1, n_phi)
        dphi = fill(90.0, 1, n_phi)

        ω = omega_weightsd.(theta, phi, dtheta, dphi)
        # Total solid angle should be 4π
        @test sum(getindex.(ω, 1)) ≈ 4π atol = 1.0e-12
    end
end
