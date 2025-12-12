using VelocityDistributionFunctions
import VelocityDistributionFunctions: _pdf_1d, V
using Test
using Random
using LinearAlgebra: norm, dot
using Statistics: mean, var
using Unitful

@testset "Maxwellian Distribution" begin
    @testset "Construction" begin
        # Valid construction
        d = Maxwellian(1.0)
        @test d.vth == 1.0

        # Invalid construction
        @test_throws "DomainError with -1.0:" Maxwellian(-1.0)  # negative vth
        @test_nowarn Maxwellian(-1.0; check_args = false)
        @test_throws "Maxwellian: the condition length(u0) == 3 is not satisfied." Maxwellian(1.0, [1, 0])  # wrong u0 dimension
    end

    # https://docs.plasmapy.org/en/stable/api/plasmapy.formulary.distribution.Maxwellian_velocity_3D.html#plasmapy.formulary.distribution.Maxwellian_velocity_3D
    @testset "PDF evaluation with Unitful" begin
        T = 30000u"K" # Temperature
        vdf = Maxwellian(T)
        𝐯 = ones(3) .* 1u"m/s"
        @test vdf(𝐯) ≈ 2.0708893e-19 * 1u"s^3/m^3"
        @test eltype(rand(vdf)) <: Quantity

        vdf2 = ustrip(vdf)
        @test vdf2(ustrip(𝐯)) ≈ 2.0708893e-19
        @test vdf2(V(1)) ≈ 2.0708893e-19 * 4π * 1^2
    end
end

@testset "BiMaxwellian Distribution" begin
    @testset "Construction" begin
        # Valid construction
        vdf = BiMaxwellian(1.0, 2.0)
        @test vdf.vth_perp == 1.0
        @test vdf.vth_para == 2.0

        # Invalid construction
        @test_throws DomainError BiMaxwellian(-1.0, 2.0)  # negative vth_perp
        @test_throws DomainError BiMaxwellian(1.0, 2.0, [1, 0])  # wrong u0 dimension
        @test_throws DomainError BiMaxwellian(1.0, 2.0, [0, 0, 0], [1, 0])  # wrong B_dir dimension
    end

    @testset "Unitful PDF evaluation" begin
        using Unitful
        T = 30000u"K"
        vdf = BiMaxwellian(T, T)
        𝐯 = ones(3) .* 1u"m/s"
        @test vdf(𝐯) ≈ 2.0708893e-19 * 1u"s^3/m^3"
        @test eltype(rand(vdf)) <: Quantity

        @test vdf(VPar(0u"m/s")) ≈ 5.916328704919331e-7 * 1u"s/m"
        @test vdf(VPerp(0u"m/s")) == 0u"s/m"
    end

    @testset "Drift velocity" begin
        Random.seed!(42)
        u0 = [0.0, 0.0, 3.0]  # Drift along z-axis
        d = BiMaxwellian(1.0, 1.0, u0)

        # Sample and check mean velocity
        n = 256
        samples = rand(d, n)
        mean_v = mean(samples)
        @info abs.(mean_v - u0)
        @test all(abs.(mean_v - u0) .< 0.1)
    end
end

@testset "Sampling" begin
    for f in [Maxwellian, BiMaxwellian]
        Random.seed!(42)
        d = f(1.0)
        # Single sample
        v = rand(d)
        @test length(v) == 3
        @test all(isfinite, v)

        # Multiple samples
        n = 100
        samples = rand(d, n)
        @test length(samples) == n
    end
end


@testset "Kappa Distribution" begin
    @testset "Construction" begin
        # Valid construction
        d = Kappa(1, 3)
        @test d.κ == 3.0
        @test d.vth == 1.0

        # Invalid construction
        @test_throws DomainError Kappa(1.0, 1.0)  # κ too small
        @test_throws DomainError Kappa(3.0, -1.0)  # negative vth
    end

    # https://docs.plasmapy.org/en/stable/api/plasmapy.formulary.distribution.kappa_velocity_3D.html
    @testset "Unitful" begin
        T = 30000u"K" # Temperature
        vdf = Kappa(T, 4.0)
        𝐯 = ones(3) .* 1u"m/s"
        @test vdf(𝐯) ≈ 3.7833969124639276e-19 * 1u"s^3/m^3"
        @test eltype(rand(vdf)) <: Quantity
        @test eltype(eltype(rand(vdf, 2))) <: Quantity
        p0 = 6.755497421769535e-7u"s/m"
        @test _pdf_1d(vdf, 1u"m/s") ≈ p0
        @test _pdf_1d.(vdf, [0u"m/s", 1u"m/s", 2u"m/s"]) ≈ [p0, p0, p0]
    end

    @testset "Sampling" begin
        κ = 2.5
        vth = 1000.0
        d = Kappa(vth, κ)
        Random.seed!(1234)
        samples = rand(d, 10000)
        # Theoretical Variance per dimension = vth^2 * κ / (2κ - 3)
        theoretical_var = (vth^2 * κ) / (2 * κ - 3)
        empirical_var_dim = sum(v -> v' * v, samples) / length(samples) / 3
        @test empirical_var_dim ≈ theoretical_var rtol = 5.0e-2
    end
end
