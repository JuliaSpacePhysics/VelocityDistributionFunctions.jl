using VelocityDistributionFunctions
import VelocityDistributionFunctions: _pdf_1d, V, ShiftedPDF
using Test
using Random
using LinearAlgebra: norm, dot
using Statistics: mean, var
using Unitful
using Unitful: 𝐓, 𝐋, dimension

@testset "Maxwellian Distribution" begin
    @testset "Construction" begin
        # Valid construction
        d = Maxwellian(1.0)
        @test d.vth == 1.0

        density = 10.0
        d = Maxwellian(density, 1.0)
        @test d.n == density
        @test d.vth == 1.0
        @test identity.(d) == d
        @test Maxwellian(density, 1.0; u0 = [1, 0, 0]).vth == 1.0

        # Invalid construction
        @test_throws "DomainError with -1.0:" Maxwellian(-1.0)  # negative vth
        @test_nowarn Maxwellian(-1.0; check_args = false)
        @test_throws "ShiftedPDF: the condition length(u0) == length(base) is not satisfied." Maxwellian(1.0; u0 = [1, 0])  # wrong u0 dimension
    end

    @testset "Sampling" begin
        for vdf in [Maxwellian(1.0), Maxwellian(1.0, 2.0)]
            Random.seed!(42)
            # Single sample
            @test length(rand(vdf, 1)[1]) == 3
            @test length(rand(vdf, 1, 1)[1]) == 3
            @test length(rand(vdf)) == 3

            # Multiple samples
            n = 2
            samples = rand(vdf, n)
            @test length(samples) == n
        end
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

    @testset "Physical distribution wrapper" begin
        n = 1.0u"m^-3"
        T = 30000u"K"
        d = Maxwellian(n, T)
        𝐯 = ones(3) .* 1u"m/s"
        @test d(𝐯) ≈ n * Maxwellian(T)(𝐯)
        p = n * Unitful.k * T
        d2 = Maxwellian(n, p)
        @test d2(𝐯) ≈ d(𝐯)
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
        @test_throws DomainError BiMaxwellian(1.0, 2.0, [1, 0])  # wrong B_dir dimension
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

        vdf2 = ShiftedPDF(vdf, 𝐯)
        @test vdf2(VPar(1u"m/s")) == vdf(VPar(0u"m/s"))
        @test_throws MethodError vdf2(VPerp(0u"m/s"))
    end

    @testset "Physical distribution wrapper" begin
        n = 1.0u"m^-3"
        T = 30000u"K"
        d = BiMaxwellian(n, T, T)
        p = n * Unitful.k * T
        d2 = BiMaxwellian(n, p, p)
        𝐯 = ones(3) .* 1u"m/s"
        @test d2(𝐯) ≈ d(𝐯)
    end

    @testset "Drift velocity" begin
        Random.seed!(42)
        u0 = [0.0, 0.0, 3.0]  # Drift along z-axis
        d = BiMaxwellian(1.0, 1.0; u0)
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
        n = 2
        samples = rand(d, n)
        @test length(samples) == n
    end
end

@testset "Physical Distribution Sampling with RNG" begin
    rng = Random.default_rng()
    d = Maxwellian(1.0, 2.0) # returns VelocityDistribution
    v = rand(rng, d)
    @test length(v) == 3
    @test all(isfinite, v)
    
    # Test reproducibility and correctness against underlying pdf
    rng1 = Random.MersenneTwister(123)
    v1 = rand(rng1, d)
    rng2 = Random.MersenneTwister(123)
    v2 = rand(rng2, d.pdf)
    @test v1 == v2
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

    @testset "Physical distribution wrapper" begin
        n = 1.0u"m^-3"
        T = 30000u"K"
        κ = 4.0
        d = Kappa(n, T, κ)
        p = n * Unitful.k * T
        d2 = Kappa(n, p, κ)
        𝐯 = ones(3) .* 1u"m/s"
        @test d2(𝐯) ≈ d(𝐯)
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


@testset "BiKappa Distribution" begin
    @testset "Construction" begin
        d = BiKappa(1.0, 2.0, 3.0)
        @test d.vth_perp == 1.0
        @test d.vth_para == 2.0
        @test d.κ == 3.0

        @test_throws DomainError BiKappa(1.0, 2.0, 1.0) # κ too small
        @test_throws DomainError BiKappa(-1.0, 2.0, 3.0) # negative vth_perp
    end

    @testset "Matches isotropic Kappa when vth_perp == vth_para" begin
        vth = 1.2
        κ = 4.5
        u0 = [0.1, -0.2, 0.3]
        b0 = [0.0, 0.0, 1.0]
        d_iso = Kappa(vth, κ; u0)
        d_bi = BiKappa(vth, vth, κ, b0; u0)
        v = [0.7, -0.4, 0.9]
        @test d_bi(v) ≈ d_iso(v)
    end

    @testset "Unitful" begin
        T = 30000u"K" # Temperature
        vdf = BiKappa(T, T, 4.0)
        𝐯 = ones(3) .* 1u"m/s"
        @test vdf(𝐯) ≈ 3.7833969124639276e-19 * 1u"s^3/m^3"
    end

    @testset "Physical distribution wrapper" begin
        n = 1.0u"m^-3"
        T = 30000u"K"
        κ = 4.0
        d = BiKappa(n, T, T, κ)
        p = n * Unitful.k * T
        d2 = BiKappa(n, p, p, κ)
        𝐯 = ones(3) .* 1u"m/s"
        @test dimension(d2(𝐯)) == 𝐓^3 / 𝐋^6
        @test d2(𝐯) ≈ d(𝐯)
    end

    @testset "Sampling variance" begin
        Random.seed!(1234)
        κ = 2.5
        vth_perp = 800.0
        vth_para = 1200.0
        d = BiKappa(vth_perp, vth_para, κ)
        samples = rand(d, 20000)

        # Per-component theoretical variance for the underlying 1D kappa = vth^2 * κ / (2κ - 3)
        theoretical_var_perp = (vth_perp^2 * κ) / (2 * κ - 3)
        theoretical_var_para = (vth_para^2 * κ) / (2 * κ - 3)

        # b0 defaults to z, so x/y are ⟂ and z is ∥
        empirical_var_x = var(getindex.(samples, 1))
        empirical_var_y = var(getindex.(samples, 2))
        empirical_var_z = var(getindex.(samples, 3))

        @test empirical_var_x ≈ theoretical_var_perp rtol = 5.0e-2
        @test empirical_var_y ≈ theoretical_var_perp rtol = 5.0e-2
        @test empirical_var_z ≈ theoretical_var_para rtol = 5.0e-2
    end
end
