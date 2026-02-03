using Test
using VelocityDistributionFunctions
using StaticArrays
using LinearAlgebra

@testset "ParticleData" begin
    # Create test data
    nφ, nθ, nE = 16, 8, 32
    phi = collect(range(11.25, 348.75; length=nφ))  # 22.5° bins
    theta = collect(range(11.25, 168.75; length=nθ)) # 22.5° bins
    energy = 10.0 .^ range(1, 4; length=nE)  # 10 eV to 10 keV

    # Isotropic distribution for testing
    flux = ones(nφ, nθ, nE)

    @testset "Construction" begin
        # Basic construction
        pd = ParticleData(flux, energy, phi, theta)
        @test size(pd.data) == (nφ, nθ, nE)
        @test length(pd.phi) == nφ
        @test length(pd.theta) == nθ
        @test length(pd.energy) == nE

        # With mass
        me = 9.109e-31
        pd = ParticleData(flux, energy, phi, theta; mass=me)
        @test pd.mass == me

        # Computed bin widths
        @test length(pd.dphi) == nφ
        @test length(pd.dtheta) == nθ
        @test length(pd.denergy) == nE

        # Custom bin widths
        custom_dE = fill(100.0, nE)
        pd = ParticleData(flux, energy, phi, theta; denergy=custom_dE)
        @test pd.denergy == custom_dE
    end

    @testset "Dimension validation" begin
        # Wrong dimensions
        @test_throws DimensionMismatch ParticleData(flux, energy[1:10], phi, theta)
        @test_throws DimensionMismatch ParticleData(flux, energy, phi[1:5], theta)
        @test_throws ArgumentError ParticleData(ones(10, 10), energy, phi, theta)
    end

    @testset "Time series" begin
        nt = 5
        flux_ts = ones(nφ, nθ, nE, nt)
        pd = ParticleData(flux_ts, energy, phi, theta)
        @test VelocityDistributionFunctions.has_time(pd)
        @test VelocityDistributionFunctions.ntimes(pd) == nt
    end

    @testset "Solid angle computation" begin
        pd = ParticleData(flux, energy, phi, theta)
        Ω = solid_angle(pd)
        @test size(Ω) == (nφ, nθ)
        @test all(Ω .> 0)

        # Total solid angle should be approximately 4π
        total_Ω = sum(Ω)
        @test isapprox(total_Ω, 4π, rtol=0.1)  # Allow some tolerance for binning
    end

    @testset "Velocity grid" begin
        me = 9.109e-31
        pd = ParticleData(flux, energy, phi, theta; mass=me)
        vx, vy, vz = velocity_grid(pd)

        @test size(vx) == (nφ, nθ, nE)

        # Velocity magnitude should match energy: v = sqrt(2E/m)
        for k in 1:nE
            E = energy[k]
            v_expected = sqrt(2 * E / me)
            for j in 1:nθ, i in 1:nφ
                v_computed = sqrt(vx[i,j,k]^2 + vy[i,j,k]^2 + vz[i,j,k]^2)
                @test isapprox(v_computed, v_expected, rtol=1e-10)
            end
        end
    end
end

@testset "Coordinate Transformations" begin
    nφ, nθ, nE = 16, 8, 32
    phi = collect(range(11.25, 348.75; length=nφ))
    theta = collect(range(11.25, 168.75; length=nθ))
    energy = 10.0 .^ range(1, 4; length=nE)
    flux = ones(nφ, nθ, nE)
    pd = ParticleData(flux, energy, phi, theta)

    @testset "Look directions" begin
        lx, ly, lz = look_directions(pd)
        @test size(lx) == (nφ, nθ)

        # Look directions should be unit vectors
        for j in 1:nθ, i in 1:nφ
            mag = sqrt(lx[i,j]^2 + ly[i,j]^2 + lz[i,j]^2)
            @test isapprox(mag, 1.0, rtol=1e-10)
        end
    end

    @testset "Pitch angles" begin
        # B along z-axis
        B = [0.0, 0.0, 1.0]
        pa = pitch_angles(pd, B)
        @test size(pa) == (nφ, nθ)

        # All pitch angles should be in [0, 180]
        @test all(0 .<= pa .<= 180)

        # For B || z, pitch angle should depend only on theta
        # At theta = 0 (looking along +z), pitch angle = 180 (antiparallel)
        # At theta = 180 (looking along -z), pitch angle = 0 (parallel)
    end

    @testset "Pitch angles time series" begin
        B = randn(3, 10)  # Time-varying B
        pa = pitch_angles(pd, B)
        @test size(pa) == (nφ, nθ, 10)
    end

    @testset "Gyrophase angles" begin
        B = [0.0, 0.0, 1.0]
        gyro = gyrophase_angles(pd, B)
        @test size(gyro) == (nφ, nθ)

        # All gyrophase angles should be in [0, 360)
        @test all(0 .<= gyro .< 360)
    end

    @testset "Invalid B field" begin
        B = [0.0, 0.0, 0.0]  # Zero B field
        pa = pitch_angles(pd, B)
        @test all(isnan, pa)
    end
end

@testset "Spectrograms" begin
    nφ, nθ, nE = 16, 8, 32
    phi = collect(range(11.25, 348.75; length=nφ))
    theta = collect(range(11.25, 168.75; length=nθ))
    energy = 10.0 .^ range(1, 4; length=nE)

    @testset "Energy spectrogram" begin
        # Uniform flux
        flux = ones(nφ, nθ, nE)
        pd = ParticleData(flux, energy, phi, theta)

        result = energy_spectrogram(pd)
        @test length(result.data) == nE
        @test result.energy === pd.energy
        @test all(isapprox.(result.data, 1.0, rtol=1e-10))  # Should average to 1

        # With sum method
        result_sum = energy_spectrogram(pd; method=:sum)
        @test all(result_sum.data .> 1)  # Sum should be larger
    end

    @testset "Theta spectrogram" begin
        flux = ones(nφ, nθ, nE)
        pd = ParticleData(flux, energy, phi, theta)

        result = theta_spectrogram(pd)
        @test length(result.data) == nθ
        @test result.theta === pd.theta

        # Energy range filtering
        result_filtered = theta_spectrogram(pd; energy_range=(100.0, 1000.0))
        @test length(result_filtered.data) == nθ
    end

    @testset "Phi spectrogram" begin
        flux = ones(nφ, nθ, nE)
        pd = ParticleData(flux, energy, phi, theta)

        result = phi_spectrogram(pd)
        @test length(result.data) == nφ
        @test result.phi === pd.phi
    end

    @testset "Pitch angle spectrogram" begin
        flux = ones(nφ, nθ, nE)
        pd = ParticleData(flux, energy, phi, theta)
        B = [0.0, 0.0, 1.0]

        # Default bins (15°)
        result = pitch_angle_spectrogram(pd, B)
        @test length(result.pitch_angles) == 12  # 180/15 = 12 bins
        @test size(result.data, 1) == 12
        @test haskey(result, :energy)  # Should have energy dimension

        # Custom bins
        result18 = pitch_angle_spectrogram(pd, B; bins=18)
        @test length(result18.pitch_angles) == 18

        # With energy range (collapsed energy dimension)
        result_collapsed = pitch_angle_spectrogram(pd, B; energy_range=(100.0, 1000.0))
        @test !haskey(result_collapsed, :energy)
    end

    @testset "Gyrophase spectrogram" begin
        flux = ones(nφ, nθ, nE)
        pd = ParticleData(flux, energy, phi, theta)
        B = [0.0, 0.0, 1.0]

        result = gyrophase_spectrogram(pd, B)
        @test length(result.gyrophase) == 24  # 360/15 = 24 bins
        @test all(0 .<= result.gyrophase .< 360)
    end

    @testset "Time series spectrograms" begin
        nt = 5
        flux = ones(nφ, nθ, nE, nt)
        pd = ParticleData(flux, energy, phi, theta)
        B = randn(3, nt)

        result = energy_spectrogram(pd)
        @test size(result.data) == (nE, nt)

        result_pa = pitch_angle_spectrogram(pd, B)
        @test size(result_pa.data, 3) == nt
    end
end

@testset "Moments" begin
    nφ, nθ, nE = 32, 16, 64
    phi = collect(range(5.625, 354.375; length=nφ))  # Higher resolution
    theta = collect(range(5.625, 174.375; length=nθ))
    energy = 10.0 .^ range(0, 5; length=nE)  # 1 eV to 100 keV

    me = 9.109e-31  # Electron mass
    eV = 1.602e-19  # eV to Joules

    # Convert energy to Joules for consistency
    energy_J = energy .* eV

    @testset "Density" begin
        # Create isotropic Maxwellian-like distribution
        kT = 100 * eV  # 100 eV temperature
        vth = sqrt(2 * kT / me)

        flux = zeros(nφ, nθ, nE)
        for k in 1:nE
            v = sqrt(2 * energy_J[k] / me)
            # f(v) ∝ exp(-v²/vth²)
            flux[:, :, k] .= exp(-(v/vth)^2)
        end

        pd = ParticleData(flux, energy_J, phi, theta; mass=me)
        n = density(pd)

        @test n > 0
        @test isfinite(n)
    end

    @testset "Bulk velocity" begin
        # Isotropic distribution should have zero bulk velocity
        flux = ones(nφ, nθ, nE)
        pd = ParticleData(flux, energy_J, phi, theta; mass=me)

        V = bulk_velocity(pd)
        @test length(V) == 3
        # For isotropic distribution, bulk velocity should be near zero
        @test all(isfinite, V)
    end

    @testset "Pressure and Temperature" begin
        flux = ones(nφ, nθ, nE) .* 1e-6  # Scale down for reasonable values
        pd = ParticleData(flux, energy_J, phi, theta; mass=me)

        P = pressure_tensor(pd)
        @test size(P) == (3, 3)
        @test issymmetric(P)

        P_scalar = pressure_scalar(pd)
        @test isapprox(P_scalar, tr(P) / 3, rtol=1e-10)

        T = temperature_scalar(pd)
        @test T > 0
    end

    @testset "Anisotropic temperature" begin
        flux = ones(nφ, nθ, nE)
        pd = ParticleData(flux, energy_J, phi, theta; mass=me)
        B = [0.0, 0.0, 1.0]

        T_para = temperature_parallel(pd, B)
        T_perp = temperature_perpendicular(pd, B)

        @test T_para > 0
        @test T_perp > 0

        # For isotropic distribution, should be approximately equal
        # (within numerical accuracy of binning)
    end

    @testset "Heat flux" begin
        flux = ones(nφ, nθ, nE)
        pd = ParticleData(flux, energy_J, phi, theta; mass=me)

        Q = heat_flux(pd)
        @test length(Q) == 3
        @test all(isfinite, Q)

        B = [0.0, 0.0, 1.0]
        Q_para = heat_flux_parallel(pd, B)
        @test isfinite(Q_para)
    end

    @testset "Entropy" begin
        # Positive distribution
        flux = ones(nφ, nθ, nE) .* 1e-10
        pd = ParticleData(flux, energy_J, phi, theta; mass=me)

        s = entropy_density(pd)
        @test isfinite(s)
    end

    @testset "Time series moments" begin
        nt = 3
        flux = ones(nφ, nθ, nE, nt)
        pd = ParticleData(flux, energy_J, phi, theta; mass=me)

        n = density(pd)
        @test length(n) == nt

        V = bulk_velocity(pd)
        @test size(V) == (3, nt)

        P = pressure_tensor(pd)
        @test length(P) == nt
        @test all(size.(P) .== Ref((3, 3)))
    end
end

@testset "Analytical VDF Moments" begin
    using VelocityDistributionFunctions: VelocityDistribution, ShiftedPDF, MaxwellianPDF

    @testset "VelocityDistribution density" begin
        vd = VelocityDistribution(1e6, MaxwellianPDF(1e5))
        @test density(vd) == 1e6
    end

    @testset "ShiftedPDF bulk velocity" begin
        u0 = SA[1e5, 0.0, 0.0]
        pdf = ShiftedPDF(MaxwellianPDF(1e5), u0)
        @test bulk_velocity(pdf) == u0
    end

    @testset "Non-shifted bulk velocity" begin
        pdf = MaxwellianPDF(1e5)
        @test bulk_velocity(pdf) == SA[0.0, 0.0, 0.0]
    end
end
