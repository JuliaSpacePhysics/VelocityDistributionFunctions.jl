using Test
using VelocityDistributionFunctions
using LinearAlgebra
using PlasmaPopulations: BiMaxwellian, Proton, pdf, MP_SI, ME_SI, E_SI

@testset "gyroaverage" begin
    n, vd = 5.0e6, 300.0e3                  # m⁻³, m/s
    b̂ = normalize([1.0, 2.0, 2.0])
    E = exp10.(range(1, 4; length = 64))
    grid = EnergyAngleBins(:energy => E, :phi => (0:63) .* 5.625 .+ 2.8125, :theta => (0:31) .* 5.625 .+ 2.8125)
    vel(e, φ, θ) = sqrt(2e * E_SI / MP_SI) .* [sind(θ) * cosd(φ), sind(θ) * sind(φ), cosd(θ)]
    function psd(vdf, u)
        return [
            begin
                    v = vel(e, φ, θ) - u
                    vp = v ⋅ b̂
                    n * pdf(vdf, sqrt(max(v ⋅ v - vp^2, 0)), vp)
                end for e in E, φ in vec(grid.phi), θ in vec(grid.theta)
        ]
    end
    kw = (; species = :H, units = :df_m, B = b̂)

    @testset "bi-Maxwellian in the plasma frame" begin
        u = vd .* b̂
        vdf = BiMaxwellian(Proton; T_perp = 20.0, T_para = 40.0)
        f = psd(vdf, u)
        r = gyroaverage(f, grid, :v_perp => 0:10.0e3:600.0e3, :v_para => -600.0e3:10.0e3:600.0e3; kw..., u)
        w = replace(r.data .* r.volume, NaN => 0.0)
        vperp, vpara = r.centers
        m = plasma_moments(f, grid; species = :H, units = :df_m)
        @test sum(w) ≈ m.density rtol = 0.005
        ref = [n * pdf(vdf, a, b) for a in vperp, b in vpara]
        core = (vperp .< 60.0e3) .& (abs.(vpara') .< 60.0e3)
        @test r.data[core] ≈ ref[core] rtol = 0.05
    end

    @testset "drifting Maxwellian is isotropic in the plasma frame" begin
        u = [vd, 0.0, 0.0]
        f = psd(BiMaxwellian(Proton; T = 20.0), u)
        pad = gyroaverage(f, grid, :energy => [10, 20, 40], :pitch => 6; kw..., u).data
        @test all(row -> maximum(row) / minimum(row) < 1.1, eachrow(pad))
    end

    @testset "partial field of view in a moving frame" begin
        g = EnergyAngleBins(:phi => [90., 270.], :theta => [85.], :energy => [100.]; dtheta = 50., denergy = 2.)
        # With B along z, w∥ = -u_z divides the shell at the detector equator.
        r = gyroaverage(reshape([1., NaN], 2, 1, 1), g, :v_perp => [0., 300e3], :v_para => [-250e3, -50e3, 150e3];
            species = :H, units = :df_m, B = [0, 0, 1], u = [70e3, 0., 50e3], oversample = 8)
        radial = (2 * E_SI / MP_SI)^1.5 * (101^1.5 - 99^1.5) / 3
        @test vec(r.volume) ≈ π * radial .* [-cosd(110), cosd(60)] rtol = 0.05
    end

    @testset "electron shell across a target boundary" begin
        shell = EnergyAngleBins(:energy => [10.0]; denergy = 2.0)
        opts = (; species = :e, units = :df_m, B = [0, 0, 1])
        # At +5 V the measured 9–11 eV channel spans 4–6 eV. At +12 V it disappears.
        e = [0., 5., 7., 12.]
        r = gyroaverage([[2.], [2.]], shell, :pitch => 1, :energy => e;
            opts..., B = [opts.B, opts.B], sc_pot = [5., 12.])
        scale = sqrt(2 * E_SI / ME_SI)
        expected = (4π / 3) * scale^3 .* [5^1.5 - 4^1.5, 6^1.5 - 5^1.5, 0.]
        @test r.volume[1, :, 1] ≈ expected
        @test isequal(r.data[1, :, 1], [2., 2., NaN])
        @test all(iszero, r.volume[:, :, 2])
        speed = gyroaverage([2.], shell, :speed => scale .* sqrt.(e), :pitch => 1; opts..., sc_pot = 5., u = zeros(3))
        @test vec(speed.volume) ≈ expected
        @test_throws ArgumentError gyroaverage([2.], shell, :energy => e, :v_perp => e; opts...)
        @test_throws ArgumentError gyroaverage([2.], shell, :energy => [0., 5., 5.], :pitch => 1; opts...)
        @test_throws ArgumentError gyroaverage([2.], shell, :energy => e, :pitch => 0; opts...)
    end
end
