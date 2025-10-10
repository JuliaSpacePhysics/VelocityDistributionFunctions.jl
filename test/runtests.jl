using VelocityDistributionFunctions
using Test
using Aqua

@testset "VelocityDistributionFunctions.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(VelocityDistributionFunctions)
    end

    @testset "pitch_angle_distribution" begin
        @testset "default bins mean" begin
            flux = fill(Float64(NaN), 1, 1, 1, 3)
            flux[1, 1, 1, 1] = 2.0
            flux[1, 1, 1, 2] = 4.0
            flux[1, 1, 1, 3] = 6.0
            B = [0.0, 0.0, -1.0]
            phi = [0.0]
            theta = [0.0, 90.0, 180.0]

            pad = pitch_angle_distribution(flux, B, phi, theta)
            data = pad.data

            @test size(data) == (1, 12, 1)
            @test data[1, 1, 1] == 2.0
            @test data[1, 6, 1] == 4.0
            @test data[1, 12, 1] == 6.0

            for idx in setdiff(collect(1:12), [1, 6, 12])
                @test isnan(data[1, idx, 1])
            end

            counts = pad.counts
            @test counts[1, 1, 1] == 1
            @test counts[1, 6, 1] == 1
            @test counts[1, 12, 1] == 1
        end

        @testset "custom edges sum" begin
            flux = ones(Float64, 1, 1, 2, 2)
            B = [0.0, 0.0, -1.0]
            phi = [0.0, 180.0]
            theta = [0.0, 180.0]
            edges = [0.0, 90.0, 180.0]

            pad = pitch_angle_distribution(flux, B, phi, theta; angles = edges, method = :sum)
            data = pad.data
            @test size(data) == (1, 2, 1)
            @test data[1, 1, 1] == 2.0
            @test data[1, 2, 1] == 2.0
            @test pad.counts[1, 1, 1] == 2
            @test pad.counts[1, 2, 1] == 2
            @test pad.edges == edges
        end

        @testset "time dependent inputs" begin
            flux = ones(Float64, 2, 1, 2, 2)
            B = [0.0 0.0 -1.0; 0.0 0.0 1.0]
            phi = reshape([0.0, 180.0], 1, :)
            theta = reshape([0.0, 180.0], 1, :)

            pad = pitch_angle_distribution(flux, B, phi, theta; angles = 2, method = :mean)
            data = pad.data
            counts = pad.counts

            @test size(data) == (2, 2, 1)
            @test all(data[1, :, 1] .== 1.0)
            @test all(data[2, :, 1] .== 1.0)
            @test all(counts .== 2)
            @test pad.edges == collect(0.0:90.0:180.0)
        end
    end
end
