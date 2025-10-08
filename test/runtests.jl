using VelocityDistributionFunctions
using Test
using Aqua

@testset "VelocityDistributionFunctions.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(VelocityDistributionFunctions)
    end
    # Write your tests here.
end
