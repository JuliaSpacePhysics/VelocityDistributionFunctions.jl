using VelocityDistributionFunctions
using Test
using Aqua

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(VelocityDistributionFunctions)
end

include("test_pad.jl")
