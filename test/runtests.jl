using VelocityDistributionFunctions
using Test
using Aqua

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(VelocityDistributionFunctions)
end

include("test_angles.jl")
include("test_omega_weights.jl")
include("test_elfin_spectra.jl")
include("test_moments.jl")
include("test_gyroaverage.jl")
