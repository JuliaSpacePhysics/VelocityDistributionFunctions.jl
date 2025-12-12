module VelocityDistributionFunctions

using Tullio: @tullio
using Bumper
using LoopVectorization
using LinearAlgebra
using StaticArrays

include("utils.jl")
include("spectra.jl")
include("pad.jl")
include("flux.jl")
include("distributions/Distributions.jl")

export pitch_angle_distribution, tpitch_angle_distribution
export directional_energy_spectra
export VelocityDistribution, KappaDistribution, BiMaxwellian, BiKappa, Maxwellian, Kappa
export pdf
export kappa_thermal_speed
export VPar, VPerp

end
