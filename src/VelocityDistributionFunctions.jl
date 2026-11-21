module VelocityDistributionFunctions

using Bumper: @alloc, @no_escape
using LinearAlgebra
using PlasmaPopulations: Particle, Proton, Electron, E_SI
using StaticArrays: SVector, SMatrix, SA, @SMatrix
using Base: tail
using StructArrays: StructArray

include("bins.jl")
include("unit_conversion.jl")
include("series.jl")
include("spectra.jl")
include("moments.jl")
include("angles.jl")
include("gyroaverage.jl")

export angle_distribution, omni_spectrum
export directional_spectra, group_mean
export plasma_moments, EnergyAngleBins, convert_flux_units
export gyroaverage

end
