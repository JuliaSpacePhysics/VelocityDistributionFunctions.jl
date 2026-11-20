module VelocityDistributionFunctions

using Bumper: @alloc, @no_escape
using LinearAlgebra
using PlasmaPopulations: Particle, Proton, Electron, E_SI
using StaticArrays: SVector, SMatrix, SA, @SMatrix
using Base: tail
using StructArrays: StructArray

include("unit_conversion.jl")
include("spectra.jl")
include("flux.jl")
include("moments.jl")
include("angles.jl")

export angle_distribution, omni_spectrum
export directional_spectra
export plasma_moments, EnergyAngleGrid, convert_flux_units

@static if VERSION >= v"1.11.0-DEV.469"
    eval(Meta.parse("public pitch_angle_spectra"))
end

end
