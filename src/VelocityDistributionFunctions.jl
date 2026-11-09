module VelocityDistributionFunctions

using Bumper: @alloc, @no_escape
using LinearAlgebra
using StaticArrays: SVector, SMatrix, SA, @SMatrix
using MuladdMacro: @muladd
using Base: tail
using StructArrays: StructArray

include("unit_conversion.jl")
include("spectra.jl")
include("pad.jl")
include("flux.jl")
include("moments.jl")

export pitch_angle_distribution, tpitch_angle_distribution
export directional_energy_spectra
export plasma_moments, tmoments

@static if VERSION >= v"1.11.0-DEV.469"
    eval(Meta.parse("public pitch_angle_spectra, sort_flux_by_pitch_angle!, is_parallel, is_anti_parallel, is_perpendicular"))
end

end
