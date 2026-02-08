module VelocityDistributionFunctions

using Tullio: @tullio
using Bumper: @alloc, @no_escape
using LinearAlgebra
using StaticArrays
using MuladdMacro: @muladd
using Base: tail
using StructArrays: StructArray
using OhMyThreads: tforeach
using Statistics: median

include("utils.jl")
include("spectra.jl")
include("pad.jl")
include("flux.jl")
include("moments.jl")
include("spectrograms.jl")
include("distributions/Distributions.jl")

export pitch_angle_distribution, tpitch_angle_distribution
export directional_energy_spectra
export plasma_moments
export VelocityDistribution, KappaDistribution, BiMaxwellian, BiKappa, Maxwellian, Kappa
export AbstractVelocityPDF, AbstractVelocityDistribution
export MaxwellianPDF, BiMaxwellianPDF, KappaPDF, BiKappaPDF, ShiftedPDF
export pdf
export kappa_thermal_speed
export VPar, VPerp
export energy_spectrogram, theta_spectrogram, phi_spectrogram
export pitch_angle_spectrogram, gyrophase_spectrogram

end
